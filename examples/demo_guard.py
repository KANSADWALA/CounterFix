"""Phase 1 demo: wrap an OpenAI-compatible client with counterfix.guard().

Keys are read from a .env file in the project folder (see .env.example), or from
environment variables already set. Then run one of:

    python examples/demo_guard.py groq      # needs GROQ_API_KEY
    python examples/demo_guard.py gemini    # needs GEMINI_API_KEY
    python examples/demo_guard.py ollama    # needs Ollama running and a pulled model

Choose a model with --model (or CF_MODEL in .env). To see which models your key
can use:

    python examples/demo_guard.py groq --list-models

Phase 1 ships no built-in detectors yet, so this demo adds one small custom detector
to show how detectors plug in. It flags responses cut off by the max_tokens limit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from counterfix import (
    CallRecord,
    Config,
    Evidence,
    FailureType,
    Finding,
    Prescription,
    PrescriptionKind,
    guard,
)

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.5-flash-lite",
    ),
    "ollama": ("http://localhost:11434/v1", None, "llama3.2:3b"),
    "openai": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
}


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Values already set in the environment win. Blank values and comments are ignored.
    Kept dependency-free on purpose.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


class CutOffExample:
    """Example custom detector: the response stopped because it hit max_tokens."""

    name = "cut_off_example"
    covers = (FailureType.TRUNCATION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if record.finish_reason != "length":
            return []
        limit = record.max_tokens
        if record.has_reasoning and not record.answer_text:
            detail = f"the model spent the whole max_tokens budget ({limit}) on reasoning"
            fix = Prescription(
                kind=PrescriptionKind.PARAM_CHANGE,
                description="Raise max_tokens, or lower reasoning effort if the provider allows it",
                patch={"max_tokens": (limit or 256) * 8, "reasoning_effort": "low"},
            )
        else:
            detail = f"the answer stopped at the max_tokens limit ({limit})"
            fix = Prescription(
                kind=PrescriptionKind.PARAM_CHANGE,
                description="Raise max_tokens, or ask for a shorter answer",
                patch={"max_tokens": (limit or 256) * 4},
            )
        return [
            Finding(
                failure_type=FailureType.TRUNCATION,
                confidence=0.95,
                evidence=[Evidence(detector=self.name, detail=detail)],
                prescription=fix,
            )
        ]


def run(client: Any, model: str) -> list[Any]:
    """Send two calls through the guarded client and print what counterfix saw."""
    seen: list[CallRecord] = []
    config = Config(on_diagnosis=lambda record, diagnosis: seen.append(record))
    guarded = guard(client, detectors=[CutOffExample()], config=config)
    calls = [
        ("short answer", {"max_tokens": 1024, "content": "Say hello in five words."}),
        ("cut off", {"max_tokens": 40, "content": "Write a detailed essay on monsoons."}),
    ]
    responses = []
    for label, spec in calls:
        print(f"\n[{label}] calling the model...", flush=True)
        response = guarded.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": spec["content"]}],
            max_tokens=spec["max_tokens"],
        )
        record = seen[-1] if seen else CallRecord(response_text=response.choices[0].message.content)
        text = record.answer_text.replace("\n", " ")
        seconds = f"{record.latency_ms / 1000:.1f}s" if record.latency_ms is not None else "?"
        thinking = " (model reasoned first)" if record.has_reasoning else ""
        print(f"[{label}] {seconds}, max_tokens={spec['max_tokens']}{thinking}")
        print(f"  answer: {text[:80] or '(empty)'}")
        print(f"  counterfix: {response.diagnosis.summary()}")
        top = response.diagnosis.top
        if top is not None:
            for evidence in top.evidence:
                print(f"  why: {evidence.detail}")
            if top.prescription is not None:
                print(f"  fix: {top.prescription.description} -> {top.prescription.patch}")
        responses.append(response)
    return responses


def list_models(client: Any) -> list[str]:
    """Model ids this key can use, skipping obvious non-chat models (speech, embeddings)."""
    skip = (
        "whisper",
        "tts",
        "embed",
        "guard",
        "moderation",
        "orpheus",
        "playai",  # speech, safety
        "image",
        "banana",
        "veo",
        "lyria",
        "imagen",  # image, video, music
        "live",
        "audio",
        "transcribe",
        "translate",
        "robotics",
        "computer-use",  # realtime, agents
        "aqa",
        "deep-research",
        "antigravity",
    )
    ids = sorted(m.id.removeprefix("models/") for m in client.models.list())
    return [i for i in ids if not any(word in i.lower() for word in skip)]


def is_model_not_found(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 404 or type(error).__name__ == "NotFoundError"


def main() -> None:
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description="counterfix guard() demo")
    parser.add_argument("provider", nargs="?", default="groq", choices=sorted(PROVIDERS))
    parser.add_argument("--model", help="model id (default: CF_MODEL or the provider default)")
    parser.add_argument("--list-models", action="store_true", help="list models and exit")
    parser.add_argument(
        "--timeout", type=float, help="seconds per call (default 60; 120 for gemini)"
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit('This demo needs the OpenAI SDK: pip install "counterfix[openai]"')

    base_url, key_var, default_model = PROVIDERS[args.provider]
    api_key = os.environ.get(key_var, "") if key_var else "ollama"
    if key_var and not api_key:
        sys.exit(f"Set {key_var} in the .env file (see .env.example) or the environment.")
    if args.timeout is None:
        args.timeout = 120.0 if args.provider == "gemini" else 60.0
    options: dict[str, Any] = {"api_key": api_key, "timeout": args.timeout, "max_retries": 1}
    client = OpenAI(base_url=base_url, **options) if base_url else OpenAI(**options)

    if args.list_models:
        print("\n".join(list_models(client)))
        return

    model = args.model or os.environ.get("CF_MODEL") or default_model
    print(f"provider={args.provider} model={model}")
    try:
        run(client, model)
    except Exception as error:
        if type(error).__name__ == "APITimeoutError":
            sys.exit(
                f"\nNo reply from {args.provider} within {args.timeout:.0f}s. The model may be "
                "overloaded or not on your free tier. Try another --model, or --timeout 120."
            )
        if type(error).__name__ == "RateLimitError":
            sys.exit(
                f"\nRate limited by {args.provider}: {error}. Wait a minute or try another --model."
            )
        if not is_model_not_found(error):
            raise
        print(f"\n{args.provider} returned 404 for model {model!r}.")
        print(f"Provider message: {getattr(error, 'message', None) or error}")
        print("Models this key can list (listing a model does not guarantee you can call it):")
        try:
            for model_id in list_models(client):
                print(f"  {model_id}")
        except Exception as list_error:
            print(f"  (could not list models: {list_error})")
        sys.exit("Re-run with --model <one of the above>.")


if __name__ == "__main__":
    main()
