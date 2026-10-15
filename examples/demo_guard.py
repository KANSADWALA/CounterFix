"""Phase 1 demo: wrap an OpenAI-compatible client with counterfix.guard().

Keys are read from a .env file in the project folder (see .env.example), or from
environment variables already set. Then run one of:

    python examples/demo_guard.py groq      # needs GROQ_API_KEY
    python examples/demo_guard.py gemini    # needs GEMINI_API_KEY
    python examples/demo_guard.py ollama    # needs Ollama running and a pulled model

Choose a model with --model (or CF_MODEL in .env). To see which models your key
can use:

    python examples/demo_guard.py groq --list-models

The guarded client runs counterfix's built-in live checks (truncation, format, refusal,
empty output, provider filter). Add --retry to let live mode re-run a failed call once
with a known fix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from counterfix import CallRecord, Config, guard
from counterfix.presets import PRESETS, load_env_file

PROVIDERS = PRESETS  # kept for backwards compatibility with earlier scripts


def run(client: Any, model: str, *, retry: bool = False) -> list[Any]:
    """Send two calls through the guarded client and print what counterfix saw."""
    seen: dict[str, CallRecord] = {}  # record id -> record, for every call incl. retries
    config = Config(
        mode="retry" if retry else "flag",
        on_diagnosis=lambda record, diagnosis: seen.update({record.id: record}),
    )
    guarded = guard(client, config=config)
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
        # The record behind the response actually returned (the retry's, if it was kept).
        record = seen.get(response.diagnosis.record_id) or CallRecord(
            response_text=response.choices[0].message.content
        )
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
        for attempt in response.diagnosis.retries:
            print(f"  retried with {attempt.changes}: {attempt.outcome}")
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
    parser.add_argument("--retry", action="store_true", help="let live mode retry once with a fix")
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
        run(client, model, retry=args.retry)
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
