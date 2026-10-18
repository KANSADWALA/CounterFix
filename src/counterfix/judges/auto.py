"""Choosing a judge: bring your own key.

counterfix never ships API keys. Live checks need none (they wrap your client and make no
calls of their own). Judge checks need a model to act as judge, and counterfix picks one
in this order:

1. the ``judge=`` you pass to :func:`counterfix.diagnose` (a judge, a spec or a list);
2. :func:`configure` (``counterfix.configure(judge="groq:qwen/qwen3.8-27b")``);
3. the ``COUNTERFIX_JUDGE`` environment variable (comma-separated specs make a panel);
4. a key found in the environment: GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY,
   OPENROUTER_API_KEY, or a model in a local Ollama; a different model family from the one
   being checked is preferred;
5. the model being checked judges itself, with a :class:`SelfJudgeWarning`, because
   judges are lenient with their own mistakes.

If nothing is available, :class:`NoJudgeError` explains how to get a free judge.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import warnings
from collections.abc import Sequence
from typing import Any, cast

from counterfix.judges.base import Judge

ENV_VAR = "COUNTERFIX_JUDGE"
KEY_ORDER = ("groq", "gemini", "openai", "openrouter")
FAMILIES = (
    "qwen",
    "gemma",
    "gemini",
    "gpt",
    "llama",
    "mistral",
    "mixtral",
    "claude",
    "deepseek",
    "phi",
    "allam",
    "kimi",
    "granite",
    "command",
)

NO_JUDGE_HELP = """Judge checks need a model to act as the judge, and none was found.
Any one of these works:
  * a free Groq key:   set GROQ_API_KEY      (https://console.groq.com/keys)
  * a free Gemini key: set GEMINI_API_KEY    (https://aistudio.google.com/apikey)
  * a local model:     install Ollama (https://ollama.com), then: ollama pull qwen2.5:7b
  * any other model:   counterfix.configure(judge="provider:model")
                       or set COUNTERFIX_JUDGE=provider:model
Run `counterfix setup` to see what counterfix can find on this machine."""

JudgeChoice = Judge | str | Sequence[str]
_configured: JudgeChoice | None = None


class NoJudgeError(RuntimeError):
    """No judge was given, configured or found."""


class SelfJudgeWarning(UserWarning):
    """The model being checked is also its own judge (less reliable)."""


def configure(*, judge: JudgeChoice | None = None) -> None:
    """Set the judge counterfix uses for judge checks, once per process.

    ``judge`` is a spec such as ``"groq:qwen/qwen3.8-27b"`` or ``"ollama:qwen2.5:7b"``, a
    list of specs (a voting panel; three model families recommended), or any object with
    ``name`` and ``ask(task)``. ``None`` goes back to automatic detection.
    """
    global _configured
    if (
        judge is not None
        and not isinstance(judge, str | list | tuple)
        and not hasattr(judge, "ask")
    ):
        raise TypeError("judge must be a 'provider:model' spec, a list of specs, or a Judge")
    _configured = judge


def family(model: str) -> str:
    """'openai/gpt-oss-120b' -> 'gpt'; unknown models are their own family."""
    lowered = model.lower()
    for name in FAMILIES:
        if re.search(rf"(?<![a-z]){name}", lowered):
            return name
    return lowered


def ollama_models(timeout: float = 0.5) -> list[str]:
    """Chat models in a running local Ollama, or [] if Ollama is not running."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    names = [str(m.get("name", "")) for m in data.get("models", []) if isinstance(m, dict)]
    return [n for n in names if n and "embed" not in n.lower()]


def detect_judges(avoid_model: str | None = None) -> list[tuple[str, str]]:
    """Judges available on this machine as (spec, where it came from), best first.
    Models from the same family as ``avoid_model`` go last."""
    from counterfix.presets import JUDGE_MODELS, PRESETS

    found: list[tuple[str, str]] = []
    for provider in KEY_ORDER:
        key_var = PRESETS[provider][1]
        if key_var and os.environ.get(key_var):
            found.append((f"{provider}:{JUDGE_MODELS[provider]}", f"{key_var} is set"))
    for model in ollama_models()[:1]:
        found.append((f"ollama:{model}", "Ollama is running locally"))
    if avoid_model:
        avoid = family(avoid_model)
        found.sort(key=lambda item: family(item[0].split(":", 1)[1]) == avoid)
    return found


def build_judge(choice: JudgeChoice) -> Judge:
    """A judge from a spec ('groq:qwen/qwen3.8-27b', 'jev'), a list of specs, or a judge."""
    if not isinstance(choice, str | list | tuple):
        return cast(Judge, choice)
    if isinstance(choice, list | tuple):
        from counterfix.judges.llm import JudgePanel

        judges = [build_judge(spec) for spec in choice]
        return judges[0] if len(judges) == 1 else JudgePanel(judges)
    specs = [s.strip() for s in choice.split(",") if s.strip()]
    if len(specs) > 1:
        return build_judge(specs)
    spec = specs[0] if specs else choice
    if spec in ("jev", "typesafe:jev"):
        from counterfix.judges.jev import JevJudge

        return JevJudge()
    from counterfix.core.caller import RateLimitedCaller
    from counterfix.judges.llm import LLMJudge
    from counterfix.presets import judge_params, make_caller, rpm_for, supports_json_mode

    return LLMJudge(
        RateLimitedCaller(make_caller(spec), rpm_for(spec)),
        params=judge_params(spec),
        json_mode=supports_json_mode(spec),
    )


def resolve_judge(
    judge: JudgeChoice | None = None,
    *,
    target_model: str | None = None,
    fallback_client: Any = None,
    fallback_provider: str | None = None,
) -> tuple[Judge, str]:
    """Pick the judge (see the module docstring for the order). Returns (judge, source)."""
    if judge is not None:
        return build_judge(judge), "passed to diagnose()"
    if _configured is not None:
        return build_judge(_configured), "counterfix.configure()"
    if os.environ.get(ENV_VAR, "").strip():
        return build_judge(os.environ[ENV_VAR]), f"{ENV_VAR} is set"
    found = detect_judges(avoid_model=target_model)
    if found:
        spec, why = found[0]
        return build_judge(spec), why
    if fallback_client is not None and target_model:
        from counterfix.core.caller import OpenAICompatibleCaller
        from counterfix.judges.llm import LLMJudge

        warnings.warn(
            f"counterfix: no separate judge found, so {target_model} is judging its own "
            "answer. Judges are lenient with their own mistakes; set GROQ_API_KEY or "
            f"GEMINI_API_KEY (free) or {ENV_VAR} for an independent judge.",
            SelfJudgeWarning,
            stacklevel=3,
        )
        caller = OpenAICompatibleCaller(fallback_client, target_model, provider=fallback_provider)
        return LLMJudge(caller), "the checked model itself (no other judge found)"
    raise NoJudgeError(NO_JUDGE_HELP)
