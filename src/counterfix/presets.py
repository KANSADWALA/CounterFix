"""Ready-made settings for OpenAI-compatible providers, and helpers to build callers and
judges from a short spec such as ``groq:openai/gpt-oss-20b``. Needs the ``openai`` extra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from counterfix.core.caller import OpenAICompatibleCaller

# provider -> (base_url or None for the SDK default, API key variable or None, default model)
PRESETS: dict[str, tuple[str | None, str | None, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.5-flash-lite",
    ),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:3b"),
    "openai": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-20b:free"),
}
DEFAULT_TIMEOUT = {"gemini": 120.0}


def load_env_file(path: str | Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ. Values already set win;
    blank values and comments are ignored. Dependency-free on purpose."""
    path = Path(path)
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


def make_client(provider: str, *, timeout: float | None = None, max_retries: int = 1) -> Any:
    """An ``openai.OpenAI`` client for a preset provider. Raises if the key is missing."""
    if provider not in PRESETS:
        raise ValueError(f"unknown provider {provider!r}; choose from {', '.join(PRESETS)}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError('this needs the OpenAI SDK: pip install "counterfix[openai]"') from exc
    base_url, key_var, _ = PRESETS[provider]
    api_key = os.environ.get(key_var, "") if key_var else "ollama"
    if key_var and not api_key:
        raise RuntimeError(f"set {key_var} in your environment or .env file")
    options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout or DEFAULT_TIMEOUT.get(provider, 60.0),
        "max_retries": max_retries,
    }
    return OpenAI(base_url=base_url, **options) if base_url else OpenAI(**options)


def parse_spec(spec: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-20b' -> ('groq', 'openai/gpt-oss-20b'); 'groq' -> default model."""
    provider, _, model = spec.partition(":")
    if provider not in PRESETS:
        raise ValueError(f"unknown provider in {spec!r}; choose from {', '.join(PRESETS)}")
    return provider, model or PRESETS[provider][2]


def make_caller(spec: str, *, timeout: float | None = None) -> OpenAICompatibleCaller:
    """A ModelCaller from a 'provider:model' spec."""
    provider, model = parse_spec(spec)
    return OpenAICompatibleCaller(make_client(provider, timeout=timeout), model, provider=provider)
