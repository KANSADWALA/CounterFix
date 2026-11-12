"""``guard()`` for OpenAI-compatible clients.

Works with the official ``openai`` SDK and with any provider that exposes the same
``client.chat.completions.create(...)`` shape: OpenAI, Groq, Ollama, vLLM, Together,
OpenRouter and others. The wrapped client keeps the same API; each response gains a
``.diagnosis`` attribute (or use :func:`counterfix.get_diagnosis`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from counterfix.core.config import Config
from counterfix.core.convert import record_from_openai as record_from_openai
from counterfix.core.detector import Detector
from counterfix.integrations.openai_compat.guarded import (
    _ANTHROPIC_PATHS as _ANTHROPIC_PATHS,
    _CLIENT_COPIES as _CLIENT_COPIES,
    _CREATE_PATH as _CREATE_PATH,
    _OPENAI_PATHS as _OPENAI_PATHS,
    RETRY_LIMIT_KEYS as RETRY_LIMIT_KEYS,
    RETRY_MIN_LIMIT as RETRY_MIN_LIMIT,
    GuardedClient as GuardedClient,
    _Guard as _Guard,
    logger as logger,
)

ClientT = TypeVar("ClientT")


def settings(
    config: Config | None = None,
    *,
    mode: Literal["flag", "retry"] | None = None,
    log: str | Path | None = None,
    judge: Any = None,
    sample_rate: float | None = None,
) -> Config:
    """The :class:`Config` for ``guard`` and ``watch``: ``config`` (or the defaults) with the
    keyword shortcuts applied, so common setups need no Config object.

    ``mode="retry"`` re-runs a flagged call once with a known fix; ``log`` appends every call
    to a JSONL file (after any ``on_diagnosis`` hook already set); ``judge`` picks the judge
    for background checks and ``sample_rate`` the share of unflagged calls it checks.
    """
    base = config if config is not None else Config()
    update: dict[str, Any] = {}
    if mode is not None:
        update["mode"] = mode
    if log is not None:
        from counterfix.core.config import hooks
        from counterfix.runtime.report import log_to

        update["on_diagnosis"] = hooks(base.on_diagnosis, log_to(log))
    if judge is not None:
        update["judge"] = judge
    if sample_rate is not None:
        update["sample_rate"] = sample_rate
    if not update:
        return base
    return Config.model_validate({**dict(base), **update})


def guard(
    client: ClientT,
    *,
    config: Config | None = None,
    detectors: Sequence[Detector] | None = None,
    provider: str | None = None,
    mode: Literal["flag", "retry"] | None = None,
    log: str | Path | None = None,
    judge: Any = None,
    sample_rate: float | None = None,
) -> ClientT:
    """Wrap an LLM client so every call is checked: an OpenAI-compatible client
    (``chat.completions.create``) or an Anthropic client (``messages.create``), sync or async.

    >>> from openai import OpenAI          # doctest: +SKIP
    >>> client = guard(OpenAI())           # doctest: +SKIP
    >>> r = client.chat.completions.create(model="gpt-4o-mini", messages=[...])  # doctest: +SKIP
    >>> r.diagnosis.summary()              # doctest: +SKIP

    Common settings need no :class:`Config`: ``guard(client, mode="retry")`` re-runs a
    flagged call once with a known fix, ``log="calls.jsonl"`` records every call, and
    ``judge=...`` with ``sample_rate=0.05`` adds background judge checks. A ``config`` and
    these shortcuts can be combined; the shortcuts win.

    ``detectors`` defaults to the built-in live detectors. ``provider`` labels each record
    (e.g. "groq"); when not given it is read from the client's base URL where known, so
    ``fix()`` and exported regression tests can re-run the call later. The result is typed
    as the client you passed, so editors keep their completions.
    """
    shortcuts = {"mode": mode, "log": log, "judge": judge, "sample_rate": sample_rate}
    if isinstance(client, GuardedClient):
        unchanged = config is None and detectors is None and provider is None
        if unchanged and all(v is None for v in shortcuts.values()):
            return client
        client = client.unwrapped  # re-wrap with the new settings
    if isinstance(client, type):
        raise TypeError(
            f"guard() needs a client instance: guard({client.__name__}()), "
            f"not guard({client.__name__})"
        )
    if not any(hasattr(client, a) for a in ("chat", "messages", "responses")):
        raise TypeError(
            f"guard() wraps an OpenAI-compatible or Anthropic client; {type(client).__name__} "
            "has no chat, responses or messages API. For your own function use "
            "@counterfix.watch; for LiteLLM or LangChain use counterfix.litellm_callback() or "
            "counterfix.langchain_callback()."
        )
    config = settings(config, **shortcuts)
    provider = provider or provider_from_url(getattr(client, "base_url", None))
    if detectors is None:
        from counterfix.detect import default_detectors

        detectors = default_detectors()
    if not hasattr(client, "chat") and hasattr(client, "messages"):
        from counterfix.integrations.anthropic import ANTHROPIC_RETRY_PARAMS, record_from_anthropic

        state = _Guard(
            config,
            detectors,
            provider or "anthropic",
            to_record=record_from_anthropic,
            create_paths=_ANTHROPIC_PATHS,
            retry_params=ANTHROPIC_RETRY_PARAMS,
        )
        state.client = client
    else:
        from counterfix.integrations.responses import (
            RESPONSES_PATHS,
            RESPONSES_RETRY_PARAMS,
            record_from_responses,
            responses_changes,
        )

        state = _Guard(config, detectors, provider)
        state.client = client
        if hasattr(client, "responses"):
            state.sibling(
                to_record=record_from_responses,
                create_paths=RESPONSES_PATHS,
                retry_params=RESPONSES_RETRY_PARAMS,
                stream_kind="responses",
                translate=responses_changes,
            )
    return cast(ClientT, GuardedClient(client, state))


_HOSTS = {
    "api.openai.com": "openai",
    "api.groq.com": "groq",
    "generativelanguage.googleapis.com": "gemini",
    "openrouter.ai": "openrouter",
    "localhost:11434": "ollama",
    "127.0.0.1:11434": "ollama",
}


def provider_from_url(base_url: Any) -> str | None:
    """'https://api.groq.com/openai/v1' -> 'groq'; None for hosts we have no preset for."""
    text = str(base_url or "")
    return next((name for host, name in _HOSTS.items() if host in text), None)
