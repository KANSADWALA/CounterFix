"""``guard()`` for OpenAI-compatible clients.

Works with the official ``openai`` SDK and with any provider that exposes the same
``client.chat.completions.create(...)`` shape: OpenAI, Groq, Ollama, vLLM, Together,
OpenRouter and others. The wrapped client keeps the same API; each response gains a
``.diagnosis`` attribute (or use :func:`counterfix.get_diagnosis`).
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from counterfix.adapters._attach import attach
from counterfix.core.config import Config
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord, Message, Usage

logger = logging.getLogger("counterfix")

_CREATE_PATH = ("chat", "completions", "create")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an object attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK objects to plain dicts."""
    if isinstance(obj, dict):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            result = dump()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    try:
        return dict(vars(obj))
    except TypeError:
        return {"value": repr(obj)}


def _usage(raw: Any) -> Usage:
    """Token usage, including reasoning tokens when the provider reports them."""
    usage = Usage.model_validate(_to_dict(raw))
    details = _get(raw, "completion_tokens_details")
    reasoning_tokens = _get(details, "reasoning_tokens")
    if isinstance(reasoning_tokens, int):
        usage.reasoning_tokens = reasoning_tokens
    return usage


def record_from_openai(
    request: dict[str, Any],
    response: Any,
    *,
    latency_ms: float | None = None,
    provider: str | None = None,
) -> CallRecord:
    """Build a :class:`CallRecord` from chat-completions kwargs and the response."""
    messages = [Message.model_validate(_to_dict(m)) for m in request.get("messages") or []]
    params = {k: v for k, v in request.items() if k not in ("messages", "model")}

    choices = _get(response, "choices") or []
    first = choices[0] if len(choices) > 0 else None
    message = _get(first, "message")
    raw_tool_calls = _get(message, "tool_calls")
    usage_raw = _get(response, "usage")
    usage = _usage(usage_raw) if usage_raw is not None else None
    # Reasoning models: Groq/OpenAI-style "reasoning", vLLM/DeepSeek-style "reasoning_content".
    reasoning = _get(message, "reasoning") or _get(message, "reasoning_content")

    return CallRecord(
        provider=provider,
        model=_get(response, "model") or request.get("model"),
        messages=messages,
        params=params,
        response_text=_get(message, "content"),
        reasoning_text=reasoning if isinstance(reasoning, str) else None,
        finish_reason=_get(first, "finish_reason"),
        tool_calls=[_to_dict(t) for t in raw_tool_calls] if raw_tool_calls else None,
        usage=usage,
        latency_ms=latency_ms,
        metadata={"n_choices": len(choices)},
    )


class _Guard:
    """Shared state for one guarded client."""

    def __init__(self, config: Config, detectors: Sequence[Detector], provider: str | None):
        self.config = config
        self.detectors = list(detectors)
        self.provider = provider

    def diagnose(self, request: dict[str, Any], response: Any, latency_ms: float) -> None:
        """Build the record, run detectors, attach the result. Never raises if fail_open."""
        try:
            record = record_from_openai(
                request, response, latency_ms=latency_ms, provider=self.provider
            )
            diagnosis: Diagnosis = run_detectors(record, self.detectors, self.config)
            attach(response, diagnosis)
            if self.config.on_diagnosis is not None:
                self.config.on_diagnosis(record, diagnosis)
        except Exception as exc:
            if not self.config.fail_open:
                raise
            logger.warning("counterfix could not diagnose this call: %s", exc)

    def wrap_create(self, create: Callable[..., Any]) -> Callable[..., Any]:
        guard = self

        def guarded_create(*args: Any, **kwargs: Any) -> Any:
            if not guard.config.enabled:
                return create(*args, **kwargs)
            if kwargs.get("stream"):
                # Streams are passed through untouched for now; buffering comes later.
                return create(*args, **kwargs)
            start = time.perf_counter()
            result = create(*args, **kwargs)  # the user's own errors propagate unchanged
            if inspect.isawaitable(result):
                return guard._finish_async(result, kwargs, start)
            guard.diagnose(kwargs, result, (time.perf_counter() - start) * 1000)
            return result

        guarded_create.__doc__ = create.__doc__
        guarded_create.__wrapped__ = create  # type: ignore[attr-defined]
        return guarded_create

    async def _finish_async(self, pending: Any, kwargs: dict[str, Any], start: float) -> Any:
        result = await pending
        self.diagnose(kwargs, result, (time.perf_counter() - start) * 1000)
        return result


class GuardedClient:
    """A transparent proxy: everything behaves like the wrapped client, except that
    ``chat.completions.create`` also produces a diagnosis."""

    __slots__ = ("_cf_guard", "_cf_path", "_cf_target")

    def __init__(self, target: Any, guard: _Guard, path: tuple[str, ...] = ()):
        object.__setattr__(self, "_cf_target", target)
        object.__setattr__(self, "_cf_guard", guard)
        object.__setattr__(self, "_cf_path", path)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_cf_target")
        guard = object.__getattribute__(self, "_cf_guard")
        path = (*object.__getattribute__(self, "_cf_path"), name)
        value = getattr(target, name)
        if path == _CREATE_PATH and callable(value):
            return guard.wrap_create(value)
        if path == _CREATE_PATH[: len(path)]:
            return GuardedClient(value, guard, path)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_cf_target"), name, value)

    def __repr__(self) -> str:
        return f"counterfix.guard({object.__getattribute__(self, '_cf_target')!r})"

    @property
    def unwrapped(self) -> Any:
        """The original client."""
        return object.__getattribute__(self, "_cf_target")


def guard(
    client: Any,
    *,
    config: Config | None = None,
    detectors: Sequence[Detector] | None = None,
    provider: str | None = None,
) -> Any:
    """Wrap an OpenAI-compatible client so every chat completion is checked.

    >>> from openai import OpenAI          # doctest: +SKIP
    >>> client = guard(OpenAI())           # doctest: +SKIP
    >>> r = client.chat.completions.create(model="gpt-4o-mini", messages=[...])  # doctest: +SKIP
    >>> r.diagnosis.summary()              # doctest: +SKIP

    ``detectors`` defaults to the built-in live detectors. ``provider`` is an optional
    label stored on each record (e.g. "groq", "ollama").
    """
    if isinstance(client, GuardedClient):
        return client
    if detectors is None:
        from counterfix.detectors import default_detectors

        detectors = default_detectors()
    return GuardedClient(client, _Guard(config or Config(), detectors, provider))
