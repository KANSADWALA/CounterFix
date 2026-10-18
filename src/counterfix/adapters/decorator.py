"""``@watch``: live checks on any function that calls a model.

>>> @counterfix.watch(model="llama3.2")                   # doctest: +SKIP
... def ask(prompt: str) -> str:
...     return my_local_model(prompt)

The function's first argument (or its ``prompt`` / ``messages`` keyword) is read as the
request: a string is one user message, a list is chat messages. Its return value is read
as the answer: a string, or an OpenAI-shaped response. Results come back unchanged; the
diagnosis is on ``ask.last_diagnosis``, on ``response.diagnosis`` for response objects,
and goes to ``Config.on_diagnosis`` and background judge checks like ``guard()``'s.
Sync and async functions both work.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from counterfix.adapters.openai_compat import _Guard, record_from_openai
from counterfix.core.config import Config
from counterfix.core.detector import Detector
from counterfix.core.record import CallRecord, Message

F = TypeVar("F", bound=Callable[..., Any])


def _messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    request = kwargs.get("messages", kwargs.get("prompt", args[0] if args else None))
    if isinstance(request, str):
        return [{"role": "user", "content": request}]
    if isinstance(request, list):
        return [m if isinstance(m, dict) else {"role": "user", "content": str(m)} for m in request]
    return []


def _record(
    args: tuple[Any, ...], kwargs: dict[str, Any], result: Any, model: str | None, ms: float
) -> CallRecord | None:
    messages = _messages(args, kwargs)
    if isinstance(result, str):
        return CallRecord(
            model=model,
            messages=[Message.model_validate(m) for m in messages],
            response_text=result,
            latency_ms=ms,
        )
    if hasattr(result, "choices"):
        request = {"model": model, "messages": messages}
        return record_from_openai(request, result, latency_ms=ms)
    return None  # an answer we cannot read: leave it alone


def watch(
    fn: F | None = None,
    *,
    model: str | None = None,
    config: Config | None = None,
    detectors: Sequence[Detector] | None = None,
) -> Any:
    """Decorate a function that calls a model so each call gets the live checks."""
    if detectors is None:
        from counterfix.detectors import default_detectors

        detectors = default_detectors()
    state = _Guard(config or Config(), detectors, provider=None)

    def decorate(func: F) -> F:
        def check(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any, ms: float) -> None:
            try:
                record = _record(args, kwargs, result, model, ms)
            except Exception:
                if not state.config.fail_open:
                    raise
                return
            if record is not None:
                wrapper.last_diagnosis = state.check(record, result)  # type: ignore[attr-defined]

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                if state.config.enabled:
                    check(args, kwargs, result, (time.perf_counter() - start) * 1000)
                return result

        else:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                if state.config.enabled:
                    check(args, kwargs, result, (time.perf_counter() - start) * 1000)
                return result

        # ponytail: one slot per function, so concurrent callers can overwrite it;
        # use Config.on_diagnosis when calls run in parallel.
        wrapper.last_diagnosis = None  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorate(fn) if fn is not None else decorate
