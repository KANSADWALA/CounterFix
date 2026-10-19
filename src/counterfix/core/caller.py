"""Calling a model again: needed by sampling checks, LLM judges and (later) counterfactual
re-runs. A caller turns (messages, params) into a CallRecord."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from counterfix.core.record import CallRecord, Message


@runtime_checkable
class ModelCaller(Protocol):
    """Anything that can send chat messages to one model and return a CallRecord."""

    @property
    def name(self) -> str:
        """A label such as 'groq:openai/gpt-oss-20b', used in verdicts and caches."""
        ...

    def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> CallRecord:
        """Send one request. Errors propagate to the caller."""
        ...


class OpenAICompatibleCaller:
    """A ModelCaller over any OpenAI-compatible client (OpenAI, Groq, Gemini, Ollama...).

    Pass the *raw* client, not a guarded one, so checks never trigger more checks.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        provider: str | None = None,
        defaults: dict[str, Any] | None = None,
    ):
        unwrapped = getattr(client, "unwrapped", None)
        self.client = unwrapped if unwrapped is not None else client
        self.model = model
        self.provider = provider
        self.defaults = dict(defaults or {})

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}" if self.provider else self.model

    def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> CallRecord:
        from counterfix.adapters.openai_compat import record_from_openai

        payload = [
            m.model_dump(exclude_none=True) if isinstance(m, Message) else dict(m) for m in messages
        ]
        request: dict[str, Any] = {"model": self.model, "messages": payload}
        request.update(self.defaults)
        request.update(params)
        start = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        latency = (time.perf_counter() - start) * 1000
        return record_from_openai(request, response, latency_ms=latency, provider=self.provider)

    def __repr__(self) -> str:
        return f"OpenAICompatibleCaller({self.name!r})"


class DailyLimitError(RuntimeError):
    """The provider's daily quota for this model is used up; waiting minutes will not help."""


_DAILY = re.compile(r"per[ _]?day|\((?:TPD|RPD)\)|daily", re.IGNORECASE)


class RateLimitedCaller:
    """Wraps a caller to stay under a requests-per-minute limit, and waits and retries
    when the provider answers 429 (rate limited). Thread-safe, so a panel can share it."""

    def __init__(self, caller: ModelCaller, rpm: float, *, retries: int = 6, backoff: float = 15.0):
        self.caller = caller
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.retries, self.backoff = retries, backoff
        self.server_retries = 2  # retries for a passing server error (500, 502, 503)
        self._lock = threading.Lock()
        self._next = 0.0

    @property
    def name(self) -> str:
        return self.caller.name

    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)

    def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> CallRecord:
        for attempt in range(self.retries + 1):
            self._wait_turn()
            try:
                return self.caller.complete(messages, **params)
            except Exception as exc:
                limited = type(exc).__name__ == "RateLimitError" or "429" in str(exc)
                if limited and _DAILY.search(str(exc)):
                    raise DailyLimitError(
                        f"{self.name}: the daily limit is used up. Try another model or "
                        f"provider, or wait for the reset. Provider said: {str(exc)[:200]}"
                    ) from exc
                flaky = _server_error(exc) and attempt < self.server_retries
                if not (limited or flaky) or attempt == self.retries:
                    raise
                wait = retry_after(exc) if limited else None
                if wait is not None:
                    delay = wait + 1.0
                elif limited:
                    delay = self.backoff * (attempt + 1)
                else:
                    delay = 5.0 * (attempt + 1)  # a passing 500/503: short pause, then retry
                with self._lock:  # everyone sharing this caller waits, not just this thread
                    self._next = max(self._next, time.monotonic() + delay)
        raise RuntimeError("unreachable")


def _server_error(exc: Exception) -> bool:
    """A provider-side failure worth one more try (not a bad request)."""
    name = type(exc).__name__
    if name in ("InternalServerError", "ServiceUnavailableError", "APIConnectionError"):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in (500, 502, 503, 504)


_TRY_AGAIN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", re.IGNORECASE)


def retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait: the Retry-After header, or Groq's
    'Please try again in 7.5s' / '1m3.2s' message. None if not given."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    match = _TRY_AGAIN.search(str(exc))
    if match:
        minutes = float(match.group(1) or 0)
        amount = float(match.group(2))
        return minutes * 60 + (amount / 1000 if match.group(3).lower() == "ms" else amount)
    return None
