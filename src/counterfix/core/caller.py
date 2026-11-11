"""Calling a model again: needed by sampling checks, LLM judges and (later) counterfactual
re-runs. A caller turns (messages, params) into a CallRecord."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from counterfix.core.record import CallRecord, Message
from counterfix.core.sync import rerun_params, resolve


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
        from counterfix.core.sync import sync_client

        unwrapped = getattr(client, "unwrapped", None)
        self.client = sync_client(unwrapped if unwrapped is not None else client)
        self.model = model
        self.provider = provider
        self.defaults = dict(defaults or {})
        # Request fields this model refused once, and what to do with them next time.
        self._adapt: dict[str, str | None] = {}

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
        request.update(rerun_params(params))
        request = self._adapted(request)
        start = time.perf_counter()
        for _ in range(3):  # each refused field is adapted once
            try:
                response = resolve(self.client.chat.completions.create(**request))
                break
            except Exception as exc:
                if not self._learn(exc, request):
                    raise
                request = self._adapted(request)
                start = time.perf_counter()
        else:
            response = resolve(self.client.chat.completions.create(**request))
        latency = (time.perf_counter() - start) * 1000
        return record_from_openai(request, response, latency_ms=latency, provider=self.provider)

    def _adapted(self, request: dict[str, Any]) -> dict[str, Any]:
        """The request with the fields this model refused renamed or dropped."""
        out = dict(request)
        if "max_output_tokens" in out:  # a Responses API limit: chat calls it max_tokens
            limit = out.pop("max_output_tokens")
            out.setdefault("max_tokens", limit)
        for key, rename in self._adapt.items():
            if key in out:
                value = out.pop(key)
                if rename is not None:
                    out[rename] = max(value, out.get(rename) or 0) if rename in out else value
        return out

    def _learn(self, exc: Exception, request: dict[str, Any]) -> bool:
        """True when ``exc`` is a 400 about a request field we can adapt.

        Newer OpenAI models (the GPT-5 and o-series reasoning models) refuse ``max_tokens``
        (they take ``max_completion_tokens``) and any ``temperature`` / ``top_p`` other than
        the default; some servers refuse ``reasoning_effort``, and Groq refuses
        ``logprobs`` (the answer then has no token confidence). Other errors are left alone.
        """
        status = getattr(exc, "status_code", None)
        if status not in (400, 422) and type(exc).__name__ != "BadRequestError":
            return False
        message = str(exc).lower()
        if "max_tokens" in request and "max_completion_tokens" in message:
            self._adapt["max_tokens"] = "max_completion_tokens"
            return True
        if "logprobs" in request and "logprob" in message and "logprobs" not in self._adapt:
            self._adapt["logprobs"] = None  # Groq and others refuse them: answer without
            self._adapt["top_logprobs"] = None
            return True
        for key in ("temperature", "top_p", "reasoning_effort", "seed"):
            if key in request and key in message and key not in self._adapt:
                self._adapt[key] = None
                return True
        return False

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
