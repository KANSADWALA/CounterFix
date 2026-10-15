"""Calling a model again: needed by sampling checks, LLM judges and (later) counterfactual
re-runs. A caller turns (messages, params) into a CallRecord."""

from __future__ import annotations

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


class RateLimitedCaller:
    """Wraps a caller to stay under a requests-per-minute limit, and waits and retries
    when the provider answers 429 (rate limited). Thread-safe, so a panel can share it."""

    def __init__(self, caller: ModelCaller, rpm: float, *, retries: int = 3, backoff: float = 20.0):
        self.caller = caller
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.retries, self.backoff = retries, backoff
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
                if not limited or attempt == self.retries:
                    raise
                time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError("unreachable")
