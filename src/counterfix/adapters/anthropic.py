"""Reading Anthropic Messages API calls (``client.messages.create``) into a CallRecord.

``guard(Anthropic())`` picks this up on its own: a client with ``messages`` and no
``chat`` is treated as an Anthropic client. The SDK is never imported; anything with the
same response shape works (``content`` blocks, ``stop_reason``, ``usage``).

Tool calls are stored in the OpenAI shape the detectors read; F2 checks ``tool_use``
arguments against each tool's ``input_schema``. Thinking blocks are
recorded as reasoning text. Streams and the ``messages.stream()`` helper pass through
unchecked for now.
"""

from __future__ import annotations

from typing import Any

from counterfix.adapters.openai_compat import _get, _to_dict
from counterfix.core.record import CallRecord, Message, Usage

# Retries may only change these request fields (other fixes are OpenAI parameters).
ANTHROPIC_RETRY_PARAMS = frozenset({"max_tokens"})


def _blocks(response: Any) -> list[dict[str, Any]]:
    return [_to_dict(b) for b in _get(response, "content") or []]


def record_from_anthropic(
    request: dict[str, Any],
    response: Any,
    *,
    latency_ms: float | None = None,
    provider: str | None = "anthropic",
) -> CallRecord:
    """Build a :class:`CallRecord` from ``messages.create`` kwargs and the Message."""
    messages = [Message.model_validate(_to_dict(m)) for m in request.get("messages") or []]
    system = request.get("system")
    if system:
        messages.insert(0, Message(role="system", content=system))
    params = {k: v for k, v in request.items() if k not in ("messages", "model", "system")}

    blocks = _blocks(response)
    text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
    thinking = "\n".join(b.get("thinking") or "" for b in blocks if b.get("type") == "thinking")
    tool_calls = [
        {
            "id": b.get("id"),
            "type": "function",
            "function": {"name": b.get("name"), "arguments": b.get("input")},
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    raw_usage = _get(response, "usage")
    usage = None
    if raw_usage is not None:
        prompt, completion = _get(raw_usage, "input_tokens"), _get(raw_usage, "output_tokens")
        total = (
            prompt + completion if isinstance(prompt, int) and isinstance(completion, int) else None
        )
        usage = Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    return CallRecord(
        provider=provider,
        model=_get(response, "model") or request.get("model"),
        messages=messages,
        params=params,
        response_text=text if blocks else None,
        reasoning_text=thinking or None,
        finish_reason=_get(response, "stop_reason"),
        tool_calls=tool_calls or None,
        usage=usage,
        latency_ms=latency_ms,
        metadata={"stop_details": _to_dict(sd)} if (sd := _get(response, "stop_details")) else {},
    )


class AnthropicCaller:
    """A ModelCaller over an Anthropic client (``messages.create``), for re-runs."""

    def __init__(self, client: Any, model: str, *, provider: str = "anthropic"):
        unwrapped = getattr(client, "unwrapped", None)
        self.client = unwrapped if unwrapped is not None else client
        self.model, self.provider = model, provider

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        import time

        payload = [
            m.model_dump(exclude_none=True) if isinstance(m, Message) else dict(m) for m in messages
        ]
        system = "\n".join(Message.model_validate(m).text for m in payload if m["role"] == "system")
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [m for m in payload if m["role"] != "system"],
            "max_tokens": 1024,
            **params,
        }
        if system:
            request["system"] = system
        start = time.perf_counter()
        response = self.client.messages.create(**request)
        ms = (time.perf_counter() - start) * 1000
        return record_from_anthropic(request, response, latency_ms=ms, provider=self.provider)
