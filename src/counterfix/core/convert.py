"""Turning provider responses into CallRecords.

Pure functions over response objects or dicts: the OpenAI Chat Completions shape (also
used by Groq, Gemini, Ollama, vLLM, OpenRouter), the Anthropic Messages shape, and token
log probabilities. No SDK is imported. They live in ``core`` because callers (re-runs,
judges) and the guards both need them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from counterfix.core.record import CallRecord, Message, Usage
from counterfix.core.sync import clean_params

LOGPROBS_KEY = "logprobs"


def token_logprobs(choice: Any) -> list[tuple[str, float]]:
    """(token, logprob) pairs from a chat-completion choice (or a stream chunk's choice)."""
    content = _get(_get(choice, "logprobs"), "content") or []
    out: list[tuple[str, float]] = []
    for item in content:
        lp = _get(item, "logprob")
        if isinstance(lp, (int, float)) and not isinstance(lp, bool) and math.isfinite(lp):
            out.append((str(_get(item, "token") or ""), float(lp)))
    return out


def summarize(pairs: Iterable[tuple[str, float]], keep: int = 5) -> dict[str, Any] | None:
    """{"n", "mean", "min", "lowest": [[token, logprob], ...]} or None with no tokens."""
    items = list(pairs)
    if not items:
        return None
    values = [lp for _, lp in items]
    lowest = sorted(items, key=lambda p: p[1])[:keep]
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "lowest": [[t, round(lp, 4)] for t, lp in lowest],
    }


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
    params = clean_params({k: v for k, v in request.items() if k not in ("messages", "model")})

    choices = _get(response, "choices") or []
    first = choices[0] if len(choices) > 0 else None
    message = _get(first, "message")
    raw_tool_calls = _get(message, "tool_calls")
    usage_raw = _get(response, "usage")
    usage = _usage(usage_raw) if usage_raw is not None else None
    # Reasoning models: Groq/OpenAI-style "reasoning", vLLM/DeepSeek-style "reasoning_content".
    reasoning = _get(message, "reasoning") or _get(message, "reasoning_content")

    metadata: dict[str, Any] = {"n_choices": len(choices)}
    if first is not None and _get(first, "logprobs") is not None:
        summary = summarize(token_logprobs(first))
        if summary is not None:
            metadata[LOGPROBS_KEY] = summary
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
        metadata=metadata,
    )


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
    params = clean_params(
        {k: v for k, v in request.items() if k not in ("messages", "model", "system")}
    )

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


# --- OpenTelemetry GenAI spans ---------------------------------------------------


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _parts_text(message: Mapping[str, Any]) -> str:
    """Text of a GenAI semconv message ({"role", "parts": [{"type": "text", ...}]})."""
    if "content" in message:
        return Message(role="x", content=message["content"]).text
    texts = [p.get("content") or p.get("text") or "" for p in message.get("parts") or []]
    return "\n".join(t for t in texts if isinstance(t, str))


def record_from_span(attrs: Mapping[str, Any], *, latency_ms: float | None = None) -> CallRecord:
    """A CallRecord from a GenAI span's attributes."""
    messages: list[Message] = []
    answer: str | None = None
    finish: str | None = None
    inputs = _json(attrs.get("gen_ai.input.messages"))
    if isinstance(inputs, list):
        messages = [Message(role=m.get("role", "user"), content=_parts_text(m)) for m in inputs]
    else:  # older layout: gen_ai.prompt.0.role / .content
        n = 0
        while f"gen_ai.prompt.{n}.content" in attrs or f"gen_ai.prompt.{n}.role" in attrs:
            role = attrs.get(f"gen_ai.prompt.{n}.role", "user")
            messages.append(Message(role=role, content=attrs.get(f"gen_ai.prompt.{n}.content")))
            n += 1
    outputs = _json(attrs.get("gen_ai.output.messages"))
    if isinstance(outputs, list) and outputs:
        answer = _parts_text(outputs[0])
        finish = outputs[0].get("finish_reason")
    elif "gen_ai.completion.0.content" in attrs:
        answer = attrs["gen_ai.completion.0.content"]
        finish = attrs.get("gen_ai.completion.0.finish_reason")
    reasons = attrs.get("gen_ai.response.finish_reasons")
    if finish is None and isinstance(reasons, list | tuple) and reasons:
        finish = reasons[0]
    params = {}
    if (limit := attrs.get("gen_ai.request.max_tokens")) is not None:
        params["max_tokens"] = limit
    if (temp := attrs.get("gen_ai.request.temperature")) is not None:
        params["temperature"] = temp
    usage = None
    if "gen_ai.usage.input_tokens" in attrs or "gen_ai.usage.output_tokens" in attrs:
        usage = Usage(
            prompt_tokens=attrs.get("gen_ai.usage.input_tokens"),
            completion_tokens=attrs.get("gen_ai.usage.output_tokens"),
        )
    return CallRecord(
        provider=attrs.get("gen_ai.system") or attrs.get("gen_ai.provider.name"),
        model=attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model"),
        messages=messages,
        params=params,
        response_text=answer,
        finish_reason=finish,
        usage=usage,
        latency_ms=latency_ms,
    )
