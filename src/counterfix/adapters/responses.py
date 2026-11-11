"""The OpenAI Responses API (``client.responses.create`` / ``.parse``).

OpenAI recommends the Responses API for new projects, so ``guard()`` checks it as well as
Chat Completions. A Responses call is turned into the same :class:`CallRecord` as a chat
call, with the request written in chat terms (``instructions`` becomes a system message,
``text.format`` becomes ``response_format``,
function tools are nested the chat way). Every detector, judge and ``fix()`` then works
unchanged, and a re-run goes through Chat Completions. The original request stays on
``record.metadata["responses_request"]``.

Known limit: with ``previous_response_id`` or ``conversation`` the earlier turns live on
OpenAI's servers, so the record (and any re-run) sees only the new input; the record notes
this in ``metadata["server_state"]``.
"""

from __future__ import annotations

import json
from typing import Any

from counterfix.core.record import CallRecord, Message, Usage
from counterfix.core.sync import clean_params

RESPONSES_PATHS = (("responses", "create"), ("responses", "parse"))
# Request fields a retry may change, in Responses terms.
RESPONSES_RETRY_PARAMS = frozenset(
    {"max_output_tokens", "temperature", "top_p", "text", "reasoning", "tool_choice"}
)
# Request fields that describe the input, not a setting.
_INPUT_KEYS = frozenset({"input", "instructions", "model"})
_SERVER_STATE_KEYS = ("previous_response_id", "conversation")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(obj: Any) -> Any:
    """SDK objects to plain dicts and lists."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:
            pass
    return obj


# --- request ------------------------------------------------------------------------------


def _part(part: Any) -> Any:
    """One Responses content part in chat terms."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return part
    kind = part.get("type")
    if kind in ("input_text", "output_text", "text"):
        return {"type": "text", "text": part.get("text", "")}
    if kind == "input_image":
        url = part.get("image_url") or part.get("file_id") or ""
        return {"type": "image_url", "image_url": {"url": url}}
    return part


def _content(content: Any) -> Any:
    if isinstance(content, list):
        parts = [_part(p) for p in content]
        if all(isinstance(p, dict) and p.get("type") == "text" for p in parts):
            return "\n".join(p["text"] for p in parts)
        return parts
    return content


def messages_from_input(instructions: Any, items: Any) -> list[Message]:
    """``instructions`` + ``input`` (a string or a list of items) as chat messages."""
    messages: list[Message] = []
    if isinstance(instructions, str) and instructions:
        messages.append(Message(role="system", content=instructions))
    if isinstance(items, str):
        messages.append(Message(role="user", content=items))
        return messages
    for raw in _plain(items) or []:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type", "message")
        if kind == "message" or ("role" in raw and kind not in _CALL_TYPES):
            role = raw.get("role", "user")
            role = "system" if role == "developer" else role
            messages.append(Message(role=role, content=_content(raw.get("content"))))
        elif kind == "function_call":
            call = {
                "id": raw.get("call_id") or raw.get("id"),
                "type": "function",
                "function": {"name": raw.get("name"), "arguments": raw.get("arguments", "")},
            }
            if messages and messages[-1].role == "assistant" and messages[-1].content is None:
                getattr(messages[-1], "tool_calls", []).append(call)
            else:
                messages.append(
                    Message.model_validate(
                        {"role": "assistant", "content": None, "tool_calls": [call]}
                    )
                )
        elif kind == "function_call_output":
            output = raw.get("output")
            messages.append(
                Message.model_validate(
                    {
                        "role": "tool",
                        "tool_call_id": raw.get("call_id"),
                        "content": output if isinstance(output, str) else json.dumps(output),
                    }
                )
            )
    return messages


_CALL_TYPES = frozenset({"function_call", "function_call_output", "reasoning"})


def _chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    """Responses function tools (flat) as chat tools (nested); built-in tools kept as is."""
    out: list[dict[str, Any]] = []
    for tool in _plain(tools) or []:
        if isinstance(tool, dict) and tool.get("type") == "function" and "function" not in tool:
            fn = {k: tool[k] for k in ("name", "description", "parameters", "strict") if k in tool}
            out.append({"type": "function", "function": fn})
        elif isinstance(tool, dict):
            out.append(tool)
    return out or None


def _response_format(text: Any) -> dict[str, Any] | None:
    fmt = _get(_plain(text), "format")
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind == "json_schema":
        spec = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
        return {"type": "json_schema", "json_schema": spec}
    if kind == "json_object":
        return {"type": "json_object"}
    return None


def chat_params(request: dict[str, Any]) -> dict[str, Any]:
    """Request settings in Chat Completions terms."""
    params: dict[str, Any] = {}
    for key, value in clean_params(request).items():
        if key in _INPUT_KEYS or key in _SERVER_STATE_KEYS:
            continue
        if key == "text":
            if (fmt := _response_format(value)) is not None:
                params["response_format"] = fmt
        elif key == "tools":
            if (tools := _chat_tools(value)) is not None:
                params["tools"] = tools
        elif key == "reasoning":
            effort = _get(_plain(value), "effort")
            if effort:
                params["reasoning_effort"] = effort
        elif key in ("store", "include", "truncation", "background", "prompt"):
            continue  # Responses-only switches with no chat meaning
        else:
            params[key] = _plain(value)
    return params


def responses_changes(changes: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """A retry's parameter change (chat terms) in Responses terms."""
    out: dict[str, Any] = {}
    for key, value in changes.items():
        if key in ("max_tokens", "max_completion_tokens"):
            out["max_output_tokens"] = value
        elif key == "response_format" and isinstance(value, dict):
            text = dict(_plain(request.get("text")) or {})
            if value.get("type") == "json_schema":
                spec = value.get("json_schema") or {}
                text["format"] = {"type": "json_schema", **spec}
            else:
                text["format"] = {"type": value.get("type", "json_object")}
            out["text"] = text
        elif key == "reasoning_effort":
            out["reasoning"] = {**(_plain(request.get("reasoning")) or {}), "effort": value}
        else:
            out[key] = value
    return out


# --- response -----------------------------------------------------------------------------


def _output_text(response: Any) -> tuple[str | None, str | None]:
    """(answer text, refusal text) from the output items."""
    texts: list[str] = []
    refusals: list[str] = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for part in _get(item, "content") or []:
            kind = _get(part, "type")
            if kind == "output_text":
                texts.append(_get(part, "text") or "")
            elif kind == "refusal":
                refusals.append(_get(part, "refusal") or "")
    text = "".join(texts) if texts else None
    if text is None:
        direct = _get(response, "output_text")
        text = direct if isinstance(direct, str) and direct else None
    return text, ("".join(refusals) or None)


def _reasoning(response: Any) -> str | None:
    parts: list[str] = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "reasoning":
            continue
        for key in ("content", "summary"):
            for part in _get(item, key) or []:
                if isinstance(t := _get(part, "text"), str):
                    parts.append(t)
    return "\n".join(parts) or None


def _tool_calls(response: Any) -> list[dict[str, Any]] | None:
    calls = [
        {
            "id": _get(item, "call_id") or _get(item, "id"),
            "type": "function",
            "function": {"name": _get(item, "name"), "arguments": _get(item, "arguments") or ""},
        }
        for item in _get(response, "output") or []
        if _get(item, "type") == "function_call"
    ]
    return calls or None


def _finish(response: Any, tool_calls: Any) -> str | None:
    status = _get(response, "status")
    if status == "incomplete":
        reason = _get(_get(response, "incomplete_details"), "reason")
        return str(reason) if reason else "incomplete"
    if status == "failed":
        return "error"
    if tool_calls:
        return "tool_calls"
    return "stop" if status in ("completed", None) else str(status)


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    usage = Usage(
        prompt_tokens=_get(raw, "input_tokens"),
        completion_tokens=_get(raw, "output_tokens"),
        total_tokens=_get(raw, "total_tokens"),
    )
    reasoning = _get(_get(raw, "output_tokens_details"), "reasoning_tokens")
    if isinstance(reasoning, int):
        usage.reasoning_tokens = reasoning
    return usage


def record_from_responses(
    request: dict[str, Any],
    response: Any,
    *,
    latency_ms: float | None = None,
    provider: str | None = None,
) -> CallRecord:
    """Build a :class:`CallRecord` from ``responses.create`` kwargs and the response."""
    text, refusal = _output_text(response)
    tool_calls = _tool_calls(response)
    metadata: dict[str, Any] = {
        "api": "responses",
        "responses_request": {
            k: _plain(v) for k, v in clean_params(request).items() if k != "input"
        },
    }
    if any(request.get(k) for k in _SERVER_STATE_KEYS):
        metadata["server_state"] = True
    if refusal:
        metadata["refusal"] = refusal
    error = _get(response, "error")
    return CallRecord(
        provider=provider,
        model=_get(response, "model") or request.get("model"),
        messages=messages_from_input(request.get("instructions"), request.get("input")),
        params=chat_params(request),
        response_text=text if text is not None else refusal,
        reasoning_text=_reasoning(response),
        finish_reason=_finish(response, tool_calls),
        tool_calls=tool_calls,
        usage=_usage(_get(response, "usage")),
        latency_ms=latency_ms,
        error=str(_get(error, "message") or error) if error else None,
        metadata=metadata,
    )


# --- streaming ----------------------------------------------------------------------------


class ResponsesParts:
    """Collects Responses stream events. The final event carries the whole response, so
    that is what gets checked; the deltas only time the stream."""

    def __init__(self) -> None:
        self.final: Any = None
        self.text: list[str] = []
        self.model: Any = None

    def feed(self, event: Any) -> tuple[bool, bool]:
        kind = str(_get(event, "type") or "")
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self.final = _get(event, "response")
            return False, False
        if kind == "response.created":
            self.model = _get(_get(event, "response"), "model")
            return False, False
        delta = _get(event, "delta")
        if kind in ("response.output_text.delta", "response.refusal.delta"):
            if isinstance(delta, str):
                self.text.append(delta)
            return bool(delta), False
        if kind == "response.function_call_arguments.delta":
            return bool(delta), False
        if kind.startswith("response.reasoning") and kind.endswith(".delta"):
            return False, bool(delta)
        return False, False

    def response(self) -> Any:
        if self.final is not None:
            return self.final
        # The stream ended without a final event: check what arrived.
        return {
            "model": self.model,
            "status": "incomplete",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "".join(self.text)}],
                }
            ],
        }
