"""guard() on the OpenAI Responses API: records, truncation, JSON, streams, retries, async."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import counterfix
from counterfix.adapters.responses import (
    chat_params,
    messages_from_input,
    record_from_responses,
    responses_changes,
)
from counterfix.core.types import FailureType


def _resp(text: str = "Paris.", status: str = "completed", reason: str | None = None, **extra):
    out = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    ]
    return {
        "model": "gpt-test",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": extra.pop("output", out),
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        **extra,
    }


class _Responses:
    def __init__(self, replies: list[Any]):
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if kwargs.get("stream"):
            return iter(reply)
        return SimpleNamespace(**reply) if isinstance(reply, dict) else reply


class _Chat:
    completions = SimpleNamespace(create=lambda **kw: None)


def _client(replies: list[Any]) -> Any:
    return SimpleNamespace(
        responses=_Responses(replies), chat=_Chat(), base_url="https://api.openai.com/v1"
    )


def test_record_in_chat_terms():
    req = {
        "model": "gpt-test",
        "instructions": "Be brief.",
        "input": [
            {"role": "developer", "content": "Use metric units."},
            {"role": "user", "content": [{"type": "input_text", "text": "Capital of France?"}]},
        ],
        "max_output_tokens": 50,
        "text": {"format": {"type": "json_schema", "name": "a", "schema": {"type": "object"}}},
        "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
        "reasoning": {"effort": "low"},
        "store": False,
    }
    rec = record_from_responses(req, _resp())
    assert [m.role for m in rec.messages] == ["system", "system", "user"]
    assert rec.messages[2].content == "Capital of France?"
    assert rec.params["max_output_tokens"] == 50 and rec.max_tokens == 50
    assert rec.limit_key == "max_output_tokens"
    assert rec.params["response_format"]["type"] == "json_schema"
    assert rec.params["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert rec.params["tools"][0]["function"]["name"] == "f"
    assert rec.params["reasoning_effort"] == "low"
    assert "store" not in rec.params
    assert rec.response_text == "Paris." and rec.finish_reason == "stop"
    assert rec.usage.prompt_tokens == 10 and rec.usage.reasoning_tokens == 2
    assert rec.metadata["api"] == "responses"


def test_string_input_and_tool_items():
    msgs = messages_from_input(None, "hi")
    assert msgs[0].role == "user" and msgs[0].content == "hi"
    msgs = messages_from_input(
        None,
        [
            {"role": "user", "content": "weather?"},
            {"type": "function_call", "call_id": "c1", "name": "w", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ],
    )
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]


def test_truncation_found_and_retried_in_responses_terms():
    client = counterfix.guard(
        _client(
            [_resp("The answer is", "incomplete", "max_output_tokens"), _resp("The answer is 42.")]
        ),
        config=counterfix.Config(mode="retry"),
    )
    r = client.responses.create(model="gpt-test", input="q", max_output_tokens=16)
    calls = client.unwrapped.responses.calls
    assert len(calls) == 2
    assert calls[1]["max_output_tokens"] > 16 and "max_tokens" not in calls[1]
    d = counterfix.get_diagnosis(r)
    assert d.retries and d.retries[0].failure_type is FailureType.TRUNCATION


def test_truncation_reported_without_retry():
    client = counterfix.guard(_client([_resp("The answer is", "incomplete", "max_output_tokens")]))
    r = client.responses.create(model="gpt-test", input="q", max_output_tokens=16)
    assert FailureType.TRUNCATION in {f.failure_type for f in r.diagnosis.findings}


def test_broken_json_with_text_format():
    client = counterfix.guard(_client([_resp('{"a": 1')]))
    r = client.responses.create(
        model="gpt-test", input="give json", text={"format": {"type": "json_object"}}
    )
    assert FailureType.FORMAT_BREAK in {f.failure_type for f in r.diagnosis.findings}


def test_changes_translated():
    req = {"text": {"verbosity": "low"}}
    out = responses_changes({"max_tokens": 64, "response_format": {"type": "json_object"}}, req)
    assert out == {
        "max_output_tokens": 64,
        "text": {"verbosity": "low", "format": {"type": "json_object"}},
    }
    assert chat_params({"model": "m", "input": "x", "previous_response_id": "r1"}) == {}


def test_stream_checked_and_timed():
    events = [
        {"type": "response.created", "response": {"model": "gpt-test"}},
        {"type": "response.output_text.delta", "delta": "The answer"},
        {"type": "response.output_text.delta", "delta": " is"},
        {
            "type": "response.incomplete",
            "response": _resp("The answer is", "incomplete", "max_output_tokens"),
        },
    ]
    client = counterfix.guard(_client([events]))
    stream = client.responses.create(model="gpt-test", input="q", stream=True, max_output_tokens=8)
    assert len(list(stream)) == 4
    assert stream.timing.ttft_ms is not None and stream.timing.chunks == 2
    assert FailureType.TRUNCATION in {f.failure_type for f in stream.diagnosis.findings}


def test_async_client():
    class _Async:
        def __init__(self) -> None:
            self.inner = _Responses([_resp("ok")])

        async def create(self, **kw: Any) -> Any:
            return self.inner.create(**kw)

    client = counterfix.guard(SimpleNamespace(responses=_Async(), chat=_Chat()))
    r = asyncio.run(client.responses.create(model="gpt-test", input="hi"))
    assert r.diagnosis is not None and r.diagnosis.ok


def test_chat_still_guarded_next_to_responses():
    client = counterfix.guard(_client([_resp()]))
    assert hasattr(client.chat.completions.create, "__wrapped__")
    assert hasattr(client.responses.create, "__wrapped__")


def test_caller_adapts_to_newer_openai_models():
    from counterfix.core.caller import OpenAICompatibleCaller

    class BadRequestError(Exception):
        status_code = 400

    sent: list[dict[str, Any]] = []

    def create(**req: Any) -> Any:
        sent.append(req)
        if "max_tokens" in req:
            raise BadRequestError(
                "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."
            )
        if "temperature" in req:
            raise BadRequestError("Unsupported value: 'temperature' does not support 0.")
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            model="gpt-5-mini",
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    caller = OpenAICompatibleCaller(client, "gpt-5-mini", provider="openai")
    rec = caller.complete([{"role": "user", "content": "hi"}], max_tokens=64, temperature=0)
    assert rec.response_text == "ok"
    assert sent[-1] == {
        "model": "gpt-5-mini",
        "messages": sent[-1]["messages"],
        "max_completion_tokens": 64,
    }
    sent.clear()
    caller.complete([{"role": "user", "content": "hi"}], max_tokens=32, temperature=0)
    assert len(sent) == 1  # learned: adapted up front


def test_caller_leaves_other_errors_alone():
    import pytest

    from counterfix.core.caller import OpenAICompatibleCaller

    def create(**req: Any) -> Any:
        raise RuntimeError("connection reset")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError):
        OpenAICompatibleCaller(client, "m").complete(
            [{"role": "user", "content": "x"}], max_tokens=5
        )
