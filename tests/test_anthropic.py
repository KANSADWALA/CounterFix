from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import counterfix
from counterfix import Config, FailureType


def message(*blocks: dict[str, Any], stop: str = "end_turn") -> SimpleNamespace:
    """A response shaped like the Anthropic SDK's Message (blocks as objects)."""
    return SimpleNamespace(
        id="msg_1",
        model="claude-haiku-5-5",
        content=[SimpleNamespace(**b) for b in blocks],
        stop_reason=stop,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
    )


class FakeMessages:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class FakeAnthropic:
    def __init__(self, messages: FakeMessages):
        self.messages = messages
        self.models = SimpleNamespace(list=lambda: [])


def _create(client: Any, **extra: Any) -> Any:
    return client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=8,
        system="Be brief.",
        messages=[{"role": "user", "content": "Write an essay on monsoons."}],
        **extra,
    )


def test_guard_reads_anthropic_messages_and_flags_truncation() -> None:
    seen: list[Any] = []
    fake = FakeMessages(
        message(
            {"type": "thinking", "thinking": "plan it"},
            {"type": "text", "text": "Monsoons are"},
            stop="max_tokens",
        )
    )
    client = counterfix.guard(
        FakeAnthropic(fake), config=Config(on_diagnosis=lambda r, d: seen.append(r))
    )
    response = _create(client)
    assert response.diagnosis.top.failure_type is FailureType.TRUNCATION
    record = seen[0]
    assert record.provider == "anthropic" and record.system_text == "Be brief."
    assert record.answer_text == "Monsoons are" and record.reasoning_text == "plan it"
    assert record.usage.total_tokens == 14 and record.max_tokens == 8


def test_retry_raises_max_tokens_only_and_keeps_the_better_answer() -> None:
    fake = FakeMessages(
        message({"type": "text", "text": "Monsoons are"}, stop="max_tokens"),
        message({"type": "text", "text": "Monsoons are seasonal winds."}),
    )
    client = counterfix.guard(FakeAnthropic(fake), config=Config(mode="retry"))
    response = _create(client)
    assert response.content[0].text == "Monsoons are seasonal winds."
    assert [a.outcome for a in response.diagnosis.retries] == ["fixed"]
    assert set(fake.calls[1]) == set(fake.calls[0]) and fake.calls[1]["max_tokens"] > 8


def test_tool_use_arguments_are_checked_against_input_schema() -> None:
    tool = {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    bad_call = {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"town": "Pune"}}
    client = counterfix.guard(FakeAnthropic(FakeMessages(message(bad_call, stop="tool_use"))))
    response = _create(client, tools=[tool])
    assert response.diagnosis.top.failure_type is FailureType.FORMAT_BREAK


def test_refusal_stop_reason_is_a_provider_filter_and_async_works() -> None:
    class AsyncMessages(FakeMessages):
        async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
            return FakeMessages.create(self, **kwargs)

    fake = AsyncMessages(message(stop="refusal"))
    client = counterfix.guard(FakeAnthropic(fake))
    response = asyncio.run(_create(client))
    assert response.diagnosis.top.failure_type is FailureType.PROVIDER_FILTER
