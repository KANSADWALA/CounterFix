from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any

import counterfix
from conftest import FakeClient, FakeCompletions
from counterfix import Config, FailureType
from test_anthropic import FakeAnthropic, FakeMessages


def chunk(content: str | None = None, finish: str | None = None, **delta: Any) -> NS:
    return NS(model="m", choices=[NS(delta=NS(content=content, **delta), finish_reason=finish)])


class Streaming(FakeCompletions):
    def __init__(self, chunks: list[Any]):
        super().__init__()
        self.chunks = chunks

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return iter(self.chunks)


def test_openai_stream_yields_every_chunk_then_checks_the_whole_answer() -> None:
    seen: list[Any] = []
    chunks = [chunk("Monsoons "), chunk("are"), chunk(None, "length")]
    client = counterfix.guard(
        FakeClient(Streaming(chunks)), config=Config(on_diagnosis=lambda r, d: seen.append(r))
    )
    stream = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "Essay"}], stream=True, max_tokens=3
    )
    assert stream.diagnosis is None
    got = [c.choices[0].delta.content for c in stream]
    assert got == ["Monsoons ", "are", None]
    assert stream.diagnosis.top.failure_type is FailureType.TRUNCATION
    assert seen[0].answer_text == "Monsoons are"


def test_streamed_tool_call_arguments_are_assembled_and_checked() -> None:
    def call(**fn: Any) -> NS:
        return NS(index=0, id="c1", function=NS(**fn))

    chunks = [
        chunk(None, tool_calls=[call(name="get_weather", arguments='{"ci')]),
        chunk(None, tool_calls=[call(name=None, arguments='ty": 3')]),
        chunk(None, "tool_calls"),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    client = counterfix.guard(FakeClient(Streaming(chunks)))
    stream = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "?"}], stream=True, tools=tools
    )
    list(stream)
    assert stream.diagnosis.top.failure_type is FailureType.FORMAT_BREAK  # '{"city": 3' breaks


def test_anthropic_async_stream_events() -> None:
    events = [
        NS(type="message_start", message=NS(model="claude-haiku-5-5", usage=NS(input_tokens=3))),
        NS(type="content_block_start", content_block=NS(type="text", id=None, name=None)),
        NS(type="content_block_delta", delta=NS(type="text_delta", text="I can't help")),
        NS(type="content_block_delta", delta=NS(type="text_delta", text=" with that.")),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=6)),
    ]

    class AsyncEvents:
        def __init__(self) -> None:
            self.items = iter(events)

        def __aiter__(self) -> AsyncEvents:
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration from None

    class AsyncStreamingMessages(FakeMessages):
        async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
            return AsyncEvents()

    client = counterfix.guard(FakeAnthropic(AsyncStreamingMessages(None)))

    async def go() -> Any:
        stream = await client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "How do I kill a hung Python process?"}],
            stream=True,
        )
        return stream, [e async for e in stream]

    stream, got = asyncio.run(go())
    assert len(got) == 5
    assert stream.diagnosis.top.failure_type is FailureType.OVER_REFUSAL
