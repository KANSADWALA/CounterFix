from __future__ import annotations

import asyncio
from typing import Any

import counterfix
from conftest import make_response
from counterfix import Config, FailureType


def test_watch_checks_a_plain_text_function_and_returns_it_unchanged() -> None:
    seen: list[Any] = []

    @counterfix.watch(model="local", config=Config(on_diagnosis=lambda r, d: seen.append(r)))
    def ask(prompt: str) -> str:
        return "The sky is usually blue on a clear day."

    answer = ask("Answer in exactly one word: sky colour?")
    assert answer == "The sky is usually blue on a clear day."
    assert ask.last_diagnosis.top.failure_type is FailureType.DROPPED_INSTRUCTION
    assert seen[0].model == "local" and seen[0].prompt_text.startswith("Answer in")


def test_watch_reads_messages_and_openai_shaped_results() -> None:
    @counterfix.watch
    def chat(messages: list[dict[str, str]]) -> Any:
        return make_response("Hello", finish_reason="length")

    response = chat(messages=[{"role": "user", "content": "Write an essay."}])
    assert response.diagnosis.top.failure_type is FailureType.TRUNCATION
    assert chat.last_diagnosis is response.diagnosis


def test_watch_async_and_unreadable_results_and_fail_open() -> None:
    @counterfix.watch
    async def ask(prompt: str) -> str:
        return "Hi!"

    assert asyncio.run(ask("Say hi")) == "Hi!" and ask.last_diagnosis.ok

    @counterfix.watch
    def numbers(prompt: str) -> list[int]:
        return [1, 2]

    assert numbers("x") == [1, 2] and numbers.last_diagnosis is None

    class Broken:
        name = "broken"

        def check(self, record: Any, config: Any) -> Any:
            raise RuntimeError("boom")

    @counterfix.watch(detectors=[Broken()])
    def ok(prompt: str) -> str:
        return "fine"

    assert ok("x") == "fine"
