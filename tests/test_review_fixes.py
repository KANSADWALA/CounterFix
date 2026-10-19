"""Regressions found in the pre-release code review (each test names the bug it guards)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import CallRecord, Config, FailureType, Message
from counterfix.core.sync import clean_params, rerun_params, resolve, sync_client
from counterfix.report import load_log, log_line

PLANETS = [{"role": "user", "content": "List the planets of the solar system, one per line."}]


class Sentinel:  # stands in for openai.NOT_GIVEN
    pass


Sentinel.__name__ = "NotGiven"


class ContextClient(FakeClient):
    def __init__(self, completions: FakeCompletions | None = None):
        super().__init__(completions)
        self.entered = self.exited = 0
        self.chat.completions.parse = self.chat.completions.create  # structured outputs

    def __enter__(self) -> ContextClient:
        self.entered += 1
        return self

    def __exit__(self, *exc: Any) -> None:
        self.exited += 1

    async def __aenter__(self) -> ContextClient:
        self.entered += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited += 1

    def with_options(self, **options: Any) -> ContextClient:
        return self


def test_with_statement_and_async_with_work() -> None:
    raw = ContextClient()
    with counterfix.guard(raw) as client:  # was AttributeError: __enter__
        r = client.chat.completions.create(model="m", messages=PLANETS)
    assert counterfix.get_diagnosis(r) is not None and raw.entered == raw.exited == 1

    async def go() -> None:
        async with counterfix.guard(raw) as client:  # was AttributeError: __aenter__
            assert client.unwrapped is raw

    asyncio.run(go())
    assert raw.entered == raw.exited == 2


def test_with_options_and_parse_stay_checked() -> None:
    client = counterfix.guard(ContextClient(FakeCompletions(make_response("x", "length"))))
    r1 = client.with_options(timeout=5).chat.completions.create(model="m", messages=PLANETS)
    r2 = client.chat.completions.parse(model="m", messages=PLANETS)
    for r in (r1, r2):  # both were silently unchecked
        assert counterfix.get_diagnosis(r).findings[0].failure_type is FailureType.TRUNCATION


def test_guard_again_with_new_settings_is_honoured() -> None:
    raw = FakeClient()
    off = counterfix.guard(counterfix.guard(raw), config=Config(enabled=False))
    r = off.chat.completions.create(model="m", messages=PLANETS)
    assert counterfix.get_diagnosis(r) is None  # was: the old settings, silently


def test_provider_is_read_from_the_base_url() -> None:
    raw = FakeClient()
    raw.base_url = "https://api.groq.com/openai/v1"
    client = counterfix.guard(raw)
    r = client.chat.completions.create(model="m", messages=PLANETS)
    assert counterfix.get_diagnosis(r).record.provider == "groq"  # fix()/export-tests need it
    raw.base_url = "http://my-vllm:8000/v1"
    r = counterfix.guard(raw).chat.completions.create(model="m", messages=PLANETS)
    assert counterfix.get_diagnosis(r).record.provider is None


def test_think_tag_mentioned_mid_answer_is_kept() -> None:
    text = "Qwen puts reasoning in a <think> tag. Gemma uses <thought> instead."
    assert CallRecord(response_text=text).answer_text == text  # was cut at "<think>"
    assert CallRecord(response_text="<think>plan</think>\n Blue.").answer_text == "Blue."
    assert CallRecord(response_text="<thought>a</thought><think>b</think>Hi").answer_text == "Hi"
    assert CallRecord(response_text="  <think>cut off mid tho").answer_text == ""


def test_watch_on_a_method_reads_the_prompt_not_self() -> None:
    class Bot:
        @counterfix.watch(model="m")
        def ask(self, prompt: str) -> str:
            return "The sky is usually blue on a clear day."

    Bot().ask("Answer in exactly one word: sky colour?")
    diagnosis = Bot.ask.last_diagnosis  # type: ignore[attr-defined]
    assert diagnosis.findings[0].failure_type is FailureType.DROPPED_INSTRUCTION


def test_log_line_masks_secrets_and_survives_odd_params(tmp_path: Any) -> None:
    record = CallRecord(
        model="m",
        messages=[Message(role="user", content="hi")],
        params={
            "extra_headers": {"Authorization": "Bearer sk-SECRET", "X-Trace": "abc"},
            "api_key": "sk-OTHER",
            "timeout": object(),  # not JSON: written as text, line not lost
            "response_format": Message,  # a class, as parse() passes
        },
        response_text="ok",
    )
    line = log_line(record, counterfix.diagnose(record, judge=False))
    assert "sk-SECRET" not in line and "sk-OTHER" not in line and '"X-Trace": "abc"' in line
    path = tmp_path / "log.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    [back] = load_log(path)
    assert back.params["extra_headers"] == {"X-Trace": "abc"}  # masked header dropped on load


def test_not_given_placeholders_are_dropped() -> None:
    assert clean_params({"a": 1, "b": Sentinel()}) == {"a": 1}
    assert rerun_params({"stream": True, "stream_options": {}, "max_tokens": 5}) == {
        "max_tokens": 5
    }
    client = counterfix.guard(FakeClient())
    r = client.chat.completions.create(model="m", messages=PLANETS, temperature=Sentinel())
    assert "temperature" not in counterfix.get_diagnosis(r).record.params


def test_deeply_nested_json_is_a_format_finding_not_a_crash() -> None:
    record = CallRecord(
        messages=[Message(role="user", content="Return ONLY JSON.")],
        response_text="[" * 5000 + "]" * 5000,
        finish_reason="stop",
    )
    d = counterfix.diagnose(record, judge=False, config=Config(fail_open=False))
    assert d.findings[0].failure_type is FailureType.FORMAT_BREAK


# --- re-runs with async clients and streams (fix() used to report false passes) ----------


class AsyncCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        big = (kwargs.get("max_tokens") or 4096) >= 100
        return make_response(
            "Mercury ... Neptune" if big else "Mercury, Ven", "stop" if big else "length"
        )


class AsyncFakeClient:  # no sync twin can be built for it: resolve() must await
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=AsyncCompletions())


def test_resolve_and_sync_client() -> None:
    async def seven() -> int:
        return 7

    assert resolve(seven()) == 7
    assert resolve(3) == 3

    async def inside_a_loop() -> int:
        return resolve(seven())  # a running loop: finished on a private thread

    assert asyncio.run(inside_a_loop()) == 7
    plain = FakeClient()
    assert sync_client(plain) is plain


def test_fix_awaits_async_clients() -> None:
    raw = AsyncFakeClient()
    client = counterfix.guard(raw)

    async def call() -> Any:
        return await client.chat.completions.create(model="m", messages=PLANETS, max_tokens=20)

    response = asyncio.run(call())
    result = counterfix.fix(response, judge=False)
    assert result.retry is not None and result.retry.passed == 0  # was a false 2/2 pass
    assert result.fix is not None and result.fix.changed == {"max_tokens": 4096}


def test_fix_never_streams_a_rerun_and_unreadable_reruns_never_pass() -> None:
    seen: list[dict[str, Any]] = []

    class Caller:
        name = "fake"

        def complete(self, messages: Any, **params: Any) -> CallRecord:
            seen.append(params)
            return CallRecord(response_text=None, finish_reason=None)  # nothing readable

    record = CallRecord(
        model="m",
        messages=[Message(**PLANETS[0])],
        params={"max_tokens": 20, "stream": True},
        response_text="Mercury, Ven",
        finish_reason="length",
    )
    result = counterfix.fix(record, judge=False, caller=Caller())
    assert result.fix is None and result.retry is not None and result.retry.passed == 0
    assert "no readable answer" in result.explain()


def test_rerun_callers_drop_stream() -> None:
    from counterfix.core.caller import OpenAICompatibleCaller

    completions = FakeCompletions()
    caller = OpenAICompatibleCaller(FakeClient(completions), "m")
    caller.complete(PLANETS, stream=True, stream_options={"include_usage": True}, max_tokens=9)
    assert "stream" not in completions.calls[-1] and completions.calls[-1]["max_tokens"] == 9
    assert json.dumps(completions.calls[-1])  # plain data only


def test_sync_twin_of_a_real_async_openai_client() -> None:
    openai = pytest.importorskip("openai")
    async_client = openai.AsyncOpenAI(api_key="k-1", base_url="http://example.test/v1")
    twin = sync_client(async_client)
    assert type(twin).__name__ == "OpenAI" and twin.api_key == "k-1"
    assert str(twin.base_url).startswith("http://example.test/v1")
