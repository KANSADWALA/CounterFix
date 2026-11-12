"""Re-running calls for fixes: async clients, streams, and sync twins of async clients."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import CallRecord, Message
from counterfix.core.sync import resolve, sync_client

PLANETS = [{"role": "user", "content": "List the planets of the solar system, one per line."}]


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
