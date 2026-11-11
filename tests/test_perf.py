"""Speed and cost (E1): TTFT from streams, percentiles, prices, OTel metrics, speed fixes."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from types import SimpleNamespace as NS
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions
from counterfix import CallRecord, Config, Message, Timing, Usage
from counterfix.perf import (
    cost_usd,
    percentile,
    set_prices,
    speed_fix,
    speed_summary,
    tokens_per_second,
)


def chunk(content: str | None = None, finish: str | None = None, **delta: Any) -> NS:
    return NS(model="m", choices=[NS(delta=NS(content=content, **delta), finish_reason=finish)])


class Clock:
    """A fake perf_counter: time moves only when the stream says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class SlowStream(FakeCompletions):
    """Yields chunks after known delays (seconds) on the fake clock."""

    def __init__(self, plan: list[tuple[float, NS]], clock: Clock):
        super().__init__()
        self.plan, self.clock = plan, clock

    def create(self, **kwargs: Any) -> Any:
        def gen() -> Iterator[NS]:
            for delay, c in self.plan:
                self.clock.now += delay
                yield c

        return gen()


def test_ttft_and_chunk_timing_from_a_stream_with_known_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "perf_counter", clock)
    seen: list[CallRecord] = []
    plan = [
        (0.05, chunk(None, reasoning="thinking")),
        (0.10, chunk("Hello")),
        (0.05, chunk(" there")),
        (0.05, chunk(" friend")),
        (0.0, chunk(None, "stop")),
    ]
    client = counterfix.guard(
        FakeClient(SlowStream(plan, clock)), config=Config(on_diagnosis=lambda r, d: seen.append(r))
    )
    list(client.chat.completions.create(model="m", messages=[], stream=True))
    t = seen[0].timing
    assert t is not None and t.chunks == 3
    assert t.first_reasoning_ms == pytest.approx(50, abs=5)
    assert t.ttft_ms == pytest.approx(150, abs=5)
    assert t.last_chunk_ms == pytest.approx(250, abs=5)
    assert t.mean_chunk_gap_ms == pytest.approx(50, abs=5)
    assert seen[0].latency_ms == pytest.approx(250, abs=5)


def test_calls_that_are_not_streamed_have_no_timing() -> None:
    seen: list[CallRecord] = []
    client = counterfix.guard(FakeClient(), config=Config(on_diagnosis=lambda r, d: seen.append(r)))
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert seen[0].timing is None and seen[0].latency_ms is not None


def test_percentile_matches_numpy_linear() -> None:
    data = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert percentile(data, 50) == 30.0
    assert percentile(data, 95) == pytest.approx(88.0)  # numpy.percentile(data, 95)
    assert percentile([], 50) is None
    assert percentile([7.0], 99) == 7.0


def rec(ms: float, out: int = 100, ttft: float | None = None, **kw: Any) -> CallRecord:
    return CallRecord(
        provider="groq",
        model="m1",
        latency_ms=ms,
        usage=Usage(prompt_tokens=1000, completion_tokens=out),
        timing=Timing(ttft_ms=ttft, last_chunk_ms=ms, chunks=out) if ttft is not None else None,
        **kw,
    )


def test_tokens_per_second_counts_from_the_first_chunk_when_streamed() -> None:
    assert tokens_per_second(rec(2000, out=100)) == pytest.approx(50)
    assert tokens_per_second(rec(2000, out=100, ttft=1000)) == pytest.approx(100)
    assert tokens_per_second(CallRecord(latency_ms=10)) is None


def test_cost_needs_a_price_and_is_never_guessed() -> None:
    r = rec(1000)
    assert cost_usd(r, {}) is None
    assert cost_usd(r, {"groq:m1": (1.0, 2.0)}) == pytest.approx((1000 * 1 + 100 * 2) / 1e6)
    assert cost_usd(r, {"m1": (1.0, 0.0)}) == pytest.approx(0.001)
    set_prices({"m1": (2.0, 0.0)}, replace=True)
    try:
        assert cost_usd(r) == pytest.approx(0.002)
    finally:
        set_prices({}, replace=True)


def test_speed_summary_percentiles_errors_and_cost() -> None:
    records = [rec(ms) for ms in (100, 200, 300, 400)] + [
        rec(50, ttft=20),
        CallRecord(model="m1", error="RateLimitError: 429"),
    ]
    s = speed_summary(records, {"m1": (1.0, 1.0)})
    assert s.calls == 6 and s.errors == 1 and s.error_rate == pytest.approx(1 / 6)
    assert s.latency_ms.n == 5 and s.latency_ms.p50 == 200
    assert s.ttft_ms.n == 1 and s.ttft_ms.p50 == 20
    assert s.input_tokens == 5000 and s.unpriced == 1
    text = s.table()
    assert "latency" in text and "p95" in text and "cost: $" in text


class TimedCaller:
    """Answers with a set latency depending on what was changed."""

    def __init__(self, name: str, latency: float, concise_latency: float | None = None):
        self._name, self.latency, self.concise = name, latency, concise_latency
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> CallRecord:
        self.calls.append(params)
        concise = any("concisely" in (m.text if isinstance(m, Message) else "") for m in messages)
        ms = self.concise if concise and self.concise is not None else self.latency
        if params.get("reasoning_effort") == "low":
            ms = ms / 4
        return CallRecord(
            model=self._name,
            messages=[m if isinstance(m, Message) else Message(**m) for m in messages],
            response_text="Fine answer.",
            finish_reason="stop",
            latency_ms=ms,
        )


def slow_record(reasoning: bool) -> CallRecord:
    return CallRecord(
        provider="x",
        model="slow",
        messages=[Message(role="user", content="Explain tides.")],
        response_text="Tides are...",
        reasoning_text="let me think" if reasoning else None,
        finish_reason="stop",
        latency_ms=8000,
    )


def test_speed_fix_lowers_reasoning_effort_first_for_reasoning_models() -> None:
    caller = TimedCaller("slow", latency=8000)
    result = speed_fix(slow_record(reasoning=True), 3000, caller=caller)
    assert result.retry is not None and not result.retry.verified
    assert result.fix is not None and result.fix.name == "lower_reasoning_effort"
    assert result.fix.changed == {"reasoning_effort": "low"}
    assert "VERIFIED" in result.explain()


def test_speed_fix_tries_concise_then_another_model() -> None:
    caller = TimedCaller("slow", latency=8000, concise_latency=7000)
    fast = TimedCaller("fast-model", latency=900)
    result = speed_fix(slow_record(reasoning=False), 3000, caller=caller, alternatives=[fast])
    assert [a.name for a in result.attempts] == ["ask_concise", "use fast-model"]
    assert result.fix is not None and result.fix.changed == {"model": "fast-model"}


def test_speed_fix_reports_a_one_off_when_a_plain_retry_is_fast() -> None:
    result = speed_fix(slow_record(reasoning=False), 3000, caller=TimedCaller("slow", 500))
    assert result.fix is None and "one-off" in result.stopped


def test_a_fast_answer_that_is_cut_off_is_not_a_fix() -> None:
    class CutOff(TimedCaller):
        def complete(
            self, messages: Sequence[Message | dict[str, Any]], **params: Any
        ) -> CallRecord:
            r = super().complete(messages, **params)
            r.finish_reason = "length"
            return r

    result = speed_fix(slow_record(reasoning=True), 3000, caller=CutOff("slow", 8000))
    effort = result.attempts[0]
    assert effort.name == "lower_reasoning_effort" and not effort.verified
    assert any("F1" in s for s in effort.seen)


def test_otel_metrics_use_genai_names_in_seconds() -> None:
    recorded: dict[str, list[tuple[float, dict[str, Any]]]] = {}

    class Hist:
        def __init__(self, name: str):
            self.name = name

        def record(self, value: float, attrs: dict[str, Any]) -> None:
            recorded.setdefault(self.name, []).append((value, attrs))

    meter = NS(create_histogram=lambda name, **kw: Hist(name))
    hook = counterfix.otel.metrics(meter)
    r = rec(1500, out=40, ttft=300)
    hook(r, counterfix.diagnose(r, judge=False))
    assert recorded["gen_ai.client.operation.duration"][0][0] == pytest.approx(1.5)
    assert recorded["gen_ai.client.operation.time_to_first_chunk"][0][0] == pytest.approx(0.3)
    kinds = {a["gen_ai.token.type"]: v for v, a in recorded["gen_ai.client.token.usage"]}
    assert kinds == {"input": 1000, "output": 40}
    assert recorded["gen_ai.client.operation.duration"][0][1]["gen_ai.request.model"] == "m1"


def test_hooks_runs_each_hook_in_order() -> None:
    order: list[str] = []
    both = counterfix.hooks(lambda r, d: order.append("a"), None, lambda r, d: order.append("b"))
    r = CallRecord(response_text="x")
    both(r, counterfix.diagnose(r, judge=False))
    assert order == ["a", "b"]


def test_text_and_html_reports_show_speed_per_model() -> None:
    from counterfix.report import html_report, text_report

    records = [rec(ms, ttft=ms / 4) for ms in (100, 200, 300)]
    reports = [counterfix.diagnose(r, judge=False) for r in records]
    text = text_report(reports)
    assert "Speed and cost" in text and "[groq:m1]" in text
    assert "<h2>Speed and cost</h2>" in html_report(reports)


def test_tool_call_streams_get_a_ttft_and_speed_tails_are_low_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "perf_counter", clock)
    call = NS(index=0, id="c1", function=NS(name="f", arguments='{"a": 1}'))
    plan = [(0.2, chunk(None, tool_calls=[call])), (0.0, chunk(None, "tool_calls"))]
    seen: list[CallRecord] = []
    client = counterfix.guard(
        FakeClient(SlowStream(plan, clock)), config=Config(on_diagnosis=lambda r, d: seen.append(r))
    )
    list(client.chat.completions.create(model="m", messages=[], stream=True))
    assert seen[0].timing is not None and seen[0].timing.ttft_ms == pytest.approx(200, abs=5)
    s = speed_summary([rec(1000, out=n) for n in (10, 50, 100, 200, 400)])
    assert s.tokens_per_s.p5 is not None and s.tokens_per_s.p5 < s.tokens_per_s.p50
    assert "columns p50/p5/p1" in s.table()
