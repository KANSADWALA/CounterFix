"""Speed and cost: TTFT, latency percentiles, tokens per second, cost per call, and
verified speed fixes.

>>> client = counterfix.guard(OpenAI(), config=Config(on_diagnosis=log_to("calls.jsonl")))
...                                                            # doctest: +SKIP
>>> print(counterfix.perf.speed_summary(load_log("calls.jsonl")).table())  # doctest: +SKIP

Streamed calls carry ``record.timing`` (time to first chunk, last chunk, chunk count);
every call carries ``latency_ms`` and, when the provider reports it, token ``usage``.
Prices are never guessed: set the ones you pay with :func:`set_prices`, and calls of
models without a price are counted as unpriced.

:func:`speed_fix` is the counterfactual engine for slowness: it re-runs a slow call with
one change at a time (lower reasoning effort, a concise-answer instruction, another
model you name) and keeps a change only when the call meets the latency target in 2 of
3 re-runs without a new failure.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.caller import ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import run_detectors
from counterfix.core.record import CallRecord, Message

# --- single-call measures ------------------------------------------------------------------


def output_tokens(record: CallRecord) -> int | None:
    """Output tokens the provider reported (reasoning tokens included where it counts them)."""
    if record.usage is None:
        return None
    return record.usage.completion_tokens


def tokens_per_second(record: CallRecord) -> float | None:
    """Output tokens per second of generation.

    For streamed calls the clock starts at the first chunk, so waiting time before the
    model starts writing (queueing, prompt processing) is not counted; otherwise it is
    the whole call. None when tokens or timing are missing.
    """
    tokens = output_tokens(record)
    if not tokens or record.latency_ms is None:
        return None
    start = 0.0
    t = record.timing
    if t is not None:
        first = [x for x in (t.first_reasoning_ms, t.ttft_ms) if x is not None]
        start = min(first) if first else 0.0
    seconds = (record.latency_ms - start) / 1000
    return tokens / seconds if seconds > 0 else None


# --- prices ----------------------------------------------------------------------------------

Price = tuple[float, float]
"""(USD per million input tokens, USD per million output tokens)."""

_PRICES: dict[str, Price] = {}


def set_prices(prices: Mapping[str, Price], *, replace: bool = False) -> None:
    """Tell counterfix what you pay, per million tokens: ``{"groq:openai/gpt-oss-120b":
    (input, output)}``. Keys are ``provider:model`` or a bare model name. counterfix ships
    no prices, since they change and differ by account."""
    if replace:
        _PRICES.clear()
    for key, (inp, out) in prices.items():
        _PRICES[key] = (float(inp), float(out))


def price_for(record: CallRecord, prices: Mapping[str, Price] | None = None) -> Price | None:
    table = _PRICES if prices is None else prices
    if record.provider and record.model and f"{record.provider}:{record.model}" in table:
        return table[f"{record.provider}:{record.model}"]
    if record.model and record.model in table:
        return table[record.model]
    return None


def cost_usd(record: CallRecord, prices: Mapping[str, Price] | None = None) -> float | None:
    """What one call cost, or None when its price or token counts are unknown."""
    price = price_for(record, prices)
    usage = record.usage
    if price is None or usage is None or usage.prompt_tokens is None:
        return None
    out = usage.completion_tokens or 0
    return (usage.prompt_tokens * price[0] + out * price[1]) / 1_000_000


# --- many calls -------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile (0-100), linear between ranks (as numpy's default)."""
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q / 100
    low = math.floor(pos)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (pos - low)


class Spread(BaseModel):
    """Percentiles of one measure, with how many calls it is based on. For times the bad
    tail is high (p95, p99); for speeds it is low (p5, p1)."""

    n: int = 0
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    p5: float | None = None
    p1: float | None = None

    @classmethod
    def of(cls, values: Iterable[float | None]) -> Spread:
        data = [v for v in values if v is not None]
        return cls(
            n=len(data),
            p50=percentile(data, 50),
            p95=percentile(data, 95),
            p99=percentile(data, 99),
            p5=percentile(data, 5),
            p1=percentile(data, 1),
        )


class SpeedSummary(BaseModel):
    """Speed and cost over a set of calls (one model or many)."""

    calls: int = 0
    errors: int = 0
    latency_ms: Spread = Field(default_factory=Spread)
    ttft_ms: Spread = Field(default_factory=Spread)
    tokens_per_s: Spread = Field(default_factory=Spread)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    """Total cost of the priced calls; None when no call had a price."""
    unpriced: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0

    def table(self) -> str:
        def row(name: str, s: Spread, unit: str, slow_is_low: bool = False) -> str:
            if not s.n:
                return f"  {name:<14} -"
            tail = (s.p50, s.p5, s.p1) if slow_is_low else (s.p50, s.p95, s.p99)
            vals = " ".join(f"{x:>9.0f}" if x is not None else f"{'-':>9}" for x in tail)
            note = ", columns p50/p5/p1" if slow_is_low else ""
            return f"  {name:<14} {vals}  {unit} (n={s.n}{note})"

        lines = [
            f"{self.calls} call(s), {self.errors} error(s); slow tail in the last two columns",
            f"  {'':<14} {'p50':>9} {'p95':>9} {'p99':>9}",
            row("latency", self.latency_ms, "ms"),
            row("TTFT", self.ttft_ms, "ms"),
            row("output speed", self.tokens_per_s, "tokens/s", slow_is_low=True),
        ]
        counted = self.input_tokens or self.output_tokens
        if counted:
            lines.append(f"  tokens: {self.input_tokens} in, {self.output_tokens} out")
        else:
            lines.append("  tokens: not reported")
        if self.cost_usd is not None:
            extra = f" ({self.unpriced} call(s) without a price)" if self.unpriced else ""
            lines.append(f"  cost: ${self.cost_usd:.4f}{extra}")
        elif self.calls and counted:
            lines.append("  cost: unknown (set prices with counterfix.perf.set_prices)")
        return "\n".join(lines)


def speed_summary(
    records: Iterable[CallRecord], prices: Mapping[str, Price] | None = None
) -> SpeedSummary:
    """Percentiles of latency, TTFT and output speed, token totals and cost."""
    items = list(records)
    costs = [cost_usd(r, prices) for r in items]
    priced = [c for c in costs if c is not None]
    ok = [r for r in items if r.error is None]
    return SpeedSummary(
        calls=len(items),
        errors=len(items) - len(ok),
        latency_ms=Spread.of(r.latency_ms for r in ok),
        ttft_ms=Spread.of(r.timing.ttft_ms if r.timing else None for r in ok),
        tokens_per_s=Spread.of(tokens_per_second(r) for r in ok),
        input_tokens=sum((r.usage.prompt_tokens or 0) for r in items if r.usage),
        output_tokens=sum((r.usage.completion_tokens or 0) for r in items if r.usage),
        cost_usd=sum(priced) if priced else None,
        unpriced=len(items) - len(priced),
    )


def by_model(records: Iterable[CallRecord]) -> dict[str, list[CallRecord]]:
    """Calls grouped by ``provider:model`` (or model), for per-model summaries."""
    groups: dict[str, list[CallRecord]] = {}
    for r in records:
        key = f"{r.provider}:{r.model}" if r.provider else (r.model or "unknown")
        groups.setdefault(key, []).append(r)
    return groups


# --- verified speed fixes ---------------------------------------------------------------------


class SpeedAttempt(BaseModel):
    """One change (or the plain retry) and how fast its re-runs were."""

    name: str
    description: str
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] | None = None
    changed: dict[str, Any] = Field(default_factory=dict)
    runs: int = 0
    passed: int = 0
    latencies_ms: list[float] = Field(default_factory=list)
    verified: bool = False
    p_value: float | None = None
    """For a change: the one-sided Fisher exact p-value against the plain retry."""
    errors: list[str] = Field(default_factory=list)
    seen: list[str] = Field(default_factory=list)
    """New failures that made a fast run not count, e.g. ["F1 Truncation"]."""

    @property
    def observed(self) -> int:
        return self.runs - len(self.errors)

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.latencies_ms) if self.latencies_ms else None


class SpeedFixResult(BaseModel):
    record_id: str
    target_ms: float
    retry: SpeedAttempt | None = None
    attempts: list[SpeedAttempt] = Field(default_factory=list)
    fix: SpeedAttempt | None = None
    calls: int = 0
    stopped: str = ""

    def explain(self) -> str:
        def line(a: SpeedAttempt) -> str:
            med = f", median {a.median_ms:.0f} ms" if a.median_ms is not None else ""
            mark = "VERIFIED" if a.verified else "no"
            still = f" (new failures: {', '.join(sorted(set(a.seen)))})" if a.seen else ""
            under = f"under {self.target_ms:.0f} ms in {a.passed}/{a.observed}"
            if a.p_value is not None:
                under += f" (p={a.p_value:.3f})"
            return f"{a.name}: {under}{med} -> {mark}{still}"

        lines = []
        if self.retry is not None:
            lines.append("plain retry: " + line(self.retry).split(": ", 1)[1])
        lines += ["tried " + line(a) for a in self.attempts]
        if self.fix is not None:
            lines.append(f"fix: {self.fix.description}")
            if self.fix.changed:
                lines.append(f"  change: {self.fix.changed}")
        lines.append(f"{self.calls} re-run(s); {self.stopped}")
        return "\n".join(lines)


_CONCISE = "Answer concisely: give the answer and only the explanation it needs."


def _concise(messages: list[Message]) -> list[Message]:
    out = copy.deepcopy(messages)
    if out and out[0].role in ("system", "developer") and isinstance(out[0].content, str):
        out[0].content = f"{out[0].content}\n\n{_CONCISE}"
    else:
        out.insert(0, Message(role="system", content=_CONCISE))
    return out


def speed_fix(
    record: CallRecord,
    target_ms: float,
    *,
    caller: ModelCaller | None = None,
    alternatives: Sequence[ModelCaller] = (),
    runs: int = 6,
    need: int | None = None,
    max_runs: int = 30,
    check_retry: bool = True,
    alpha: float = 0.05,
) -> SpeedFixResult:
    """Find a verified change that brings a slow call under ``target_ms``.

    A run counts only if it is under the target and the live checks find no failure that
    the original call did not have (a faster answer that is cut off is not a fix). Tried
    in order: lower reasoning effort (for reasoning models), a concise-answer instruction,
    then each caller in ``alternatives`` (other models you would accept). ``caller``
    re-runs the original model (default: the provider preset named on the record).

    A change is verified when it meets the target clearly more often than the plain
    retry, by the same test as :func:`counterfix.fix` (see :mod:`counterfix.fixes.verify`).
    ``need`` is no longer used and only warns.
    """
    import warnings

    from counterfix.fixes.verify import beatable, level, p_value, settled

    if need is not None:
        warnings.warn(
            "speed_fix(need=...) is ignored: a change is now verified when it beats the "
            "plain retry (see counterfix.fixes.verify)",
            DeprecationWarning,
            stacklevel=2,
        )
    from counterfix.detect import default_detectors
    from counterfix.fixes.engine import _caller_for

    rerun = caller or _caller_for(record, None)
    result = SpeedFixResult(record_id=record.id, target_ms=target_ms)
    before = {
        f.failure_type for f in run_detectors(record, default_detectors(), Config()).model_failures
    }
    checks = default_detectors()

    tests = 2 + len(alternatives)  # lower effort (when it applies), concise, alternatives
    at = level(alpha, tests)
    base = [0, runs if not check_retry else 0]

    def trial(
        attempt: SpeedAttempt, who: ModelCaller, messages: list[Message], is_retry: bool = False
    ) -> SpeedAttempt:
        for _ in range(runs):
            if result.calls >= max_runs:
                break
            result.calls += 1
            attempt.runs += 1
            try:
                new = who.complete(messages, **attempt.params)
            except Exception as exc:
                attempt.errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if new.latency_ms is not None:
                attempt.latencies_ms.append(new.latency_ms)
            left = {f.failure_type for f in run_detectors(new, checks, Config()).model_failures}
            fast = new.latency_ms is not None and new.latency_ms <= target_ms
            if fast and left <= before:
                attempt.passed += 1
            elif fast:
                attempt.seen += [f"{t.value} {t.label}" for t in sorted(left - before, key=str)]
            if is_retry:
                if not beatable(attempt.passed, attempt.observed, runs, at):
                    break
            elif settled(
                base[0],
                base[1],
                attempt.passed,
                attempt.observed,
                attempt.observed + (runs - attempt.runs),
                at,
            ):
                break
        if is_retry:
            attempt.verified = not beatable(attempt.passed, attempt.observed, runs, at)
        else:
            attempt.p_value = p_value(base[0], base[1], attempt.passed, attempt.observed)
            attempt.verified = attempt.passed > 0 and attempt.p_value <= at
        return attempt

    if check_retry:
        retry = SpeedAttempt(
            name="plain_retry", description="Re-run unchanged", params=record.params
        )
        result.retry = trial(retry, rerun, record.messages, is_retry=True)
        base[0], base[1] = retry.passed, retry.observed
        if retry.verified:
            result.stopped = "a plain retry is fast enough too: the slow call was a one-off"
            return result

    plans: list[tuple[SpeedAttempt, ModelCaller, list[Message]]] = []
    if record.has_reasoning and record.params.get("reasoning_effort") != "low":
        params = {**record.params, "reasoning_effort": "low"}
        plans.append(
            (
                SpeedAttempt(
                    name="lower_reasoning_effort",
                    description="Lower the reasoning effort (reasoning_effort='low')",
                    params=params,
                    changed={"reasoning_effort": "low"},
                ),
                rerun,
                record.messages,
            )
        )
    concise = _concise(record.messages)
    plans.append(
        (
            SpeedAttempt(
                name="ask_concise",
                description="Ask for a concise answer (fewer output tokens)",
                params=record.params,
                messages=concise,
            ),
            rerun,
            concise,
        )
    )
    for other in alternatives:
        plans.append(
            (
                SpeedAttempt(
                    name=f"use {other.name}",
                    description=f"Send this call to {other.name}",
                    model=other.name,
                    params=record.params,
                    changed={"model": other.name},
                ),
                other,
                record.messages,
            )
        )
    for attempt, who, messages in plans:
        if result.calls >= max_runs:
            result.stopped = f"stopped at the budget of {max_runs} re-runs"
            return result
        result.attempts.append(trial(attempt, who, messages))
        if attempt.verified:
            result.fix = attempt
            result.stopped = "stopped at the first verified fix"
            return result
    result.stopped = "no change met the target clearly more often than a plain retry"
    return result


__all__ = [
    "Price",
    "SpeedAttempt",
    "SpeedFixResult",
    "SpeedSummary",
    "Spread",
    "by_model",
    "cost_usd",
    "output_tokens",
    "percentile",
    "price_for",
    "set_prices",
    "speed_fix",
    "speed_summary",
    "tokens_per_second",
]
