"""Scoring an answer against its metrics, and the fix search for one missed item."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from counterfix.core.caller import DailyLimitError, ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import run_detectors
from counterfix.core.diagnosis import Diagnosis, Evidence, Finding
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType
from counterfix.evals.metrics import Metric, MetricResult
from counterfix.evals.results import FixTry, ItemFix


def score(record: CallRecord, reference: Any, metrics: Sequence[Metric]) -> list[MetricResult]:
    return [m.score(record, reference) for m in metrics]


def _passed(record: CallRecord, results: Sequence[MetricResult]) -> bool:
    return record.error is None and all(r.passed is not False for r in results)


def _with_metric_findings(diagnosis: Diagnosis, results: Sequence[MetricResult]) -> Diagnosis:
    """Add a finding for each failed metric that names a failure type (e.g. F2)."""
    have = {f.failure_type for f in diagnosis.findings}
    for r in results:
        if r.passed is False and r.failure_type is not None and r.failure_type not in have:
            diagnosis.findings.append(
                Finding(
                    failure_type=r.failure_type,
                    confidence=0.9,
                    evidence=[Evidence(detector=f"metric:{r.name}", detail=r.reason)],
                )
            )
            have.add(r.failure_type)
    from counterfix.core.precedence import resolve

    # a metric can name the root cause (F13, F12): let it supersede live findings it explains
    kept, suppressed = resolve(diagnosis.findings)
    diagnosis.findings = kept
    diagnosis.suppressed = [*diagnosis.suppressed, *suppressed]
    return diagnosis


def _metric_miss(r: MetricResult) -> str:
    """'latency over 5000' / 'exact_match missed': short, so repeats can be counted."""
    if r.threshold is not None and r.value is not None and not r.higher_is_better:
        return f"{r.name} over {r.threshold:g}"
    return f"{r.name} missed"


def fix_item(
    record: CallRecord,
    reference: Any,
    metrics: Sequence[Metric],
    diagnosis: Diagnosis,
    caller: ModelCaller,
    *,
    runs: int = 6,
    max_runs: int = 30,
    must_clear: set[FailureType] | None = None,
    on_run: Callable[[int, str], None] | None = None,
    alpha: float = 0.05,
) -> ItemFix:
    """Search for a change that makes this item pass clearly more often than a plain
    retry (see :mod:`counterfix.fixes.verify`).

    A plain retry runs first; if it passes too often for any change to beat, no change
    is claimed. Then the
    engine's interventions for each cause found, most confident cause first. A run
    counts only when every metric passes and the live checks find nothing new; with
    ``must_clear`` (a flagged item, whose metrics already pass) those failure types must
    also be gone. ``on_run(calls, change)`` is called before each re-run (for progress).
    """
    from counterfix.detect import default_detectors
    from counterfix.fixes.engine import INTERVENTIONS
    from counterfix.fixes.verify import beatable, level, p_value, settled

    result = ItemFix()
    checks = default_detectors()
    before = {f.failure_type for f in diagnosis.model_failures}
    causes = {f.failure_type for f in diagnosis.model_failures}
    at = level(alpha, sum(1 for i in INTERVENTIONS if i.failure_type in causes))
    base = [0, 0]

    def trial(
        attempt: FixTry, messages: list[Message], params: dict[str, Any], is_retry: bool = False
    ) -> FixTry:
        for _ in range(runs):
            if result.calls >= max_runs:
                break
            result.calls += 1
            attempt.runs += 1
            if on_run is not None:
                on_run(result.calls, attempt.name)
            try:
                new = caller.complete(messages, **params)
            except DailyLimitError:
                raise
            except Exception as exc:
                attempt.why_not.append(f"call failed ({type(exc).__name__})")
                attempt.errors += 1
                continue  # an unreadable run is not evidence either way
            new.metadata = {**record.metadata, **new.metadata, **attempt.metadata}
            left = {f.failure_type for f in run_detectors(new, checks, Config()).model_failures}
            still = (must_clear or set()) & left
            results = score(new, reference, metrics)
            if _passed(new, results) and left <= before and not still:
                attempt.passed += 1
            else:
                attempt.why_not += [_metric_miss(r) for r in results if r.passed is False]
                attempt.why_not += [f"still {t.value}" for t in sorted(still, key=str)]
                attempt.why_not += [f"new {t.value}" for t in sorted(left - before, key=str)]
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

    result.retry = trial(
        FixTry(name="plain_retry", description="Re-run unchanged"),
        record.messages,
        record.params,
        is_retry=True,
    )
    base[0], base[1] = result.retry.passed, result.retry.observed
    if result.retry.verified:
        result.note = "a plain retry passes: the failure was not repeatable"
        return result
    failures = sorted(diagnosis.model_failures, key=lambda f: -f.confidence)
    if not failures:
        result.note = "no cause found, so no targeted change to try"
        return result
    for finding in failures:
        for iv in INTERVENTIONS:
            if iv.failure_type is not finding.failure_type:
                continue
            change = iv.apply(record, finding)
            if change is None:
                continue
            if result.calls >= max_runs:
                result.note = f"stopped at the budget of {max_runs} re-runs"
                return result
            messages, params = change
            attempt = FixTry(
                name=iv.name,
                description=iv.description,
                changed={k: v for k, v in params.items() if record.params.get(k) != v},
                messages=None if messages == record.messages else messages,
                metadata=iv.metadata(record, finding) if iv.metadata else {},
            )
            result.tried.append(trial(attempt, messages, params))
            if attempt.verified:
                result.fix = attempt
                result.note = "stopped at the first verified fix"
                return result
    result.note = result.note or "no change beat the plain retry clearly enough"
    return result


# --- the run ----------------------------------------------------------------------------------
