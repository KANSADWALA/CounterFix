"""What an evaluation returns: per-item results, metric summaries with confidence
intervals, and the fixes tried."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.record import CallRecord, Message
from counterfix.evals.metrics import MetricResult
from counterfix.runtime.perf import SpeedSummary


class FixTry(BaseModel):
    name: str
    description: str
    runs: int = 0
    passed: int = 0
    verified: bool = False
    p_value: float | None = None
    """For a change: the one-sided Fisher exact p-value against the plain retry."""
    errors: int = 0
    changed: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    why_not: list[str] = Field(default_factory=list)
    """Why runs that did not pass failed, one entry per reason per run, e.g.
    "latency over 5000" or "still F1"."""

    @property
    def observed(self) -> int:
        return self.runs - self.errors

    def reasons(self) -> str:
        """The reasons, counted: 'latency over 5000 in 3/3, still F1 in 1/3'."""
        counts = Counter(self.why_not)
        return ", ".join(f"{r} in {k}/{self.runs}" for r, k in counts.most_common(3))


class ItemFix(BaseModel):
    """The fix search on one missed item."""

    retry: FixTry | None = None
    tried: list[FixTry] = Field(default_factory=list)
    fix: FixTry | None = None
    calls: int = 0
    note: str = ""


class ItemResult(BaseModel):
    id: str
    record: CallRecord
    reference: Any = None
    metrics: list[MetricResult] = Field(default_factory=list)
    passed: bool = True
    flagged: bool = False
    """Passed its metrics, but the live checks found a model failure (e.g. a cut-off
    answer that still met a latency gate)."""
    causes: list[str] = Field(default_factory=list)
    """Failure types found on this item, e.g. ["F8"], most confident first."""
    fix: ItemFix | None = None

    @property
    def top_cause(self) -> str | None:
        return self.causes[0] if self.causes else None


class MetricSummary(BaseModel):
    name: str
    n: int = 0
    mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    passed: int = 0
    failed: int = 0
    not_scored: int = 0
    """Items the metric could not score (no reference, no TTFT...). Failed calls count as
    failed, not here; the mean is over scored items only."""
    threshold: float | None = None
    higher_is_better: bool = True

    @property
    def pass_rate(self) -> float | None:
        judged = self.passed + self.failed
        return self.passed / judged if judged else None


class EvalResult(BaseModel):
    name: str
    model: str | None = None
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    items: list[ItemResult] = Field(default_factory=list)
    metrics: list[MetricSummary] = Field(default_factory=list)
    by_cause: dict[str, int] = Field(default_factory=dict)
    """Missed items by their top cause; "none found" when no check explained the miss."""
    flagged_by_cause: dict[str, int] = Field(default_factory=dict)
    """Items that passed their metrics but that the live checks flagged, by top cause."""
    fixes: dict[str, int] = Field(default_factory=dict)
    """Missed items by the verified fix ("no verified fix", "plain retry passes" too)."""
    fix_tried: int = 0
    speed: SpeedSummary = Field(default_factory=SpeedSummary)
    judge: str | None = None
    judge_errors: list[str] = Field(default_factory=list)
    """Missed items the judge could not check (e.g. its daily limit), so their cause may
    be missing: "id: error"."""
    stopped: str = ""
    samples: int = 1
    """Answers per question (``samples=n``); items are then ``id#1`` .. ``id#n``."""
    pass_at_k: dict[str, dict[int, float]] = Field(default_factory=dict)
    """With samples > 1: per pass/fail metric, pass@k (unbiased estimator) for each k."""
    code_causes: dict[str, int] = Field(default_factory=dict)
    """Code that failed its tests, by sub-cause (syntax, wrong_output, timeout...)."""
    info: dict[str, Any] = Field(default_factory=dict)
    """About the run (benchmarks: source, licence, template version, seed, subset)."""
    notes: list[str] = Field(default_factory=list)
    """Things to check about the run itself (a fix search that could not run, gold
    passage ids that never match what was retrieved)."""

    @property
    def failed(self) -> list[ItemResult]:
        return [i for i in self.items if not i.passed]

    @property
    def flagged(self) -> list[ItemResult]:
        """Passed every metric, yet a live check found a model failure."""
        return [i for i in self.items if i.passed and i.flagged]

    def metric(self, name: str) -> MetricSummary:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(
            f"no metric {name!r} in this result ({', '.join(m.name for m in self.metrics)})"
        )

    def text(self) -> str:
        from counterfix.evals.report import text_report

        return text_report(self)

    def html(self) -> str:
        from counterfix.evals.report import html_report

        return html_report(self)

    def save(self, path: str | Path) -> Path:
        """Write the whole result as JSON."""
        target = Path(path)
        target.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return target


def _id_mismatch_note(items: Sequence[ItemResult]) -> str:
    """A note when items carry gold ``relevant`` ids but no retrieved passage ever matches
    one: almost always ids from a different scheme (the dataset's vs the retriever's)."""
    from counterfix.detect.retrieval import RELEVANT_KEY, chunks_of, grades

    labelled = retrieved = matched = 0
    seen_ids: list[str] = []
    gold_ids: list[str] = []
    for i in items:
        relevant = i.record.metadata.get(RELEVANT_KEY)
        chunks = chunks_of(i.record)
        if relevant is None or not chunks:
            continue
        labelled += 1
        retrieved += len(chunks)
        g = grades(i.record) or []
        matched += sum(1 for v in g if v > 0)
        seen_ids += [str(c.id) for c in chunks if c.id is not None][:2]
        labels = [relevant] if isinstance(relevant, str) else list(relevant)
        gold_ids += [str(x) for x in labels][:2]
    if labelled < 2 or matched or not seen_ids:
        return ""
    seen = ", ".join(dict.fromkeys(seen_ids[:4]))
    gold = ", ".join(dict.fromkeys(gold_ids[:4]))
    return (
        f"no retrieved passage matched a gold `relevant` id in any of {labelled} items "
        f"(retrieved ids like {seen}; labelled {gold}). If the ids come from different "
        "schemes, retrieval metrics and F10 are wrong: label with your retriever's ids, or "
        "with the passage text."
    )


# --- statistics ------------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], *, samples: int = 1000, seed: int = 0
) -> tuple[float | None, float | None]:
    """95% confidence interval of the mean by bootstrap (seeded, so runs repeat)."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def summarize(name: str, results: Iterable[MetricResult]) -> MetricSummary:
    rs = list(results)
    values = [r.value for r in rs if r.value is not None]
    low, high = bootstrap_ci(values)
    first = rs[0] if rs else None
    return MetricSummary(
        name=name,
        n=len(values),
        mean=statistics.fmean(values) if values else None,
        ci_low=low,
        ci_high=high,
        passed=sum(r.passed is True for r in rs),
        failed=sum(r.passed is False for r in rs),
        not_scored=sum(r.value is None and r.passed is not False for r in rs),
        threshold=first.threshold if first else None,
        higher_is_better=first.higher_is_better if first else True,
    )


# --- one item ---------------------------------------------------------------------------------
