"""The metric interface: a detector says what went wrong, a metric says how good.

>>> from counterfix.metrics import get_metric
>>> get_metric("exact_match").score(record, reference="Paris")    # doctest: +SKIP
MetricResult(name='exact_match', value=1.0, passed=True, ...)

A metric turns one call (and, for most metrics, a reference answer) into a number.
With a threshold it also says pass or fail, and a failed metric that points at a
failure type (``failure_type``) goes straight to :func:`counterfix.fix`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType


class MetricResult(BaseModel):
    """One metric on one call."""

    name: str
    value: float | None = None
    """None when the metric could not be computed (no reference, the call failed...)."""
    passed: bool | None = None
    """True/False against the threshold; None when there is no threshold or no value
    (not scored). A failed call is always False."""
    threshold: float | None = None
    higher_is_better: bool = True
    reason: str = ""
    failure_type: FailureType | None = None
    """The failure type a miss points to (e.g. F2 for invalid JSON), when it is certain."""


class Metric:
    """Base class. Subclasses set ``name`` and implement :meth:`compute`.

    ``threshold`` turns the value into pass/fail: ``value >= threshold`` when higher is
    better, ``value <= threshold`` otherwise. Metrics that are right or wrong (exact
    match, valid JSON) default to a threshold of 1.0; graded ones (F1, ROUGE-L) have
    none unless you set one, so they report a number without failing anything.
    """

    name = "metric"
    higher_is_better = True
    needs_reference = True
    default_threshold: float | None = None
    failure_type: FailureType | None = None

    def __init__(self, threshold: float | None = None, **options: Any):
        self.threshold = self.default_threshold if threshold is None else threshold
        self.options = options

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        """Return (value, reason). Raise nothing for ordinary misses: return a value."""
        raise NotImplementedError

    def score(self, record: CallRecord, reference: Any = None) -> MetricResult:
        result = MetricResult(
            name=self.name, threshold=self.threshold, higher_is_better=self.higher_is_better
        )
        if record.error is not None:
            result.passed = False
            result.reason = f"the call failed: {record.error[:120]}"
            return result
        if self.needs_reference and reference is None:
            result.reason = "no reference answer to compare with"
            return result
        value, reason = self.compute(record, reference)
        result.value, result.reason = value, reason
        if value is not None and self.threshold is not None:
            ok = value >= self.threshold if self.higher_is_better else value <= self.threshold
            result.passed = ok
            if not ok:
                result.failure_type = self.failure_type
        # value None: the metric could not be computed (no TTFT on an unstreamed call, a
        # reference without a number...). That is "not scored", not a miss by the model.
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(threshold={self.threshold!r})"


def references(reference: Any) -> list[str]:
    """One reference or several (a list) as a list of strings."""
    if reference is None:
        return []
    if isinstance(reference, (list, tuple)):
        return [str(r) for r in reference]
    return [str(reference)]
