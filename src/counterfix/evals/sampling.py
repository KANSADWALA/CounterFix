"""Several samples per item (pass@k) and the guard for running generated code."""

from __future__ import annotations

from collections.abc import Sequence

from counterfix.evals.data import EvalItem
from counterfix.evals.metrics import Metric
from counterfix.evals.results import EvalResult


def _check_code_execution(metrics: Sequence[Metric]) -> None:
    """Fail before any call when a metric would run code that is not allowed to run."""
    from counterfix.detect.code.sandbox import CodeExecutionDisabled, execution_allowed

    for m in metrics:
        if getattr(m, "needs_code_execution", False) and not execution_allowed(
            m.options.get("allow")
        ):
            raise CodeExecutionDisabled()


def _repeat(items: Sequence[EvalItem], n: int) -> list[EvalItem]:
    """Each item n times, as ``id#1`` .. ``id#n`` (the base id kept in metadata)."""
    out: list[EvalItem] = []
    for item in items:
        for k in range(1, n + 1):
            copy = item.model_copy(deep=True)
            copy.id = f"{item.id}#{k}"
            copy.metadata[SAMPLE_KEY] = item.id
            out.append(copy)
    return out


SAMPLE_KEY = "sample_of"


def _pass_at_k(
    result: EvalResult, metrics: Sequence[Metric], n: int
) -> dict[str, dict[int, float]]:
    """pass@k per pass/fail metric over the questions, for k in 1, 5, 10, 100 up to n."""
    from counterfix.detect.code.passk import mean_pass_at_k

    ks = [k for k in (1, 5, 10, 100) if k <= n] or [1]
    out: dict[str, dict[int, float]] = {}
    for j, m in enumerate(metrics):
        if m.threshold is None:
            continue
        groups: dict[str, list[bool]] = {}
        for item in result.items:
            r = item.metrics[j]
            base = str(item.record.metadata.get(SAMPLE_KEY) or item.id.rsplit("#", 1)[0])
            groups.setdefault(base, []).append(r.passed is True)
        counts = [(len(v), sum(v)) for v in groups.values()]
        values = {k: v for k in ks if (v := mean_pass_at_k(counts, k)) is not None}
        if values:
            out[m.name] = values
    return out
