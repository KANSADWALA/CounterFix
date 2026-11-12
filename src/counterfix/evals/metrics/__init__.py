"""Metrics: how good an answer is, beside the detectors that say what went wrong.

>>> from counterfix.evals.metrics import get_metric
>>> m = get_metric("f1:0.6")          # name, optionally ":threshold"
>>> m.score(record, reference="the Eiffel Tower").value    # doctest: +SKIP

Names: exact_match, f1, contains, choice, number, json_valid, json_match, latency, ttft;
retrieval (gold labels): hit, recall, precision, mrr, ndcg (each also as name@k); code
(runs the code, opt-in): tests; token probabilities: confidence; tool use: tool_call;
prompt injection (its probe set): injection.

These are the metrics counterfix's diagnosis and fixes rely on. For ROUGE, BLEU, embedding
scores, judge-scored RAG metrics, PII or toxicity, use your evaluation tool (DeepEval,
Ragas, Promptfoo ...), or wrap a scorer in your own :class:`Metric`.
"""

from __future__ import annotations

from typing import Any

from counterfix._internal.suggest import did_you_mean
from counterfix.evals.metrics.base import Metric, MetricResult, references
from counterfix.evals.metrics.code import TestsPass
from counterfix.evals.metrics.confidence import Confidence
from counterfix.evals.metrics.extract import (
    ChoiceMatch,
    NumberMatch,
    extract_choice,
    extract_code,
    extract_number,
)
from counterfix.evals.metrics.retrieval import HitRate, Mrr, Ndcg, Precision, Recall
from counterfix.evals.metrics.speed import Latency, Ttft
from counterfix.evals.metrics.structured import JsonMatch, JsonValid, parse_json
from counterfix.evals.metrics.text import (
    Contains,
    ExactMatch,
    TokenF1,
    normalize,
    token_f1,
)
from counterfix.evals.metrics.tools import ToolCallMatch

METRICS: dict[str, type[Metric]] = {
    m.name: m
    for m in (
        ExactMatch,
        TokenF1,
        Contains,
        ChoiceMatch,
        NumberMatch,
        JsonValid,
        JsonMatch,
        Latency,
        Ttft,
        HitRate,
        Recall,
        Precision,
        Mrr,
        Ndcg,
        TestsPass,
        Confidence,
        ToolCallMatch,
    )
}


def _benchmark_metrics() -> None:
    """Metrics that score against counterfix's probe sets."""
    from counterfix.evals.benchmarks.injection import InjectionResisted

    METRICS[InjectionResisted.name] = InjectionResisted


_benchmark_metrics()
RANKED = {"hit", "recall", "precision", "mrr", "ndcg"}
"""Metrics that take ``@k`` (the number of top-ranked passages to look at)."""


def get_metric(spec: str | Metric, **options: Any) -> Metric:
    """A metric by name, e.g. ``"exact_match"``, ``"f1:0.6"``, ``"latency:2000"`` or
    ``"recall@5:0.8"`` (``@k``: top k passages; after the colon: the threshold). Metric
    objects pass through."""
    if isinstance(spec, Metric):
        return spec
    name, _, threshold = spec.partition(":")
    name, at, k = name.strip().partition("@")
    cls = METRICS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown metric {name!r}{did_you_mean(name, METRICS)}; "
            f"choose from {', '.join(sorted(METRICS))}"
        )
    if at:
        if name not in RANKED or not k.isdigit() or int(k) < 1:
            raise ValueError(f"{spec!r}: @k works only with {', '.join(sorted(RANKED))}, k >= 1")
        options["k"] = int(k)
    return cls(float(threshold) if threshold else None, **options)


__all__ = [
    "METRICS",
    "ChoiceMatch",
    "Confidence",
    "Contains",
    "ExactMatch",
    "HitRate",
    "JsonMatch",
    "JsonValid",
    "Latency",
    "Metric",
    "MetricResult",
    "Mrr",
    "Ndcg",
    "NumberMatch",
    "Precision",
    "Recall",
    "TestsPass",
    "TokenF1",
    "ToolCallMatch",
    "Ttft",
    "extract_choice",
    "extract_code",
    "extract_number",
    "get_metric",
    "normalize",
    "parse_json",
    "references",
    "token_f1",
]
