"""Metrics: how good an answer is, beside the detectors that say what went wrong.

>>> from counterfix.evals.metrics import get_metric
>>> m = get_metric("f1:0.6")          # name, optionally ":threshold"
>>> m.score(record, reference="the Eiffel Tower").value    # doctest: +SKIP

Names: exact_match, f1, contains, rouge_l, bleu, chrf, choice, number, json_valid,
json_match, embedding_similarity, bertscore, latency, ttft; retrieval (gold labels):
hit, recall, precision, mrr, ndcg (each also as name@k); RAG with a judge:
faithfulness, context_precision, context_recall, answer_relevancy, citations; any
task with a judge: answer_correctness; code (runs the code, opt-in): tests; token
probabilities: confidence; safety: pii, toxicity; tool use:
tool_call.
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
from counterfix.evals.metrics.rag import (
    AnswerCorrectness,
    AnswerRelevancy,
    Citations,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    JudgeMetric,
)
from counterfix.evals.metrics.retrieval import HitRate, Mrr, Ndcg, Precision, Recall
from counterfix.evals.metrics.safety import PiiLeak, Toxicity
from counterfix.evals.metrics.semantic import BertScore, EmbeddingSimilarity
from counterfix.evals.metrics.speed import Latency, Ttft
from counterfix.evals.metrics.structured import JsonMatch, JsonValid, parse_json
from counterfix.evals.metrics.text import (
    Bleu,
    Chrf,
    Contains,
    ExactMatch,
    RougeL,
    TokenF1,
    normalize,
    rouge_l,
    token_f1,
)
from counterfix.evals.metrics.tools import ToolCallMatch

METRICS: dict[str, type[Metric]] = {
    m.name: m
    for m in (
        ExactMatch,
        TokenF1,
        Contains,
        RougeL,
        Bleu,
        Chrf,
        ChoiceMatch,
        NumberMatch,
        JsonValid,
        JsonMatch,
        EmbeddingSimilarity,
        BertScore,
        Latency,
        Ttft,
        HitRate,
        Recall,
        Precision,
        Mrr,
        Ndcg,
        Faithfulness,
        ContextPrecision,
        ContextRecall,
        AnswerRelevancy,
        Citations,
        AnswerCorrectness,
        TestsPass,
        Confidence,
        PiiLeak,
        Toxicity,
        ToolCallMatch,
    )
}


def _benchmark_metrics() -> None:
    """Metrics that score against benchmark data (they import their library on use)."""
    from counterfix.evals.benchmarks.evalplus import EvalPlusTests
    from counterfix.evals.benchmarks.ifeval import IfEvalStrict
    from counterfix.evals.benchmarks.injection import InjectionResisted

    METRICS.update({m.name: m for m in (EvalPlusTests, IfEvalStrict, InjectionResisted)})


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
    "AnswerCorrectness",
    "AnswerRelevancy",
    "BertScore",
    "Bleu",
    "ChoiceMatch",
    "Chrf",
    "Citations",
    "Confidence",
    "Contains",
    "ContextPrecision",
    "ContextRecall",
    "EmbeddingSimilarity",
    "ExactMatch",
    "Faithfulness",
    "HitRate",
    "JsonMatch",
    "JsonValid",
    "JudgeMetric",
    "Latency",
    "Metric",
    "MetricResult",
    "Mrr",
    "Ndcg",
    "NumberMatch",
    "PiiLeak",
    "Precision",
    "Recall",
    "RougeL",
    "TestsPass",
    "TokenF1",
    "ToolCallMatch",
    "Toxicity",
    "Ttft",
    "extract_choice",
    "extract_code",
    "extract_number",
    "get_metric",
    "normalize",
    "parse_json",
    "references",
    "rouge_l",
    "token_f1",
]
