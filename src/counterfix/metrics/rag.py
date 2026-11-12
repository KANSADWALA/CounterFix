"""Metrics scored by a judge: faithfulness, context precision, context recall, answer
relevancy and citation correctness (RAG), and answer correctness (any task with a
reference answer).

They follow the RAGAS definitions (Es et al., 2023) with one judge call per metric per
answer: faithfulness = supported claims / claims; context recall = reference statements
attributable to the context / statements; context precision = mean of precision@k over
the ranks k of useful passages (rewards useful passages ranked first). The judge is the
one counterfix picks for diagnose (bring your own key) unless you bind one:
``get_metric("faithfulness").bind(judge)``. A judge failure leaves the item not scored.
"""

from __future__ import annotations

import json
import re
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.judges import tasks
from counterfix.judges.base import Judge, JudgeTask, Verdict
from counterfix.metrics.base import Metric

_CITE = re.compile(r"\[(\d{1,3})\]")


def _items(value: Any) -> list[dict[str, Any]]:
    """A judge's list field as dicts, tolerating a JSON string or stray entries."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _yes(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1", "supported", "attributed")


def _numbers(value: Any) -> set[int]:
    if isinstance(value, str):
        return {int(x) for x in re.findall(r"\d+", value)}
    if isinstance(value, list):
        return {int(x) for x in value if isinstance(x, (int, float)) or str(x).isdigit()}
    return set()


class JudgeMetric(Metric):
    """A metric that asks a judge. Bind one with :meth:`bind`, or ``evaluate()`` will."""

    needs_reference = False
    needs_context = True
    judge: Judge | None = None

    def bind(self, judge: Judge) -> JudgeMetric:
        self.judge = judge
        return self

    def task(self, record: CallRecord) -> JudgeTask:
        raise NotImplementedError

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        raise NotImplementedError

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        if self.judge is None:
            from counterfix.judges.auto import resolve_judge

            self.judge = resolve_judge(None, target_model=record.model)[0]
        from counterfix.detect.retrieval import has_context

        if self.needs_context and not has_context(record):
            return None, "no retrieved context on this call"
        if isinstance(reference, str) and tasks.REFERENCE_KEY not in record.metadata:
            record = record.model_copy(
                update={"metadata": {**record.metadata, tasks.REFERENCE_KEY: reference}}
            )
        verdict = self.judge.ask(self.task(record))
        if verdict.failed:
            return None, f"judge failed: {verdict.reason[:120]}"
        return self.from_verdict(verdict, record)


class Faithfulness(JudgeMetric):
    """Share of the answer's claims that the retrieved context supports, 0 to 1."""

    name = "faithfulness"

    def task(self, record: CallRecord) -> JudgeTask:
        return tasks.faithfulness(record)

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        claims = _items(verdict.extra.get("claims"))
        if claims:
            bad = [str(c.get("claim", ""))[:80] for c in claims if not _yes(c.get("supported"))]
            value = 1 - len(bad) / len(claims)
            why = f"{len(claims) - len(bad)}/{len(claims)} claims supported"
            return value, why + (f"; unsupported: {'; '.join(bad[:2])}" if bad else "")
        if verdict.label == "no_claims":
            return None, "the answer makes no claims to check"
        value = {"faithful": 1.0, "partly_faithful": 0.5, "unfaithful": 0.0}[verdict.label]
        return value, f"{verdict.label} (judge listed no claims): {verdict.reason}"


class ContextPrecision(JudgeMetric):
    """Are the useful passages ranked first? Mean precision@k at each useful rank, 0 to 1."""

    name = "context_precision"

    def task(self, record: CallRecord) -> JudgeTask:
        return tasks.context_precision(record)

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        from counterfix.detect.retrieval import chunks_of

        n = len(chunks_of(record)) or len(_CITE.findall(record.metadata[tasks.CONTEXT_KEY]))
        useful = {u for u in _numbers(verdict.extra.get("useful")) if 1 <= u <= max(n, 1)}
        if verdict.label == "none_useful" or not useful:
            return 0.0, "no passage was useful"
        hits, total = 0, 0.0
        for k in range(1, n + 1):
            if k in useful:
                hits += 1
                total += hits / k
        value = total / len(useful)
        return value, f"useful passages: {sorted(useful)} of {n}"


class ContextRecall(JudgeMetric):
    """Share of the reference answer's statements found in the context, 0 to 1."""

    name = "context_recall"
    needs_reference = True

    def task(self, record: CallRecord) -> JudgeTask:
        return tasks.context_recall(record)

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        statements = _items(verdict.extra.get("statements"))
        if statements:
            found = sum(_yes(s.get("attributed")) for s in statements)
            missing = [
                str(s.get("statement", ""))[:80]
                for s in statements
                if not _yes(s.get("attributed"))
            ]
            why = f"{found}/{len(statements)} reference statements in the context"
            return found / len(statements), why + (f"; missing: {missing[0]}" if missing else "")
        value = {"all_attributed": 1.0, "some_attributed": 0.5, "none_attributed": 0.0}[
            verdict.label
        ]
        return value, f"{verdict.label} (judge listed no statements)"


class AnswerRelevancy(JudgeMetric):
    """How directly the answer addresses the question: 1, 0.5 (partly) or 0."""

    name = "answer_relevancy"
    needs_context = False

    def task(self, record: CallRecord) -> JudgeTask:
        return tasks.answer_relevancy(record)

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        value = {"relevant": 1.0, "partly_relevant": 0.5, "irrelevant": 0.0}[verdict.label]
        return value, f"{verdict.label}: {verdict.reason}"


class Citations(JudgeMetric):
    """Share of cited sentences ([n]) that the cited passages support. Answers with no
    citations are not scored."""

    name = "citations"

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        if not _CITE.search(record.answer_text):
            return None, "the answer cites no passage"
        return super().compute(record, reference)

    def task(self, record: CallRecord) -> JudgeTask:
        return tasks.citations(record)

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        cites = _items(verdict.extra.get("citations"))
        if cites:
            ok = sum(_yes(c.get("supported")) for c in cites)
            return ok / len(cites), f"{ok}/{len(cites)} cited sentences supported"
        value = {"all_supported": 1.0, "some_unsupported": 0.5, "none_supported": 0.0}[
            verdict.label
        ]
        return value, f"{verdict.label} (judge listed no citations)"


class AnswerCorrectness(JudgeMetric):
    """Does the answer agree with the reference (meaning, not wording)? 1, 0.5 or 0; passes
    at 1 by default. Use it where exact or contained matches are too strict."""

    name = "answer_correctness"
    needs_reference = True
    needs_context = False
    default_threshold = 1.0

    def task(self, record: CallRecord) -> JudgeTask:
        return tasks.answer_correctness(record)

    def from_verdict(self, verdict: Verdict, record: CallRecord) -> tuple[float | None, str]:
        value = {"correct": 1.0, "partly_correct": 0.5, "incorrect": 0.0}[verdict.label]
        return value, f"{verdict.label}: {verdict.reason}"
