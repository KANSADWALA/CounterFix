"""F10 retrieval miss and F11 context ignored: why a RAG answer went wrong.

* :class:`RetrievalLabelCheck` (live, free): with gold labels on the call, F10 when none
  of the gold passages was retrieved.
* :class:`ContextUseCheck` (judge): was the answer in the retrieved passages at all
  (if not: F10), and if it was, did the answer use it (if not: F11)? Its evidence names
  the passage that holds the answer, which the fix search moves to the front.

Both skip calls that carry no retrieved passages. F10 and F11 take precedence over F4: an
unsupported claim is the symptom, the missing or ignored passage the cause.
"""

from __future__ import annotations

from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detect.judged import _JudgeCheck
from counterfix.judges import tasks

_RETRIEVER_FIX = Prescription(
    kind=PrescriptionKind.ADD_CONTEXT,
    description=(
        "Fix retrieval, not the prompt: retrieve more passages, rewrite the query, or add "
        "keyword (hybrid) search so the passage with the answer is found"
    ),
)


class RetrievalLabelCheck:
    """F10 from gold labels: no gold passage among the retrieved ones."""

    name = "retrieval_labels"
    covers = (FailureType.RETRIEVAL_MISS,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        from counterfix.detect.retrieval import chunks_of, grades, total_relevant

        g, total = grades(record), total_relevant(record)
        if g is None or not total or not chunks_of(record):
            raise SkipCheck("no retrieved passages with gold labels")
        if any(x > 0 for x in g):
            return []
        return [
            Finding(
                failure_type=FailureType.RETRIEVAL_MISS,
                confidence=0.95,
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=f"none of the {total} gold passage(s) is among the "
                        f"{len(g)} retrieved",
                        data={"retrieved": len(g), "gold": total},
                    )
                ],
                prescription=_RETRIEVER_FIX,
            )
        ]


class ContextUseCheck(_JudgeCheck):
    """F10 / F11 by judge: answer missing from the context, or there but not used."""

    name = "judge_context_use"
    covers = (FailureType.RETRIEVAL_MISS, FailureType.CONTEXT_IGNORED)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        from counterfix.detect.retrieval import has_context

        if not has_context(record):
            raise SkipCheck("no retrieved context on this call")
        if not record.answer_text:
            raise SkipCheck("no answer to check")
        verdict = self._ask(tasks.context_use(record))
        try:
            passage = int(verdict.extra.get("passage") or 0)
        except (TypeError, ValueError):
            passage = 0
        if verdict.label == "context_lacks_answer":
            return [
                Finding(
                    failure_type=FailureType.RETRIEVAL_MISS,
                    confidence=round(verdict.confidence * 0.9, 4),
                    evidence=[
                        self._evidence(
                            verdict,
                            f"the retrieved passages do not hold the answer: {verdict.reason}",
                        )
                    ],
                    prescription=_RETRIEVER_FIX,
                )
            ]
        if verdict.label == "answer_ignores_context":
            where = f" (passage [{passage}])" if passage else ""
            return [
                Finding(
                    failure_type=FailureType.CONTEXT_IGNORED,
                    confidence=round(verdict.confidence * 0.9, 4),
                    evidence=[
                        self._evidence(
                            verdict,
                            f"the answer was in the context{where} but not used: {verdict.reason}",
                            passage=passage,
                        )
                    ],
                    prescription=Prescription(
                        kind=PrescriptionKind.PROMPT_CHANGE,
                        description="Put the passage with the answer first, send fewer "
                        "passages, or ask the model to quote the passage before answering",
                    ),
                )
            ]
        return []
