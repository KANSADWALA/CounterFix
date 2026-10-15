"""F1: the answer was cut off by the output-token limit."""

from __future__ import annotations

from counterfix.core.config import Config
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detectors.basic import LENGTH_REASONS

_TERMINAL = (*".!?)\"'`]}>*_", "```")


def _raised_limit(limit: int | None, factor: int) -> int:
    return min((limit or 256) * factor, 32768)


class TruncationDetector:
    """Flags F1 from the stop reason, or infers it when the provider gives none.

    * ``finish_reason`` of ``length`` (or ``max_tokens``): confident F1.
    * Reasoning models that spent the whole budget thinking get their own cause and fix.
    * No stop reason at all (some local servers): infer F1 when the output token count
      reaches the limit, or when a long answer stops mid-sentence. Marked ``inferred``.
    """

    name = "truncation"
    covers = (FailureType.TRUNCATION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        reason = (record.finish_reason or "").lower()
        limit = record.max_tokens
        if reason in LENGTH_REASONS:
            return [self._finding(record, limit, confidence=0.95, inferred=False)]
        if reason:
            return []  # a known, different stop reason (stop, tool_calls, ...)
        return self._infer(record, limit)

    def _infer(self, record: CallRecord, limit: int | None) -> list[Finding]:
        if limit is None:
            return []
        used = record.usage.completion_tokens if record.usage is not None else None
        if used is not None and used >= int(limit * 0.98):
            return [self._finding(record, limit, confidence=0.8, inferred=True)]
        answer = record.answer_text
        approx_tokens = len(answer) / 4
        if answer and approx_tokens >= limit * 0.8 and not answer.rstrip().endswith(_TERMINAL):
            return [self._finding(record, limit, confidence=0.6, inferred=True)]
        return []

    def _finding(
        self, record: CallRecord, limit: int | None, *, confidence: float, inferred: bool
    ) -> Finding:
        if record.has_reasoning and not record.answer_text:
            detail = f"the model spent the whole max_tokens budget ({limit}) on reasoning"
            fix = Prescription(
                kind=PrescriptionKind.PARAM_CHANGE,
                description=(
                    "Raise max_tokens (reasoning counts against it), or lower reasoning "
                    "effort if the provider supports it"
                ),
                patch={"max_tokens": _raised_limit(limit, 8)},
            )
        else:
            detail = f"the answer stopped at the max_tokens limit ({limit})"
            fix = Prescription(
                kind=PrescriptionKind.PARAM_CHANGE,
                description="Raise max_tokens, or ask for a shorter answer",
                patch={"max_tokens": _raised_limit(limit, 4)},
            )
        if inferred:
            detail += "; inferred, the provider gave no stop reason"
        return Finding(
            failure_type=FailureType.TRUNCATION,
            confidence=confidence,
            inferred=inferred,
            evidence=[Evidence(detector=self.name, detail=detail)],
            prescription=fix,
        )
