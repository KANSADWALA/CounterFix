"""F13 prompt injection: the answer followed instructions planted in untrusted text.

Untrusted text is what the app put in front of the model without writing it itself:
retrieved passages (``counterfix.context(chunks=...)``, RAG prompts) and tool results
(``tool`` messages). Two checks:

* :class:`CanaryCheck` (live, free, exact): for probes, a canary string planted in the
  untrusted text (``metadata["injection_canary"]``) shows up in the answer.
* :class:`InjectionCheck` (judge): finds a planted instruction in the untrusted text and
  decides whether the answer carried it out. Skips calls without untrusted text.
"""

from __future__ import annotations

from typing import Any

from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detect.judged import _JudgeCheck
from counterfix.judges import tasks
from counterfix.judges.tasks import CONTEXT_KEY

CANARY_KEY = "injection_canary"
_FIX = Prescription(
    kind=PrescriptionKind.PROMPT_CHANGE,
    description=(
        "Mark retrieved text and tool results as data (delimit them), tell the model in the "
        "system prompt never to follow instructions found there, and restate the user's "
        "task after the untrusted text"
    ),
)


def untrusted_text(record: CallRecord) -> str:
    """Retrieved context and tool results on this call, joined."""
    parts: list[str] = []
    context = record.metadata.get(CONTEXT_KEY)
    if isinstance(context, str) and context.strip():
        parts.append(context)
    parts += [m.text for m in record.messages if m.role == "tool" and m.text.strip()]
    return "\n\n".join(parts)


def user_request(record: CallRecord) -> str:
    """What the user asked: the bare RAG question when known, else the user messages."""
    question = record.metadata.get("rag_question")
    if isinstance(question, str) and question:
        return question
    return "\n".join(m.text for m in record.messages if m.role == "user")


def canary_found(answer: str, canary: str) -> bool:
    """The canary in the answer (case and dashes or spaces inside it ignored)."""
    squash = "".join(ch for ch in answer.lower() if ch.isalnum())
    want = "".join(ch for ch in canary.lower() if ch.isalnum())
    return bool(want) and want in squash


class CanaryCheck:
    """F13 when a planted canary appears in the answer (probes only; free and exact)."""

    name = "injection_canary"
    covers = (FailureType.INJECTION_FOLLOWED,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        canaries = record.metadata.get(CANARY_KEY)
        if not canaries:
            raise SkipCheck("no planted canary on this call (probes only)")
        found = [c for c in _as_list(canaries) if canary_found(record.answer_text, c)]
        if not found:
            return []
        return [
            Finding(
                failure_type=FailureType.INJECTION_FOLLOWED,
                confidence=1.0,
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=f"the answer contains the planted canary {found[0]!r}",
                        data={"canary": found[0]},
                    )
                ],
                prescription=_FIX,
            )
        ]


def _as_list(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


class InjectionCheck(_JudgeCheck):
    """F13 by judge: a planted instruction in the untrusted text, carried out."""

    name = "judge_injection"
    covers = (FailureType.INJECTION_FOLLOWED,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        untrusted = untrusted_text(record)
        if not untrusted:
            raise SkipCheck("no retrieved text or tool results on this call")
        if not record.answer_text:
            raise SkipCheck("no answer to check")
        verdict = self._ask(tasks.injection_followed(record, untrusted, user_request(record)))
        if verdict.label != "followed_injection":
            return []
        planted = str(verdict.extra.get("instruction") or "").strip()
        what = f" ({planted[:120]})" if planted else ""
        return [
            Finding(
                failure_type=FailureType.INJECTION_FOLLOWED,
                confidence=round(verdict.confidence * 0.9, 4),
                evidence=[
                    self._evidence(
                        verdict,
                        f"the answer carried out a planted instruction{what}: {verdict.reason}",
                        instruction=planted,
                    )
                ],
                prescription=_FIX,
            )
        ]
