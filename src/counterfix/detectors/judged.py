"""Diagnose-mode checks that ask a judge. Each one costs judge calls, so they are not part
of ``default_detectors()`` and never run in live mode. Use ``judge_detectors(judge)``."""

from __future__ import annotations

import re
from typing import Any

from counterfix.core.config import Config
from counterfix.core.detector import Detector, SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detectors.refusal import _REFUSAL
from counterfix.judges import tasks
from counterfix.judges.base import Judge, JudgeTask, Verdict
from counterfix.safe_eval import UnsafeExpression, safe_eval

_CONSTRAINT = re.compile(
    r"\b(must|never|always|do not|don't|only|exactly|at most|at least|no more than|under|"
    r"without|avoid|end with|start with|in \w+ (?:words|sentences)|lowercase|uppercase|"
    r"bullet|numbered|reply in|respond in|write in|\d+ (?:words|sentences|bullet|tips|items))\b",
    re.IGNORECASE,
)


class _JudgeCheck:
    """Shared plumbing: ask the judge, skip on judge failure, turn a verdict into evidence."""

    name = "judge"
    covers: tuple[FailureType, ...] = ()

    def __init__(self, judge: Judge):
        self.judge = judge

    def _ask(self, task: JudgeTask) -> Verdict:
        verdict = self.judge.ask(task)
        if verdict.failed:
            raise SkipCheck(f"judge failed: {verdict.reason}")
        return verdict

    def _evidence(self, verdict: Verdict, detail: str, **data: Any) -> Evidence:
        votes = [v.model_dump() for v in verdict.votes]
        return Evidence(
            detector=self.name,
            detail=detail,
            data={
                "label": verdict.label,
                "reason": verdict.reason,
                "judge": verdict.judge,
                "votes": votes,
                **data,
            },
        )


class HallucinationCheck(_JudgeCheck):
    """F4: the answer states false or unsupported specifics."""

    name = "judge_hallucination"
    covers = (FailureType.HALLUCINATION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if not record.answer_text:
            raise SkipCheck("no answer to check")
        verdict = self._ask(tasks.hallucination(record))
        if verdict.label not in ("contains_false", "contains_unsupported"):
            return []
        claims = verdict.extra.get("claims") or []
        claims_text = (
            "; ".join(str(c) for c in claims[:3]) if isinstance(claims, list) else str(claims)
        )
        grounded = tasks.CONTEXT_KEY in record.metadata
        weight = 1.0 if verdict.label == "contains_false" else 0.85
        return [
            Finding(
                failure_type=FailureType.HALLUCINATION,
                confidence=round(verdict.confidence * weight, 4),
                evidence=[
                    self._evidence(
                        verdict,
                        f"{verdict.label.replace('_', ' ')}: {claims_text or verdict.reason}",
                        claims=claims,
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE
                    if grounded
                    else PrescriptionKind.ADD_CONTEXT,
                    description=(
                        "Tell the model to answer only from the context and to say when the "
                        "context does not contain the answer"
                        if grounded
                        else "Give the model the source (retrieval), or allow 'I don't know'"
                    ),
                ),
            )
        ]


class FalsePremiseCheck(_JudgeCheck):
    """F5: the request assumes something false and the answer goes along with it."""

    name = "judge_false_premise"
    covers = (FailureType.FALSE_PREMISE,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if not record.prompt_text or not record.answer_text:
            raise SkipCheck("needs a request and an answer")
        verdict = self._ask(tasks.false_premise(record))
        if verdict.label != "premise_accepted":
            return []
        premise = str(verdict.extra.get("premise") or "").strip()
        return [
            Finding(
                failure_type=FailureType.FALSE_PREMISE,
                confidence=verdict.confidence,
                evidence=[
                    self._evidence(
                        verdict,
                        f"accepted a false premise: {premise or verdict.reason}",
                        premise=premise,
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=(
                        "Add to the system prompt: 'Check the question's assumptions first; if "
                        "one is false, say so and correct it before answering.'"
                    ),
                ),
            )
        ]


class DroppedInstructionCheck(_JudgeCheck):
    """F6: the answer breaks an explicit instruction. Skipped when the prompt has none."""

    name = "judge_dropped_instruction"
    covers = (FailureType.DROPPED_INSTRUCTION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if not _CONSTRAINT.search(f"{record.system_text}\n{record.prompt_text}"):
            raise SkipCheck("no explicit constraints in the prompt")
        if not record.answer_text:
            raise SkipCheck("no answer to check")
        verdict = self._ask(tasks.dropped_instruction(record))
        if verdict.label != "some_unmet":
            return []
        unmet = verdict.extra.get("unmet") or []
        listed = "; ".join(str(u) for u in unmet[:4]) if isinstance(unmet, list) else str(unmet)
        return [
            Finding(
                failure_type=FailureType.DROPPED_INSTRUCTION,
                confidence=verdict.confidence,
                evidence=[
                    self._evidence(
                        verdict, f"instructions not met: {listed or verdict.reason}", unmet=unmet
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=(
                        "Move the dropped constraints to the end of the prompt as a short "
                        "checklist, or split the task into steps"
                    ),
                ),
            )
        ]


class AmbiguityCheck(_JudgeCheck):
    """F9: the request had several readings and the answer silently picked one."""

    name = "judge_ambiguity"
    covers = (FailureType.AMBIGUITY,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if not record.prompt_text or not record.answer_text:
            raise SkipCheck("needs a request and an answer")
        verdict = self._ask(tasks.ambiguity(record))
        if verdict.label != "ambiguous_silent":
            return []
        readings = verdict.extra.get("readings") or []
        listed = (
            " | ".join(str(r) for r in readings[:4])
            if isinstance(readings, list)
            else str(readings)
        )
        return [
            Finding(
                failure_type=FailureType.AMBIGUITY,
                confidence=verdict.confidence,
                evidence=[
                    self._evidence(
                        verdict,
                        f"several readings, one picked silently: {listed}",
                        readings=readings,
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.ASK_USER,
                    description=(
                        "Add the missing detail to the request, or let the model ask a "
                        "clarifying question when a request has several readings"
                    ),
                ),
            )
        ]


class OverRefusalCheck(_JudgeCheck):
    """F3: confirms whether a refusal was unwarranted. Only asks the judge when the answer
    contains a refusal phrase, so normal answers cost nothing."""

    name = "judge_over_refusal"
    covers = (FailureType.OVER_REFUSAL,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        answer = record.answer_text.replace(chr(0x2019), "'")
        if not _REFUSAL.search(answer[:400]):
            raise SkipCheck("no refusal phrase in the answer")
        verdict = self._ask(tasks.over_refusal(record))
        if verdict.label != "over_refusal":
            return []
        return [
            Finding(
                failure_type=FailureType.OVER_REFUSAL,
                confidence=verdict.confidence,
                evidence=[self._evidence(verdict, f"refused a harmless request: {verdict.reason}")],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description="State the purpose and audience of the request in the prompt",
                ),
            )
        ]


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORDS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}


_SEPARATORS = re.compile(r"(?<=\d)(?:\{,\}|\\,|[\u2009\u202f\u00a0_'])(?=\d)")


def normalise_numbers(text: str) -> str:
    """Remove digit-group separators models use: LaTeX {,} and \\, thin and narrow
    no-break spaces, underscores and apostrophes (1{,}419{,}857 -> 1419857)."""
    return _SEPARATORS.sub("", text)


def answer_contains(answer: str, expected: Any) -> bool:
    """Does the answer state the expected value? Numbers match ignoring group separators
    (commas, LaTeX, thin spaces); small numbers also match as words; strings match
    case-insensitively."""
    answer = normalise_numbers(answer)
    if isinstance(expected, bool):
        expected = str(expected)
    if isinstance(expected, int | float):
        found = []
        for raw in _NUMBER.findall(answer):
            try:
                found.append(float(raw.replace(",", "")))
            except ValueError:
                continue
        lowered = answer.lower()
        found += [float(v) for w, v in _WORDS.items() if re.search(rf"\b{w}\b", lowered)]
        return any(abs(f - float(expected)) <= 1e-9 * max(1.0, abs(float(expected))) for f in found)
    return str(expected).strip().lower() in answer.lower()


_COMPUTE_HINT = re.compile(
    r"\d|\b(?:how many|count|sum|total|multipl\w*|divid\w*|power|square|percent\w*|"
    r"calculat\w*|average|reverse\w*|backwards|letters?|digits?|characters?)\b",
    re.IGNORECASE,
)


class ComputationCheck(_JudgeCheck):
    """F8: the judge writes an expression for the exact answer; counterfix evaluates it
    safely and checks the model's answer. The judge never computes the result itself."""

    name = "computation"
    covers = (FailureType.COMPUTATION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if not record.answer_text:
            raise SkipCheck("no answer to check")
        if not _COMPUTE_HINT.search(record.prompt_text):
            raise SkipCheck("no computation in the request")
        verdict = self._ask(tasks.computation(record))
        expression = str(verdict.extra.get("expression") or "").strip()
        if verdict.label != "computable" or not expression:
            return []
        try:
            expected = safe_eval(expression)
        except (UnsafeExpression, ValueError, TypeError, ZeroDivisionError, IndexError) as exc:
            raise SkipCheck(f"could not evaluate the judge's expression: {exc}") from exc
        if isinstance(expected, list | tuple):
            raise SkipCheck("expression did not produce a single value")
        if answer_contains(record.answer_text, expected):
            return []
        return [
            Finding(
                failure_type=FailureType.COMPUTATION,
                confidence=0.9,
                evidence=[
                    self._evidence(
                        verdict,
                        f"the exact answer is {expected!r} (computed from `{expression}`), "
                        "but the model's answer does not contain it",
                        expression=expression,
                        expected=expected,
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.USE_TOOL,
                    description="Give the model a code or calculator tool for exact computation",
                ),
            )
        ]


def judge_detectors(judge: Judge) -> list[Detector]:
    """All judge-based checks, sharing one judge (a single LLM or a panel)."""
    return [
        OverRefusalCheck(judge),
        HallucinationCheck(judge),
        FalsePremiseCheck(judge),
        DroppedInstructionCheck(judge),
        ComputationCheck(judge),
        AmbiguityCheck(judge),
    ]
