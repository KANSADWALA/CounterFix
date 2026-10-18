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
from counterfix.detectors.basic import LENGTH_REASONS
from counterfix.detectors.instructions import extract_constraints, mentions
from counterfix.detectors.refusal import _REFUSAL
from counterfix.judges import tasks
from counterfix.judges.base import Judge, JudgeTask, Verdict
from counterfix.safe_eval import UnsafeExpression, safe_eval
from counterfix.text import normalise_numbers

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


def _unmet_items(value: Any) -> list[str]:
    """The judge's unmet list as text, tolerant of strings or {instruction, evidence}."""
    if not isinstance(value, list):
        return [str(value)] if value else []
    items = []
    for entry in value:
        if isinstance(entry, dict):
            instruction = str(entry.get("instruction") or "").strip()
            evidence = str(entry.get("evidence") or "").strip()
            if instruction:
                items.append(f"{instruction} (answer: {evidence})" if evidence else instruction)
        elif str(entry).strip():
            items.append(str(entry).strip())
    return items


class DroppedInstructionCheck(_JudgeCheck):
    """F6, the part code cannot verify: content instructions such as 'mention X' or 'never
    mention competitors'. Countable instructions are left to ``InstructionCheck`` and any
    judge complaint about them is discarded, because code is exact and judges miscount."""

    name = "judge_dropped_instruction"
    covers = (FailureType.DROPPED_INSTRUCTION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        request = f"{record.system_text}\n{record.prompt_text}"
        if not _CONSTRAINT.search(request):
            raise SkipCheck("no explicit constraints in the prompt")
        if (record.finish_reason or "").lower() in LENGTH_REASONS:
            raise SkipCheck("the answer was cut off (see F1)")
        if not record.answer_text:
            raise SkipCheck("no answer to check")
        in_code = extract_constraints(request)
        verdict = self._ask(tasks.dropped_instruction(record, [c.description for c in in_code]))
        if verdict.label != "some_unmet":
            return []
        unmet = [
            item
            for item in _unmet_items(verdict.extra.get("unmet"))
            if not any(mentions(c, item.split(" (answer:")[0]) for c in in_code)
        ]
        if not unmet:  # nothing quoted, or only things code already checked
            return []
        return [
            Finding(
                failure_type=FailureType.DROPPED_INSTRUCTION,
                confidence=verdict.confidence,
                evidence=[
                    self._evidence(
                        verdict, "instructions not met: " + "; ".join(unmet[:4]), unmet=unmet
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


def _yes_no(answer: str) -> bool | None:
    """True for an answer opening with yes/true, False for no/false, else None."""
    first = re.match(r"\W*(\w+)", answer)
    word = first.group(1).lower() if first else ""
    return True if word in ("yes", "true") else False if word in ("no", "false") else None


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
        if isinstance(expected, bool):
            # A yes/no fact ('is 0.1 + 0.2 == 0.3?'). Only checkable when the answer opens
            # with yes/no; "9.9 is larger" cannot be matched against True or False.
            said = _yes_no(record.answer_text)
            if said is None:
                raise SkipCheck("a yes/no result, but the answer does not start with yes or no")
            if said == expected:
                return []
        elif answer_contains(record.answer_text, expected):
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
