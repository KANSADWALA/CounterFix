"""Pulling the answer out of a free-text reply: a choice letter, a final number, code.

Benchmarks ask chat models questions and need the one thing to score: "B" from "The
answer is (B) because...", 18 from "...so she earns $18.", the function from a reply
with prose around a code block.
"""

from __future__ import annotations

import math
import re
from typing import Any

from counterfix.code.check import extract_code as extract_code
from counterfix.core.record import CallRecord
from counterfix.metrics.base import Metric, references

_LETTERS = "ABCDEFGHIJKLMNOP"
_ANSWER_IS = re.compile(
    r"(?i:answer|option|choice)\s*(?i:is|:)?\s*(?i:option\s*)?[\(\[\*]*\s*([A-P])\b(?![a-z'])"
)
_LINE_START = re.compile(r"^\s*[\(\[\*]*\s*([A-P])\s*[\)\]\.:\*]", re.MULTILINE)
_BARE = re.compile(r"^\s*[\(\[\*]*\s*([A-P])\s*[\)\]\.\*]*\s*$")
_PAREN = re.compile(r"\(([A-P])\)")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_IS_LETTER = re.compile(
    r"\b(?i:is|it's|choose|pick|select|go with)\s+[\(\[\*]*([A-P])\b[\)\]\*]*(?![a-z'])"
)


def extract_choice(text: str, letters: str = "ABCD") -> str | None:
    """The chosen option letter, or None when the reply does not commit to one.

    Explicit statements win ("Answer: B", "the answer is (C)", \\boxed{D}, "it is B");
    the last one counts, so a corrected answer is read as corrected. A letter at the start
    of a line, or in parentheses, counts only when the reply has exactly one such letter:
    a reply that lists the options back ("A. Paris  B. Rome ...") is not read as a choice.
    """
    allowed = set(letters.upper())

    def found(pattern: re.Pattern[str]) -> list[str]:
        return [
            str(x).strip().upper()
            for x in pattern.findall(text)
            if str(x).strip().upper() in allowed
        ]

    if (m := _BARE.match(text)) and m.group(1).upper() in allowed:
        return str(m.group(1)).upper()
    for pattern in (_ANSWER_IS, _BOXED, _IS_LETTER):
        if hits := found(pattern):
            return hits[-1]
    for pattern in (_LINE_START, _PAREN):
        if len(set(hits := found(pattern))) == 1:
            return hits[0]
    return None


# thousands separators only in groups of three (1,234,567), so a list "3,4" is two numbers
_NUMBER = re.compile(r"-?\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)|-?\.\d+")


def _to_float(token: str) -> float | None:
    cleaned = token.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """The final numeric answer: after GSM8K's '####', in \\boxed{}, after 'answer is',
    else the last number in the reply."""
    for marker in ("####",):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
            break
    else:
        if box := _BOXED.findall(text):
            text = box[-1]
        else:
            lowered = text.lower()
            at = lowered.rfind("answer is")
            if at != -1 and _NUMBER.search(text[at:]):
                text = text[at:]
                first = _NUMBER.search(text)
                return _to_float(first.group(0)) if first else None
    numbers = _NUMBER.findall(text)
    return _to_float(numbers[-1]) if numbers else None


class ChoiceMatch(Metric):
    """1.0 when the chosen letter equals the reference letter (multiple choice)."""

    name = "choice"
    default_threshold = 1.0

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        letters = str(self.options.get("letters", "ABCD"))
        got = extract_choice(record.answer_text, letters)
        refs = reference if isinstance(reference, (list, tuple)) else [reference]
        # an int is an option index (MMLU stores 0-3), anything else is the letter
        want = {
            _LETTERS[r] if isinstance(r, int) and 0 <= r < len(_LETTERS) else str(r).strip().upper()
            for r in refs
        }
        if got is None:
            return 0.0, "no option letter found in the answer"
        return float(got in want), f"chose {got}, expected {'/'.join(sorted(want))}"


class NumberMatch(Metric):
    """1.0 when the final number equals the reference number (GSM8K style)."""

    name = "number"
    default_threshold = 1.0

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        got = extract_number(record.answer_text)
        wants = [extract_number(r) for r in references(reference)]
        wants_known = [w for w in wants if w is not None]
        if not wants_known:
            return None, "the reference holds no number"
        if got is None:
            return 0.0, "no number found in the answer"
        tol = float(self.options.get("tolerance", 1e-6))
        hit = any(math.isclose(got, w, rel_tol=tol, abs_tol=1e-9) for w in wants_known)
        return float(hit), f"answered {got:g}, expected {wants_known[0]:g}"
