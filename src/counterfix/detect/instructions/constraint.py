"""A checkable instruction (:class:`Constraint`) and the helpers that parse counts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from counterfix.detect.instructions.reading import (
    _lines,
    _number,
    list_items,
    sentences,
    words,
)


@dataclass(frozen=True)
class Constraint:
    """One instruction code can check. ``violation`` returns why the answer breaks it."""

    kind: str
    description: str
    quote: str
    test: Callable[[str], str | None] = field(compare=False, repr=False)

    def violation(self, answer: str) -> str | None:
        return self.test(answer)


_CMP = (
    r"(?:(?P<cmp>exactly|precisely|only|at most|no more than|not more than|up to|"
    r"a maximum of|maximum of|max\.?|at least|no fewer than|no less than|a minimum of|"
    r"minimum of|under|fewer than|less than|more than|over|in)\s+)?"
)
_RELATION = {
    None: ("==", "exactly"),
    "exactly": ("==", "exactly"),
    "precisely": ("==", "exactly"),
    "only": ("==", "exactly"),
    "in": ("==", "exactly"),
    "at most": ("<=", "at most"),
    "no more than": ("<=", "at most"),
    "not more than": ("<=", "at most"),
    "up to": ("<=", "at most"),
    "a maximum of": ("<=", "at most"),
    "maximum of": ("<=", "at most"),
    "max": ("<=", "at most"),
    "max.": ("<=", "at most"),
    "at least": (">=", "at least"),
    "no fewer than": (">=", "at least"),
    "no less than": (">=", "at least"),
    "a minimum of": (">=", "at least"),
    "minimum of": (">=", "at least"),
    "under": ("<", "fewer than"),
    "fewer than": ("<", "fewer than"),
    "less than": ("<", "fewer than"),
    "more than": (">", "more than"),
    "over": (">", "more than"),
}
_COMPARE: dict[str, Callable[[int, int], bool]] = {
    "==": lambda a, b: a == b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}


def _per_item(prompt: str, start: int) -> str | None:
    """'each tip ... under 12 words' -> 'tip'; None when the limit is for the whole answer."""
    before = prompt[max(0, start - 60) : start]
    clause = re.split(r"[.;!?\n]", before)[-1]
    match = re.search(r"\b(?:each|every|per)\s+(\w+)", clause, re.IGNORECASE)
    return match.group(1).lower().rstrip("s") if match else None


def _counted(
    kind: str, unit: str, count: Callable[[str], list[str]], prompt: str, m: re.Match[str]
) -> Constraint | None:
    op, word = _RELATION.get((m.group("cmp") or "").lower() or None, ("==", "exactly"))
    low, high = _number(m.group("n")), m.group("n2")
    target = f"{_number(m.group('n'))}"
    if high is not None:  # "2-3 sentences"
        lo, hi = low, _number(high)
        target = f"{lo} to {hi}"

        def in_range(answer: str) -> str | None:
            n = len(count(answer))
            return None if lo <= n <= hi else f"has {n}"

        return Constraint(kind, f"{target} {unit}s", m.group(0).strip(), in_range)
    per = _per_item(prompt, m.start())
    description = f"{word} {target} {unit}{'' if low == 1 and op == '==' else 's'}"
    if per is None:

        def whole(answer: str) -> str | None:
            n = len(count(answer))
            return None if _COMPARE[op](n, low) else f"has {n}"

        return Constraint(kind, description, m.group(0).strip(), whole)
    if unit != "word":
        return None  # "each paragraph in 2 sentences": too loose to check reliably

    def each(answer: str) -> str | None:
        parts = sentences(answer) if per == "sentence" else (list_items(answer) or _lines(answer))
        bad = [p for p in parts if not _COMPARE[op](len(words(p)), low)]
        if not bad:
            return None
        return f"{len(bad)} {per}(s) break it, e.g. {bad[0][:60]!r} ({len(words(bad[0]))} words)"

    return Constraint(f"{kind}_each", f"each {per}: {description}", m.group(0).strip(), each)
