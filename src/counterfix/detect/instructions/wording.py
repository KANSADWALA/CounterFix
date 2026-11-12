"""Banned words, how items start, ending with a question, letter case."""

from __future__ import annotations

import re
import unicodedata

from counterfix.detect.instructions.constraint import (
    Constraint,
)
from counterfix.detect.instructions.reading import (
    _FENCE,
    _core,
    _lines,
    _plain,
    list_items,
    sentences,
)

_QUOTED = re.compile(r"'([^'\n]{1,40})'|\"([^\"\n]{1,40})\"")
_FORBID_HEAD = re.compile(
    r"\b(?:do not|don't|never|avoid|without)\s+(?:use|using|say|saying|write|writing|include|"
    r"including|mention|mentioning)?\s*(?:the\s+|any\s+of\s+the\s+)?"
    r"(?:words?|terms?|phrases?)?\s*(?=['\"])",
    re.IGNORECASE,
)


def _forbidden_words(prompt: str) -> list[Constraint]:
    out = []
    for head in _FORBID_HEAD.finditer(prompt):
        clause = re.split(r"(?<=['\"])\s*[.;!?\n]|(?<=[a-z])[.;!?\n]", prompt[head.end() :])[0]
        for q in _QUOTED.finditer(clause):
            banned = (q.group(1) or q.group(2)).strip()
            if not banned:
                continue
            pattern = re.compile(
                rf"(?<!\w){re.escape(banned)}(?:s|es|ed|ing|y)?(?!\w)", re.IGNORECASE
            )

            def test(answer: str, pattern: re.Pattern[str] = pattern) -> str | None:
                found = pattern.search(_plain(answer))
                return f"uses {found.group(0)!r}" if found else None

            out.append(
                Constraint(
                    "forbidden_word", f"does not use '{banned}'", head.group(0) + banned, test
                )
            )
    return out


def _start_rules(prompt: str) -> list[Constraint]:
    out = []
    for m in re.finditer(
        r"\b(?:do not|don't|never)\s+(?:start|begin)\s+(?:any|each|every|a|the)?\s*"
        r"(?P<unit>[a-z]+?)s?\s+with\s+(?:the\s+word\s+|a\s+)?['\"](?P<w>[^'\"]{1,30})['\"]",
        prompt,
        re.IGNORECASE,
    ):
        unit, banned = m.group("unit").lower(), m.group("w").strip()
        starts = re.compile(rf"^\W*{re.escape(banned)}(?!\w)", re.IGNORECASE)

        def test(answer: str, unit: str = unit, starts: re.Pattern[str] = starts) -> str | None:
            parts = (
                sentences(answer) if unit == "sentence" else (list_items(answer) or _lines(answer))
            )
            bad = [p for p in parts if starts.match(p)]
            return f"{len(bad)} {unit}(s) start with it, e.g. {bad[0][:50]!r}" if bad else None

        out.append(Constraint("not_start", f"no {unit} starts with '{banned}'", m.group(0), test))
    for m in re.finditer(
        r"\b(?:start|begin)\s+(?:your\s+|the\s+)?(?:answer|reply|response)\s+with\s+"
        r"(?:the\s+(?:word|phrase)\s+)?['\"](?P<w>[^'\"]{1,60})['\"]",
        prompt,
        re.IGNORECASE,
    ):
        want = m.group("w").strip()

        def starts_with(answer: str, want: str = want) -> str | None:
            return None if _core(answer).lower().startswith(want.lower()) else "starts differently"

        out.append(Constraint("starts_with", f"starts with '{want}'", m.group(0), starts_with))
    for m in re.finditer(
        r"\bend\s+(?:your\s+|the\s+)?(?:answer|reply|response)\s+with\s+"
        r"(?:the\s+(?:exact\s+)?(?:word|phrase)\s+)?['\"](?P<w>[^'\"]{1,60})['\"]",
        prompt,
        re.IGNORECASE,
    ):
        want = m.group("w").strip()

        def ends_with(answer: str, want: str = want) -> str | None:
            core = _core(answer).lower()
            ok = core.endswith(want.lower()) or core.rstrip(".!").endswith(
                want.lower().rstrip(".!")
            )
            return None if ok else "ends differently"

        out.append(Constraint("ends_with", f"ends with '{want}'", m.group(0), ends_with))
    return out


def _question_ending(prompt: str) -> list[Constraint]:
    m = re.search(
        r"\b(?:end|finish|conclude|close)\s+(?:your\s+\w+\s+|it\s+|the\s+\w+\s+)?with\s+"
        r"(?:a\s+|one\s+)?(?:short\s+|follow-up\s+)?question\b",
        prompt,
        re.IGNORECASE,
    )
    if not m:
        return []

    def test(answer: str) -> str | None:
        core = _core(answer)
        while core and unicodedata.category(core[-1]) in ("So", "Sk", "Cf"):
            core = core[:-1].rstrip()  # trailing emoji
        return None if core[-1:] in ("?", "？", "؟") else "does not end with a question"

    return [Constraint("ends_with_question", "ends with a question", m.group(0), test)]


def _letter_case(prompt: str) -> list[Constraint]:
    out = []
    if re.search(
        r"\b(?:only|all|entirely|completely|in|use|using|write|written)\s+(?:in\s+)?(?:all\s+)?"
        r"(?:only\s+)?lower[\s-]?case\b|\blower[\s-]?case\s+(?:letters\s+)?only\b|"
        r"\bno\s+capital\s+letters\b",
        prompt,
        re.IGNORECASE,
    ):

        def lower(answer: str) -> str | None:
            text = _FENCE.sub("", answer)
            bad = [w for w in text.split() if any(ch.isupper() for ch in w)]
            return f"has capitals, e.g. {bad[0]!r}" if bad else None

        out.append(Constraint("lowercase", "all lowercase", "lowercase", lower))
    if re.search(
        r"\b(?:all|entirely|only|in)\s+(?:capital letters|caps|upper[\s-]?case)\b|\ball caps\b",
        prompt,
        re.IGNORECASE,
    ):

        def upper(answer: str) -> str | None:
            text = _FENCE.sub("", answer)
            bad = [w for w in text.split() if any(ch.islower() for ch in w)]
            return f"has lowercase letters, e.g. {bad[0]!r}" if bad else None

        out.append(Constraint("uppercase", "all capital letters", "uppercase", upper))
    return out
