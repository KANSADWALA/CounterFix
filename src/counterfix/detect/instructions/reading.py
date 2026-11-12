"""Reading an answer: sentences, words, list items, paragraphs, the bare answer text."""

from __future__ import annotations

import re

# --- reading the answer -------------------------------------------------------------------

_WORD_NUMBERS = {
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
_NUM = r"(\d{1,4}|" + "|".join(_WORD_NUMBERS) + r")"
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_ITEM = re.compile(r"^(?P<indent>[ \t]*)(?:[-*•+▪◦]|\d{1,3}[.)]|\(\d{1,3}\))\s+(?P<text>\S.*)$")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*]+\*\*:?\s*$|__[^_]+__:?\s*$|[-=*_]{3,}\s*$)")
_MARKUP = re.compile(r"\*\*|__|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)|`")
_SENTENCE_END = ".!?।॥؟。！？"
_SPLIT = re.compile(rf"(?<=[{_SENTENCE_END}])[\"')\]]*\s+(?=[^\sa-z0-9])")
_TRAILING_JUNK = re.compile(r"[\s\"')\]*_`]+$")


def _number(text: str) -> int:
    return int(text) if text.isdigit() else _WORD_NUMBERS[text.lower()]


def _plain(text: str) -> str:
    """Remove markdown emphasis and inline code markers."""
    return _MARKUP.sub("", text)


def _lines(answer: str) -> list[str]:
    """Non-empty content lines, without code blocks, headings and rules."""
    answer = _FENCE.sub("", answer)
    return [ln.rstrip() for ln in answer.splitlines() if ln.strip() and not _HEADING.match(ln)]


def list_items(answer: str) -> list[str]:
    """Text of the top-level list items (bulleted or numbered)."""
    found = [
        (len(m.group("indent").expandtabs(4)), m.group("text"))
        for ln in _lines(answer)
        if (m := _ITEM.match(ln))
    ]
    if not found:
        return []
    top = min(indent for indent, _ in found)
    return [_plain(text).strip() for indent, text in found if indent == top]


def is_numbered(answer: str) -> bool:
    items = [ln for ln in _lines(answer) if _ITEM.match(ln)]
    return bool(items) and all(re.match(r"^\s*(?:\d{1,3}[.)]|\(\d{1,3}\))\s", ln) for ln in items)


def sentences(answer: str) -> list[str]:
    """Sentences, counting a line without end punctuation as one (titles, list items).
    A line that continues the previous one in lowercase is joined to it."""
    blocks: list[str] = []
    for line in _lines(answer):
        match = _ITEM.match(line)
        text = _plain(match.group("text") if match else line).strip()
        if (
            blocks
            and not match
            and text[:1].islower()
            and blocks[-1][-1:] not in _SENTENCE_END + ":"
        ):
            blocks[-1] += " " + text
        else:
            blocks.append(text)
    out: list[str] = []
    for block in blocks:
        out += [p for p in _SPLIT.split(block) if any(ch.isalnum() for ch in p)]
    return out


def words(text: str) -> list[str]:
    """Whitespace-separated tokens that contain a letter or digit (works for any script)."""
    return [t for t in _plain(_FENCE.sub("", text)).split() if any(ch.isalnum() for ch in t)]


def _core(answer: str) -> str:
    """The answer without markup and trailing quotes or brackets."""
    return _TRAILING_JUNK.sub("", _plain(_FENCE.sub("", answer)).strip())


# --- constraints --------------------------------------------------------------------------
