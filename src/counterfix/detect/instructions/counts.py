"""Sentence, word, item and paragraph counts, and numbered lists."""

from __future__ import annotations

import re

from counterfix.detect.instructions.constraint import (
    _CMP,
    _COMPARE,
    _RELATION,
    Constraint,
    _counted,
    _per_item,
)
from counterfix.detect.instructions.reading import (
    _FENCE,
    _HEADING,
    _NUM,
    _number,
    is_numbered,
    list_items,
    sentences,
    words,
)

_SENTENCE_COUNT = re.compile(
    rf"\b{_CMP}(?P<n>{_NUM[1:-1]})(?:\s*(?:-|to|–)\s*(?P<n2>{_NUM[1:-1]}))?\s+"
    r"(?:short\s+|full\s+|complete\s+|concise\s+)?sentences?\b",
    re.IGNORECASE,
)
_WORD_COUNT = re.compile(
    rf"\b{_CMP}(?P<n>{_NUM[1:-1]})(?:\s*(?:-|to|–)\s*(?P<n2>{_NUM[1:-1]}))?\s+words?\b"
    r"(?P<tail>\s+or\s+(?:less|fewer)|\s+max(?:imum)?)?",
    re.IGNORECASE,
)
_ITEM_COUNT = re.compile(
    rf"\b{_CMP}(?P<n>{_NUM[1:-1]})\s+(?:(?!bullet)[a-z-]+\s+)?"
    r"(?P<noun>bullet points|bullets|bullet|points|tips|items|steps|reasons|ideas|examples|"
    r"ways|benefits|facts|suggestions)\b",
    re.IGNORECASE,
)


def _sentence_counts(prompt: str) -> list[Constraint]:
    out = []
    for m in _SENTENCE_COUNT.finditer(prompt):
        made = _counted("sentence_count", "sentence", sentences, prompt, m)
        if made:
            out.append(made)
    return out


def _word_counts(prompt: str) -> list[Constraint]:
    out = []
    for m in _WORD_COUNT.finditer(prompt):
        if m.group("cmp") is None and m.group("tail") is None:
            continue  # "a 300 words essay" is a target, not a limit
        if m.group("cmp") is None or (m.group("cmp") or "").lower() == "in":
            if not m.group("tail"):
                continue  # "in 50 words" is a target too
            # "50 words or fewer", "in 50 words max": an upper limit
            m2 = re.match(r"", "")  # placeholder to keep types simple
            del m2
            n = _number(m.group("n"))

            def limit(answer: str, n: int = n) -> str | None:
                count = len(words(answer))
                return None if count <= n else f"has {count}"

            out.append(Constraint("word_count", f"at most {n} words", m.group(0).strip(), limit))
            continue
        made = _counted("word_count", "word", words, prompt, m)
        if made:
            out.append(made)
    return out


def _item_counts(prompt: str) -> list[Constraint]:
    out = []
    for m in _ITEM_COUNT.finditer(prompt):
        if _per_item(prompt, m.start()):
            continue
        noun = m.group("noun").lower()
        bullets = noun.startswith("bullet")
        op, word = _RELATION.get((m.group("cmp") or "").lower() or None, ("==", "exactly"))
        n = _number(m.group("n"))

        def test(answer: str, n: int = n, op: str = op, bullets: bool = bullets) -> str | None:
            items = list_items(answer)
            if not items:
                return "has no bullet points" if bullets else None  # prose is fine for "3 ideas"
            return None if _COMPARE[op](len(items), n) else f"has {len(items)} list items"

        out.append(Constraint("item_count", f"{word} {n} {noun}", m.group(0).strip(), test))
    return out


_PARAGRAPH_COUNT = re.compile(
    rf"\b{_CMP}(?P<n>{_NUM[1:-1]})\s+(?:short\s+)?paragraphs?\b", re.IGNORECASE
)


def paragraphs(answer: str) -> list[str]:
    """Blocks of text separated by blank lines (headings and rules not counted)."""
    blocks = re.split(r"\n\s*\n", _FENCE.sub("", answer).strip())
    return [b for b in blocks if b.strip() and not all(_HEADING.match(ln) for ln in b.splitlines())]


def _paragraph_counts(prompt: str) -> list[Constraint]:
    out = []
    for m in _PARAGRAPH_COUNT.finditer(prompt):
        if _per_item(prompt, m.start()):
            continue
        op, word = _RELATION.get((m.group("cmp") or "").lower() or None, ("==", "exactly"))
        n = _number(m.group("n"))

        def test(answer: str, n: int = n, op: str = op) -> str | None:
            found = len(paragraphs(answer))
            return None if _COMPARE[op](found, n) else f"has {found}"

        out.append(
            Constraint("paragraph_count", f"{word} {n} paragraph{'s' * (n != 1)}", m.group(0), test)
        )
    return out


def _numbered(prompt: str) -> list[Constraint]:
    m = re.search(r"\bnumber them\b|\bnumbered\s+(?:list|items|points|tips|steps)\b", prompt, re.I)
    if not m:
        return []

    def test(answer: str) -> str | None:
        return None if is_numbered(answer) else "the items are not numbered"

    return [Constraint("numbered", "numbered list", m.group(0), test)]
