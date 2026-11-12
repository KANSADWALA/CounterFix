"""Reply language (by script), British or American spelling, and a title in brackets."""

from __future__ import annotations

import re
import unicodedata

from counterfix.detect.instructions.constraint import (
    Constraint,
)
from counterfix.detect.instructions.reading import (
    _FENCE,
    _plain,
)

_SCRIPTS = {
    "hindi": ("DEVANAGARI",),
    "marathi": ("DEVANAGARI",),
    "nepali": ("DEVANAGARI",),
    "sanskrit": ("DEVANAGARI",),
    "bengali": ("BENGALI",),
    "bangla": ("BENGALI",),
    "gujarati": ("GUJARATI",),
    "punjabi": ("GURMUKHI",),
    "tamil": ("TAMIL",),
    "telugu": ("TELUGU",),
    "kannada": ("KANNADA",),
    "malayalam": ("MALAYALAM",),
    "odia": ("ORIYA",),
    "urdu": ("ARABIC",),
    "arabic": ("ARABIC",),
    "persian": ("ARABIC",),
    "farsi": ("ARABIC",),
    "russian": ("CYRILLIC",),
    "ukrainian": ("CYRILLIC",),
    "bulgarian": ("CYRILLIC",),
    "greek": ("GREEK",),
    "hebrew": ("HEBREW",),
    "thai": ("THAI",),
    "korean": ("HANGUL",),
    "chinese": ("CJK",),
    "mandarin": ("CJK",),
    "japanese": ("HIRAGANA", "KATAKANA", "CJK"),
    "english": ("LATIN",),
}
_LANGUAGE = re.compile(
    r"\b(?:reply|respond|answer|write|written|explain|speak|talk|output)\b[^.?!\n]{0,40}?"
    r"\bin\s+(?P<lang>"
    + "|".join(_SCRIPTS)
    + r")\b|\bin\s+(?P<lang2>"
    + "|".join(_SCRIPTS)
    + r")\s+only\b",
    re.IGNORECASE,
)


def script_share(text: str, scripts: tuple[str, ...]) -> float:
    """Share of letters (and their combining marks) written in one of ``scripts``."""
    letters = [ch for ch in text if unicodedata.category(ch)[0] in ("L", "M")]
    if not letters:
        return 0.0
    hits = sum(unicodedata.name(ch, "").startswith(scripts) for ch in letters)
    return hits / len(letters)


def _language(prompt: str) -> list[Constraint]:
    m = _LANGUAGE.search(prompt)
    if not m:
        return []
    language = (m.group("lang") or m.group("lang2")).lower()
    scripts = _SCRIPTS[language]

    def test(answer: str) -> str | None:
        share = script_share(_FENCE.sub("", answer), scripts)
        return None if share >= 0.5 else f"only {share:.0%} of the letters are in {language} script"

    return [Constraint("language", f"written in {language.title()}", m.group(0), test)]


_OR_WORDS = [
    "col",
    "flav",
    "fav",
    "behavi",
    "hon",
    "hum",
    "lab",
    "neighb",
    "harb",
    "rum",
    "vap",
    "sav",
    "arm",
    "endeav",
    "glam",
    "rig",
    "splend",
    "vig",
]
_AMERICAN = [
    *(
        rf"{stem}or(?:s|ed|ing|ful|fully|less|able|ably|ite|ites|y|hood|hoods)?"
        for stem in _OR_WORDS
    ),
    r"cent(?:er|ers|ered)",
    r"theat(?:er|ers)",
    r"fib(?:er|ers)",
    r"lit(?:er|ers)",
    r"catalog(?:s|ed|ing)?",
    r"analyz(?:e|es|ed|ing|er|ers)",
    r"paralyz(?:e|es|ed|ing)",
    r"travel(?:ed|ing|er|ers)",
    r"cancel(?:ed|ing)",
    r"model(?:ed|ing)",
    r"label(?:ed|ing)",
    r"fuel(?:ed|ing)",
    r"jewelry",
    r"gray(?:s|er|est|ish)?",
    r"defense(?:s|less)?",
    r"offense(?:s)?",
    r"pretense(?:s)?",
    r"aluminum",
    r"plow(?:s|ed|ing)?",
    r"cozy",
    r"pajamas",
    r"mustache(?:s)?",
    r"enrollment(?:s)?",
    r"fulfill(?:s|ment)?",
    r"skillful(?:ly)?",
    r"installment(?:s)?",
    r"airplane(?:s)?",
]
_BRITISH = [
    *(
        rf"{stem}our(?:s|ed|ing|ful|fully|less|able|ably|ite|ites|y|hood|hoods)?"
        for stem in _OR_WORDS
    ),
    r"cent(?:re|res|red)",
    r"theat(?:re|res)",
    r"fib(?:re|res)",
    r"lit(?:re|res)",
    r"catalogue(?:s|d)?",
    r"analys(?:e|es|ed|ing)",
    r"travel(?:led|ling|ler|lers)",
    r"cancel(?:led|ling)",
    r"model(?:led|ling)",
    r"label(?:led|ling)",
    r"jewellery",
    r"grey(?:s|er|est|ish)?",
    r"defence(?:s|less)?",
    r"offence(?:s)?",
    r"aluminium",
    r"plough(?:s|ed|ing)?",
    r"cosy",
    r"pyjamas",
    r"moustache(?:s)?",
    r"aeroplane(?:s)?",
    r"(?:real|organ|recogn|apolog|priorit|optim|custom|summar|categor|util|minim|maxim|"
    r"emphas|special|personal|standard|visual|memor|final|author|critic|familiar|stabil|"
    r"modern)is(?:e|es|ed|ing|ation|ations)",
]
_AMERICAN_RE = re.compile(r"\b(?:" + "|".join(_AMERICAN) + r")\b", re.IGNORECASE)
_BRITISH_RE = re.compile(r"\b(?:" + "|".join(_BRITISH) + r")\b", re.IGNORECASE)


def _spelling(prompt: str) -> list[Constraint]:
    if re.search(
        r"\b(?:british|uk)\s+(?:english\s+)?spellings?\b|\bbritish english\b", prompt, re.I
    ):

        def british(answer: str) -> str | None:
            found = sorted({m.group(0) for m in _AMERICAN_RE.finditer(_plain(answer))})
            return f"American spelling: {', '.join(found[:4])}" if found else None

        return [Constraint("spelling", "British spelling", "British spelling", british)]
    if re.search(
        r"\b(?:american|us)\s+(?:english\s+)?spellings?\b|\bamerican english\b", prompt, re.I
    ):

        def american(answer: str) -> str | None:
            found = sorted({m.group(0) for m in _BRITISH_RE.finditer(_plain(answer))})
            return f"British spelling: {', '.join(found[:4])}" if found else None

        return [Constraint("spelling", "American spelling", "American spelling", american)]
    return []


def _title(prompt: str) -> list[Constraint]:
    m = re.search(
        r"\btitle\b[^.\n]{0,30}?\b(?:in|inside|within|wrapped in|enclosed in)\s+(?:double\s+)?"
        r"(?P<b>square brackets|angular brackets|angle brackets|<<\s*>>|\[\s*\])",
        prompt,
        re.IGNORECASE,
    )
    if not m:
        return []
    square = "square" in m.group("b").lower() or "[" in m.group("b")
    pattern = re.compile(r"^\[[^\]\n]+\]" if square else r"^<<[^>\n]+>>")

    def test(answer: str) -> str | None:
        lines = [_plain(ln).strip() for ln in _FENCE.sub("", answer).splitlines() if ln.strip()]
        return (
            None if lines and pattern.match(lines[0]) else "no title in brackets on the first line"
        )

    kind = "square brackets" if square else "<<double angular brackets>>"
    return [Constraint("title", f"title in {kind}", m.group(0), test)]
