"""Exact checks for instructions that code can verify, in the spirit of IFEval
(Zhou et al., 2023, "Instruction-Following Evaluation for Large Language Models").

LLM judges miscount sentences, words and letters as often as the models they judge, so
anything countable is checked here instead: sentence, word and item counts, letter case,
script of the reply language, British or American spelling, banned words, how items start,
ending with a question, a title in brackets, "only the number", "one word", "yes or no".
Instructions code cannot verify (tone, "mention X", "no competitor brands") are left to the
judge-based check. Costs nothing, so it runs in live mode.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detectors.basic import LENGTH_REASONS
from counterfix.text import normalise_numbers, straight_quotes

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


_ONLY_NUMBER = re.compile(r"^[-+−]?\$?\d[\d,.\s]*%?$")


def _answer_shape(prompt: str) -> list[Constraint]:
    out = []
    if re.search(
        r"\b(?:give|reply with|answer with|return|output|respond with|write)\s+(?:only|just)\s+"
        r"the\s+(?:final\s+)?(?:number|numeric answer|result)\b|\b(?:only|just)\s+the\s+number\b",
        prompt,
        re.IGNORECASE,
    ):

        def number(answer: str) -> str | None:
            core = normalise_numbers(_core(answer)).rstrip(".")
            core = re.sub(r"^\\\(|\\\)$|^\$+|\$+$", "", core).strip()
            return None if _ONLY_NUMBER.match(core) else "has more than the number"

        out.append(Constraint("only_number", "only the number", "only the number", number))
    for m in re.finditer(
        r"\b(?:in|with|answer in|reply in|respond in|use)?\s*(?:one|a single|1)[\s-]word\b"
        r"(?:\s+(?:answer|reply|response))?",
        prompt,
        re.IGNORECASE,
    ):
        if _per_item(prompt, m.start()):
            continue

        def one_word(answer: str) -> str | None:
            n = len(words(_core(answer)))
            return None if n == 1 else f"has {n} words"

        out.append(Constraint("one_word", "a one-word answer", m.group(0).strip(), one_word))
        break
    if re.search(
        r"\b(?:answer|reply|respond)\s+(?:with\s+)?(?:only\s+)?(?:a\s+)?(?:'?yes'?\s+or\s+'?no'?|"
        r"yes/no)\b|(?:^|[.?!:]\s*)yes\s+or\s+no\s*[.?!]?\s*$",
        prompt.strip(),
        re.IGNORECASE,
    ):

        def yes_no(answer: str) -> str | None:
            first = words(_core(answer))[:1]
            ok = bool(first) and re.sub(r"\W", "", first[0]).lower() in ("yes", "no")
            return None if ok else "does not start with yes or no"

        out.append(Constraint("yes_no", "starts with yes or no", "yes or no", yes_no))
    return out


_PARSERS = (
    _sentence_counts,
    _word_counts,
    _item_counts,
    _paragraph_counts,
    _numbered,
    _forbidden_words,
    _start_rules,
    _question_ending,
    _letter_case,
    _language,
    _spelling,
    _title,
    _answer_shape,
)


def extract_constraints(prompt: str) -> list[Constraint]:
    """Every instruction in ``prompt`` that code can verify, without duplicates."""
    prompt = straight_quotes(" ".join(prompt.split()))
    seen: dict[tuple[str, str], Constraint] = {}
    # In a JSON request, "<one sentence>" or "3 items" describe fields, not the whole reply.
    is_json = bool(re.search(r"\bjson\b", prompt, re.IGNORECASE))
    for parser in _PARSERS:
        if is_json and parser in (_sentence_counts, _word_counts, _item_counts, _paragraph_counts):
            continue
        for constraint in parser(prompt):
            seen.setdefault((constraint.kind, constraint.description), constraint)
    return list(seen.values())


_TOPIC = {
    "sentence_count": r"sentence",
    "word_count": r"\bwords?\b",
    "word_count_each": r"\bwords?\b",
    "item_count": r"bullet|\bitems?\b|\bpoints?\b|\btips?\b|\blist",
    "paragraph_count": r"paragraph",
    "numbered": r"number",
    "not_start": r"\bstart|\bbegin",
    "starts_with": r"\bstart|\bbegin",
    "ends_with": r"\bend",
    "ends_with_question": r"question",
    "lowercase": r"lower|capital|\bcase\b",
    "uppercase": r"upper|capital|\bcaps\b|\bcase\b",
    "language": r"language",
    "spelling": r"spelling|british|american",
    "title": r"title|bracket",
    "only_number": r"\bnumber\b",
    "one_word": r"one word|single word|one-word",
    "yes_no": r"\byes\b",
}


def mentions(constraint: Constraint, text: str) -> bool:
    """Is ``text`` (e.g. a judge's 'unmet instruction') about this constraint?"""
    pattern = _TOPIC.get(constraint.kind)
    if pattern and re.search(pattern, text, re.IGNORECASE):
        return True
    quoted = re.findall(r"'([^']+)'", constraint.description)  # banned or required words
    if quoted and any(q.lower() in text.lower() for q in quoted):
        return True
    if constraint.kind == "language":
        return constraint.description.split()[-1].lower() in text.lower()
    return False


# --- the detector -------------------------------------------------------------------------


class InstructionCheck:
    """F6: the answer breaks an instruction that code can verify exactly. No LLM calls."""

    name = "instructions"
    covers = (FailureType.DROPPED_INSTRUCTION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        constraints = extract_constraints(f"{record.system_text}\n{record.prompt_text}")
        if not constraints:
            raise SkipCheck("no instructions that code can check")
        if (record.finish_reason or "").lower() in LENGTH_REASONS:
            raise SkipCheck("the answer was cut off (see F1)")
        answer = record.answer_text
        if not answer.strip():
            raise SkipCheck("no answer to check")
        broken = [(c, why) for c in constraints if (why := c.violation(answer))]
        if not broken:
            return []
        detail = "; ".join(f"{c.description} ({why})" for c, why in broken)
        return [
            Finding(
                failure_type=FailureType.DROPPED_INSTRUCTION,
                confidence=0.9,
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=f"instructions not met: {detail}",
                        data={
                            "broken": [
                                {"kind": c.kind, "instruction": c.description, "found": why}
                                for c, why in broken
                            ],
                            "checked": [c.description for c in constraints],
                        },
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=(
                        "Repeat the broken instructions at the end of the prompt as a short "
                        "checklist. Models count words and sentences poorly: for hard limits, "
                        "check the reply in code and ask again when it fails"
                    ),
                ),
            )
        ]
