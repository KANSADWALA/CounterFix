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

from counterfix._internal.text import normalise_numbers, straight_quotes
from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detect.basic import LENGTH_REASONS
from counterfix.detect.instructions.constraint import (
    _CMP as _CMP,
    _COMPARE as _COMPARE,
    _RELATION as _RELATION,
    Constraint as Constraint,
    _counted as _counted,
    _per_item as _per_item,
)
from counterfix.detect.instructions.counts import (
    _ITEM_COUNT as _ITEM_COUNT,
    _PARAGRAPH_COUNT as _PARAGRAPH_COUNT,
    _SENTENCE_COUNT as _SENTENCE_COUNT,
    _WORD_COUNT as _WORD_COUNT,
    _item_counts as _item_counts,
    _numbered as _numbered,
    _paragraph_counts as _paragraph_counts,
    _sentence_counts as _sentence_counts,
    _word_counts as _word_counts,
    paragraphs as paragraphs,
)
from counterfix.detect.instructions.language import (
    _AMERICAN as _AMERICAN,
    _AMERICAN_RE as _AMERICAN_RE,
    _BRITISH as _BRITISH,
    _BRITISH_RE as _BRITISH_RE,
    _LANGUAGE as _LANGUAGE,
    _OR_WORDS as _OR_WORDS,
    _SCRIPTS as _SCRIPTS,
    _language as _language,
    _spelling as _spelling,
    _title as _title,
    script_share as script_share,
)
from counterfix.detect.instructions.reading import (
    _FENCE as _FENCE,
    _HEADING as _HEADING,
    _ITEM as _ITEM,
    _MARKUP as _MARKUP,
    _NUM as _NUM,
    _SENTENCE_END as _SENTENCE_END,
    _SPLIT as _SPLIT,
    _TRAILING_JUNK as _TRAILING_JUNK,
    _WORD_NUMBERS as _WORD_NUMBERS,
    _core as _core,
    _lines as _lines,
    _number as _number,
    _plain as _plain,
    is_numbered as is_numbered,
    list_items as list_items,
    sentences as sentences,
    words as words,
)
from counterfix.detect.instructions.wording import (
    _FORBID_HEAD as _FORBID_HEAD,
    _QUOTED as _QUOTED,
    _forbidden_words as _forbidden_words,
    _letter_case as _letter_case,
    _question_ending as _question_ending,
    _start_rules as _start_rules,
)

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


def _instruction_text(record: CallRecord) -> str:
    """The instructions the app and the user gave: system and user text, without retrieved
    passages pasted into the prompt. Instructions inside retrieved text are not the user's;
    following them is F13 (prompt injection), ignoring them is right."""
    from counterfix.judges.tasks import CONTEXT_KEY

    text = f"{record.system_text}\n{record.prompt_text}"
    context = record.metadata.get(CONTEXT_KEY)
    if isinstance(context, str) and context.strip():
        text = text.replace(context, "")
    return text


class InstructionCheck:
    """F6: the answer breaks an instruction that code can verify exactly. No LLM calls."""

    name = "instructions"
    covers = (FailureType.DROPPED_INSTRUCTION,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        constraints = extract_constraints(_instruction_text(record))
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
