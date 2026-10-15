"""F3: the model refused. Keyword matching only; whether a refusal was *unwarranted*
(over-refusal) needs the judge, which arrives in Phase 3."""

from __future__ import annotations

import re

from counterfix.core.config import Config
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind

_REFUSAL = re.compile(
    r"\b(?:"
    r"I(?:'m| am)? (?:sorry,? )?(?:but )?(?:I(?:'m| am)? )?"
    r"(?:can(?:'|no)t|cannot|won'?t|will not|not able to|unable to) "
    r"(?:help|assist|provide|comply|do that|fulfil|fulfill|create|write|support|answer|"
    r"engage|share|give)"
    r"|I must (?:decline|refuse)"
    r"|I(?:'m| am) not (?:comfortable|going to) (?:help|provid|writ|assist|creat)"
    r"|(?:against|violates?) (?:my|our|the) (?:guidelines|policies|policy|usage policies)"
    r"|as an AI(?: language model)?,? I (?:can(?:'|no)t|cannot|am not able|don'?t)"
    r"|(?:is|are) not (?:aligned|in line|consistent) with my "
    r"(?:programming|values|guidelines|principles|purpose)"
    r"|I (?:do not|don'?t) (?:feel comfortable(?: \w+ing)?|create|produce|generate|write) "
    r"(?:that|this|such|content|material)"
    r")",
    re.IGNORECASE,
)
_WINDOW = 400  # refusals appear at the start; later matches are usually quotes or caveats

# "I cannot tell you when X did Y because X never did Y" is a correct premise correction,
# not an over-refusal. Seen live: qwen2.5:3b on a false-premise question (seed F5-003).
_CORRECTION = re.compile(
    r"\b(?:because|since|as)\b[^.!?\n]{0,160}\b(?:not|never|no|isn'?t|wasn'?t|didn'?t|"
    r"doesn'?t|aren'?t|weren'?t|incorrect|false|mistaken|actually)\b"
    r"|\b(?:premise|assumption)\b[^.!?\n]{0,60}\b(?:incorrect|false|wrong|mistaken)\b",
    re.IGNORECASE,
)


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(c, 0, start) for c in ".!?\n")
    stops = [i for i in (text.find(c, end) for c in ".!?\n") if i != -1]
    return text[left + 1 : min(stops) if stops else len(text)]


class RefusalDetector:
    """Flags answers that open with a refusal. Confidence is moderate on purpose:
    keyword checks misfire, so a judge confirms over-refusal in diagnose mode."""

    name = "refusal"
    covers = (FailureType.OVER_REFUSAL,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        answer = record.answer_text.replace("\u2019", "'")  # curly apostrophes
        if not answer:
            return []
        match = _REFUSAL.search(answer[:_WINDOW])
        if match is None:
            return []
        short = len(answer) < 600
        sentence = _sentence_around(answer, match.start(), match.end())
        correcting = _CORRECTION.search(sentence) is not None
        detail = f'refusal phrase near the start: "{match.group(0)}". '
        if correcting:
            detail += (
                "It may be correcting a false premise rather than refusing; the judge decides."
            )
        else:
            detail += "Whether the request was actually harmless needs the judge."
        return [
            Finding(
                failure_type=FailureType.OVER_REFUSAL,
                confidence=0.3 if correcting else (0.7 if short else 0.5),
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=detail,
                        data={
                            "phrase": match.group(0),
                            "short_answer": short,
                            "premise_correction": correcting,
                        },
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=(
                        "If the request is legitimate, state its purpose and audience "
                        "(e.g. 'for a chess lesson', 'for a school play')"
                    ),
                ),
            )
        ]
