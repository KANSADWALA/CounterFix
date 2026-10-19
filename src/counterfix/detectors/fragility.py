"""F7: prompt fragility. Ask the same thing in other words; flag it when the answer flips.

The rewordings come from ``record.metadata["variants"]`` when you have them (the probe
suite does), otherwise the model itself writes them. Each reworded prompt is answered
once with the call's own settings. A judge, when given, decides whether two answers to
the two wordings say the same thing ("No" to "Is 9.11 larger than 9.9?" agrees with
"9.9 is larger"); without one, a cheap number and word-overlap test is used and the
finding is marked inferred, at lower confidence. Short answers only. It costs 2 calls per
rewording when the model writes them, 1 when you supply them, so it never runs live:
pass it to ``diagnose(detectors=...)`` or use it in a profile.
"""

from __future__ import annotations

import copy
from typing import Any

from counterfix.core.caller import ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detectors.sampling import same_meaning_heuristic
from counterfix.judges.base import Judge, JudgeTask

VARIANTS_KEY = "variants"
REWRITE = (
    "Rewrite the request below in different words. Keep its meaning, its instructions and "
    "any quoted text exactly. Reply with the rewritten request only.\n\n<request>\n{text}\n"
    "</request>"
)


class FragilityCheck:
    """Flags F7 when rewordings of the request get answers with a different meaning."""

    name = "fragility"
    covers = (FailureType.PROMPT_FRAGILITY,)

    def __init__(
        self, caller: ModelCaller, *, judge: Judge | None = None, n: int = 3, max_words: int = 40
    ):
        self.caller, self.judge, self.n, self.max_words = caller, judge, n, max_words

    def _same(self, q1: str, a1: str, q2: str, a2: str) -> bool:
        if self.judge is None:
            return same_meaning_heuristic(a1, a2)
        verdict = self.judge.ask(
            JudgeTask(
                name="same_answer",
                instructions=(
                    "Two wordings of one request were answered separately. Do the two "
                    "answers give the same answer to it (same choice, number or fact), "
                    "ignoring wording, length and explanations?"
                ),
                material={"request_a": q1, "answer_a": a1, "request_b": q2, "answer_b": a2},
                labels=("same", "different"),
            )
        )
        return verdict.label == "same" if not verdict.failed else same_meaning_heuristic(a1, a2)

    def _variants(self, record: CallRecord, params: dict[str, Any]) -> list[str]:
        given = record.metadata.get(VARIANTS_KEY)
        if isinstance(given, list) and given:
            return [str(v) for v in given[: self.n]]
        out = []
        for _ in range(self.n):
            ask = [Message(role="user", content=REWRITE.format(text=record.prompt_text))]
            text = self.caller.complete(ask, **params).answer_text.strip()
            if text and text not in out:
                out.append(text)
        return out

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        original = record.answer_text.strip()
        if not original:
            raise SkipCheck("no answer to compare")
        if len(original.split()) > self.max_words:
            raise SkipCheck(f"answer longer than {self.max_words} words")
        if not any(m.role == "user" for m in record.messages):
            raise SkipCheck("no user message to reword")
        params = {k: v for k, v in record.params.items() if k not in ("stream", "n")}
        try:
            variants = self._variants(record, params)
            answers = []
            for variant in variants:
                messages = copy.deepcopy(record.messages)
                last = max(i for i, m in enumerate(messages) if m.role == "user")
                messages[last] = Message(role="user", content=variant)
                answers.append((variant, self.caller.complete(messages, **params).answer_text))
        except Exception as exc:
            raise SkipCheck(f"could not re-ask the model: {exc}") from exc
        asked = record.prompt_text
        flipped = [(v, a) for v, a in answers if a and not self._same(asked, original, v, a)]
        if not answers or not flipped:
            return []
        agreement = 1 - len(flipped) / len(answers)
        return [
            Finding(
                failure_type=FailureType.PROMPT_FRAGILITY,
                confidence=round(
                    min(0.9, 0.5 + len(flipped) / (2 * len(answers)))
                    * (1.0 if self.judge else 0.7),
                    3,
                ),
                inferred=self.judge is None,
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=(
                            f"{len(flipped)} of {len(answers)} rewordings got a different "
                            f'answer, e.g. "{flipped[0][1][:80]}" instead of "{original[:80]}"'
                        ),
                        data={
                            "agreement": round(agreement, 3),
                            "flipped": [{"prompt": v, "answer": a} for v, a in flipped],
                        },
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=(
                        "Make the prompt less sensitive to wording: state the expected answer "
                        "format and add one or two worked examples (few-shot)"
                    ),
                ),
            )
        ]
