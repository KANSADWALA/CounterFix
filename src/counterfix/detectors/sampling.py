"""F4 signal from sampling: ask the same question several more times and see whether the
answers agree. Confabulated answers tend to change between samples; known facts do not
(the idea behind SelfCheckGPT and semantic entropy).

Short answers only: clustering long free text by surface similarity is unreliable, so
long answers are skipped and left to the judge. A judge can be given to decide whether
two answers mean the same thing, at the cost of extra calls.
"""

from __future__ import annotations

import math
import re
from typing import Any

from counterfix.core.caller import ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.judges.base import Judge, JudgeTask

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace(",", "")))


def same_meaning_heuristic(a: str, b: str) -> bool:
    """Cheap equivalence for short answers: same numbers, or strong word overlap."""
    na, nb = _NUM.findall(a.replace(",", "")), _NUM.findall(b.replace(",", ""))
    if na or nb:
        return sorted(set(na)) == sorted(set(nb))
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return ta == tb
    return len(ta & tb) / len(ta | tb) >= 0.6


class SamplingCheck:
    """Re-asks the model ``n`` times at a higher temperature and measures agreement.

    Reports the share of answers in the largest meaning-cluster (agreement) and the
    normalised entropy over clusters. Flags F4 when agreement is below ``threshold``.
    """

    name = "sampling"
    covers = (FailureType.HALLUCINATION,)

    def __init__(
        self,
        caller: ModelCaller,
        *,
        n: int = 4,
        temperature: float = 1.0,
        threshold: float = 0.6,
        max_words: int = 60,
        judge: Judge | None = None,
    ):
        self.caller, self.n, self.temperature = caller, n, temperature
        self.threshold, self.max_words, self.judge = threshold, max_words, judge

    def _same(self, a: str, b: str) -> bool:
        if self.judge is None:
            return same_meaning_heuristic(a, b)
        verdict = self.judge.ask(
            JudgeTask(
                name="same_meaning",
                instructions="Do these two answers to the same question state the same "
                "answer (same facts, same numbers), ignoring wording and length?",
                material={"answer_a": a, "answer_b": b},
                labels=("same", "different"),
            )
        )
        return verdict.label == "same" if not verdict.failed else same_meaning_heuristic(a, b)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        original = record.answer_text
        if not original:
            raise SkipCheck("no answer to compare")
        if len(original.split()) > self.max_words:
            raise SkipCheck(f"answer longer than {self.max_words} words; left to the judge")
        params: dict[str, Any] = {
            k: v for k, v in record.params.items() if k not in ("stream", "n", "temperature")
        }
        samples = [original]
        for _ in range(self.n):
            try:
                again = self.caller.complete(
                    record.messages, temperature=self.temperature, **params
                )
            except Exception as exc:
                raise SkipCheck(f"could not re-sample the model: {exc}") from exc
            if again.answer_text:
                samples.append(again.answer_text)
        if len(samples) < 3:
            raise SkipCheck("too few samples")

        clusters: list[list[str]] = []
        for answer in samples:
            for cluster in clusters:
                if self._same(cluster[0], answer):
                    cluster.append(answer)
                    break
            else:
                clusters.append([answer])
        sizes = sorted((len(c) for c in clusters), reverse=True)
        total = len(samples)
        agreement = sizes[0] / total
        entropy = -sum((s / total) * math.log(s / total) for s in sizes)
        normalised = entropy / math.log(total) if total > 1 else 0.0
        original_agrees = self._same(clusters[0][0], original) if clusters else True
        data = {
            "agreement": round(agreement, 3),
            "entropy": round(normalised, 3),
            "n_samples": total,
            "clusters": [c[0][:120] for c in clusters],
            "original_in_majority": original_agrees,
        }
        if agreement >= self.threshold:
            return []
        return [
            Finding(
                failure_type=FailureType.HALLUCINATION,
                confidence=round(min(0.9, 1 - agreement), 3),
                inferred=True,
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=(
                            f"the model gave {len(clusters)} different answers in {total} "
                            f"samples (agreement {agreement:.0%}); it is likely guessing"
                        ),
                        data=data,
                    )
                ],
                prescription=Prescription(
                    kind=PrescriptionKind.ADD_CONTEXT,
                    description="Give the model the source (retrieval), or allow 'I don't know'",
                ),
            )
        ]
