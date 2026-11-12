"""Reference metrics on text: exact match, token F1 and contains.

Exact match and token F1 follow the SQuAD evaluation script (lowercase, drop punctuation
and the articles a/an/the, collapse spaces). With several references a metric takes the
best match. For ROUGE, BLEU, chrF and embedding scores, use your evaluation tool.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.evals.metrics.base import Metric, references

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """SQuAD normalisation: lowercase, no punctuation, no articles, single spaces."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


class ExactMatch(Metric):
    """1.0 when the normalised answer equals a normalised reference."""

    name = "exact_match"
    default_threshold = 1.0

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        answer = normalize(record.answer_text)
        hit = any(answer == normalize(r) for r in references(reference))
        return float(hit), "matches the reference" if hit else f"answer {answer[:60]!r} differs"


class TokenF1(Metric):
    """Word overlap with the best reference (SQuAD F1), 0 to 1."""

    name = "f1"

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        best = max(token_f1(record.answer_text, r) for r in references(reference))
        return best, f"token F1 {best:.2f}"


class Contains(Metric):
    """1.0 when a reference appears in the answer (after normalisation)."""

    name = "contains"
    default_threshold = 1.0

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        answer = f" {normalize(record.answer_text)} "
        hit = any(f" {normalize(r)} " in answer for r in references(reference) if normalize(r))
        return float(hit), "reference found in the answer" if hit else "reference not in the answer"
