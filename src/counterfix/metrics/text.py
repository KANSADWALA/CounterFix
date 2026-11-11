"""Reference metrics on text: exact match, token F1, contains, ROUGE-L, BLEU and chrF.

Exact match and token F1 follow the SQuAD evaluation script (lowercase, drop punctuation
and the articles a/an/the, collapse spaces). ROUGE-L is computed here with the same
tokenizer as Google's ``rouge-score`` package (no stemming), so the two agree. BLEU and
chrF use ``sacrebleu`` (``pip install "counterfix[metrics]"``). With several references
a metric takes the best match.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.metrics.base import Metric, references

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


def _rouge_tokens(text: str) -> list[str]:
    """rouge-score's default tokenizer: lowercase, non-alphanumerics become spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure (longest common subsequence of tokens)."""
    pred, ref = _rouge_tokens(prediction), _rouge_tokens(reference)
    lcs = _lcs(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
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


class RougeL(Metric):
    """ROUGE-L F-measure with the best reference, 0 to 1."""

    name = "rouge_l"

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        best = max(rouge_l(record.answer_text, r) for r in references(reference))
        return best, f"ROUGE-L {best:.2f}"


def _sacrebleu() -> Any:
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError(
            'BLEU and chrF need sacrebleu: pip install "counterfix[metrics]"'
        ) from exc
    return sacrebleu


class Bleu(Metric):
    """Sentence BLEU (sacrebleu), scaled to 0 to 1."""

    name = "bleu"

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        score = _sacrebleu().sentence_bleu(record.answer_text, references(reference)).score / 100
        return score, f"BLEU {score:.2f}"


class Chrf(Metric):
    """Sentence chrF (sacrebleu), scaled to 0 to 1."""

    name = "chrf"

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        score = _sacrebleu().sentence_chrf(record.answer_text, references(reference)).score / 100
        return score, f"chrF {score:.2f}"
