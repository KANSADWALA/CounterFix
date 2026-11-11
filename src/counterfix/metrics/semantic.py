"""Meaning-level similarity: embedding cosine similarity and BERTScore.

Both run a local model (``pip install "counterfix[semantic]"``); nothing is sent to an
API. The model loads once per metric object.
"""

from __future__ import annotations

from typing import Any

from counterfix.core.record import CallRecord
from counterfix.metrics.base import Metric, references


class EmbeddingSimilarity(Metric):
    """Cosine similarity of sentence embeddings with the best reference (-1 to 1)."""

    name = "embedding_similarity"

    def __init__(
        self, threshold: float | None = None, model: str = "all-MiniLM-L6-v2", **options: Any
    ):
        super().__init__(threshold, **options)
        self.model_name = model
        self._model: Any = None

    def _encoder(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "embedding_similarity needs sentence-transformers: "
                    'pip install "counterfix[semantic]"'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        from sentence_transformers import util

        model = self._encoder()
        refs = references(reference)
        vectors = model.encode([record.answer_text, *refs], convert_to_tensor=True)
        best = float(util.cos_sim(vectors[0:1], vectors[1:]).max())
        return best, f"cosine similarity {best:.2f} ({self.model_name})"


class BertScore(Metric):
    """BERTScore F1 with the best reference (``bert-score`` package)."""

    name = "bertscore"

    def __init__(self, threshold: float | None = None, lang: str = "en", **options: Any):
        super().__init__(threshold, **options)
        self.lang = lang
        self._scorer: Any = None

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        if self._scorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError as exc:
                raise ImportError(
                    'bertscore needs bert-score: pip install "counterfix[semantic]"'
                ) from exc
            self._scorer = BERTScorer(lang=self.lang)
        refs = references(reference)
        _, _, f1 = self._scorer.score([record.answer_text] * len(refs), refs)
        best = float(f1.max())
        return best, f"BERTScore F1 {best:.2f}"
