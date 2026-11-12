"""Retrieval metrics from gold labels: hit@k, recall@k, precision@k, MRR, nDCG@k.

They read the ranked chunks and the gold labels from the record (see
:mod:`counterfix.detect.retrieval`); no LLM is called. Write ``recall@5`` for the top 5,
or ``recall`` for every retrieved chunk. Definitions follow trec_eval: precision@k divides by k,
nDCG uses linear gain (grade / log2(rank + 1)) with the ideal ranking of all gold
passages, retrieved or not. Items without labels are not scored.
"""

from __future__ import annotations

import math
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.evals.metrics.base import Metric


class _Retrieval(Metric):
    needs_reference = False

    def __init__(self, threshold: float | None = None, k: int | None = None, **options: Any):
        super().__init__(threshold, **options)
        self.k = k
        if k is not None:
            self.name = f"{type(self).name}@{k}"

    def _top(self, record: CallRecord) -> tuple[list[float], int] | None:
        from counterfix.detect.retrieval import grades, total_relevant

        g, total = grades(record), total_relevant(record)
        if g is None or total is None:
            return None
        return (g if self.k is None else g[: self.k]), total

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        top = self._top(record)
        if top is None:
            return None, "no gold labels for the retrieved passages"
        return self.measure(*top)

    def measure(self, top: list[float], total: int) -> tuple[float | None, str]:
        raise NotImplementedError


class HitRate(_Retrieval):
    """1.0 when a gold passage is among the retrieved ones (top k)."""

    name = "hit"
    default_threshold = 1.0

    def measure(self, top: list[float], total: int) -> tuple[float | None, str]:
        hit = any(g > 0 for g in top)
        where = next((i for i, g in enumerate(top, 1) if g > 0), None)
        return float(hit), f"gold passage at rank {where}" if hit else "no gold passage retrieved"


class Recall(_Retrieval):
    """Share of the gold passages that were retrieved (top k)."""

    name = "recall"

    def measure(self, top: list[float], total: int) -> tuple[float | None, str]:
        if total == 0:
            return None, "the labels list no gold passage"
        found = sum(1 for g in top if g > 0)
        return min(found / total, 1.0), f"{found} of {total} gold passage(s) retrieved"


class Precision(_Retrieval):
    """Share of the retrieved passages (top k) that are gold."""

    name = "precision"

    def measure(self, top: list[float], total: int) -> tuple[float | None, str]:
        size = self.k or len(top)
        if size == 0:
            return None, "nothing was retrieved"
        found = sum(1 for g in top if g > 0)
        return found / size, f"{found} of {size} retrieved passage(s) are gold"


class Mrr(_Retrieval):
    """1 / rank of the first gold passage (0 if none in the top k)."""

    name = "mrr"

    def measure(self, top: list[float], total: int) -> tuple[float | None, str]:
        rank = next((i for i, g in enumerate(top, 1) if g > 0), None)
        return (
            1 / rank if rank else 0.0
        ), f"first gold passage at rank {rank}" if rank else "no gold passage"


class Ndcg(_Retrieval):
    """Normalised discounted cumulative gain of the ranking (top k), 0 to 1."""

    name = "ndcg"

    def _ideal(self, record: CallRecord) -> list[float]:
        from counterfix.detect.retrieval import RELEVANT_KEY

        relevant = record.metadata.get(RELEVANT_KEY)
        if isinstance(relevant, dict):
            ideal = sorted((float(v) for v in relevant.values()), reverse=True)
        else:
            ideal = [1.0] * (1 if isinstance(relevant, str) else len(list(relevant or [])))
        return ideal if self.k is None else ideal[: self.k]

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        top = self._top(record)
        if top is None:
            return None, "no gold labels for the retrieved passages"
        dcg = sum(g / math.log2(i + 1) for i, g in enumerate(top[0], 1))
        idcg = sum(g / math.log2(i + 1) for i, g in enumerate(self._ideal(record), 1))
        if idcg == 0:
            return None, "the labels list no gold passage"
        value = dcg / idcg
        return value, f"nDCG {value:.2f}"
