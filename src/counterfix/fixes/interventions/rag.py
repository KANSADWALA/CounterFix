"""Interventions for RAG (F10 retrieval miss, F11 context ignored): reorder, trim or
replace the passages the model saw."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from counterfix.core.diagnosis import Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import PrescriptionKind
from counterfix.fixes.interventions.base import (
    Change,
    F,
    Intervention,
    _system,
)


def _passage(f: Finding) -> int:
    for e in f.evidence:
        try:
            return int(e.data.get("passage") or 0)
        except (TypeError, ValueError):
            continue
    return 0


def _rag_order(
    kind: str,
) -> tuple[
    Callable[[CallRecord, Finding], Change | None],
    Callable[[CallRecord, Finding], dict[str, Any]],
]:
    """A fix that re-asks with the passages changed: "gold" (the right passage added
    first), "key_first" (the passage with the answer moved first) or "fewer" (top 3)."""
    from counterfix.detect import retrieval as rag

    def chunks(r: CallRecord, f: Finding) -> list[rag.Chunk] | None:
        if not rag.rebuildable(r):
            return None
        have = rag.chunks_of(r)
        if kind == "gold":
            gold = rag.gold_chunks(r)
            if not gold:
                return None
            golds = {c.text for c in gold}
            return gold + [c for c in have if c.text not in golds]
        key = rag.key_index(r, _passage(f))
        if kind == "key_first":
            if key is None or key == 0:
                return None
            return [have[key], *have[:key], *have[key + 1 :]]
        if len(have) <= 3:
            return None
        if key is not None and key >= 3:
            return [have[key], *have[:2]]
        return have[:3]

    def apply(r: CallRecord, f: Finding) -> Change | None:
        new = chunks(r, f)
        return None if new is None else (rag.with_chunks(r, new)[0], dict(r.params))

    def metadata(r: CallRecord, f: Finding) -> dict[str, Any]:
        new = chunks(r, f)
        if new is None:
            return {}
        meta = rag.with_chunks(r, new)[1]
        if kind == "gold":  # the added gold passages are gold: say so, for the label check
            relevant = r.metadata.get(rag.RELEVANT_KEY)
            ids = [c.id for c in rag.gold_chunks(r) if c.id is not None]
            if isinstance(relevant, dict):
                meta[rag.RELEVANT_KEY] = {**relevant, **{i: 1 for i in ids}}
            elif relevant is not None:
                base = [relevant] if isinstance(relevant, str) else list(relevant)
                meta[rag.RELEVANT_KEY] = base + [i for i in ids if i not in base]
        return meta

    return apply, metadata


_gold_apply, _gold_meta = _rag_order("gold")
_first_apply, _first_meta = _rag_order("key_first")
_fewer_apply, _fewer_meta = _rag_order("fewer")


RAG: list[Intervention] = [
    Intervention(
        "gold_passage",
        F.RETRIEVAL_MISS,
        PrescriptionKind.ADD_CONTEXT,
        "With the right passage in the context the answer passes: fix the retriever (more "
        "passages, a rewritten query, hybrid search)",
        _gold_apply,
        _gold_meta,
    ),
    Intervention(
        "key_passage_first",
        F.CONTEXT_IGNORED,
        PrescriptionKind.PROMPT_CHANGE,
        "Put the passage that holds the answer first (rerank)",
        _first_apply,
        _first_meta,
    ),
    Intervention(
        "fewer_passages",
        F.CONTEXT_IGNORED,
        PrescriptionKind.PROMPT_CHANGE,
        "Send fewer passages (the top 3, the one with the answer first)",
        _fewer_apply,
        _fewer_meta,
    ),
    Intervention(
        "quote_then_answer",
        F.CONTEXT_IGNORED,
        PrescriptionKind.PROMPT_CHANGE,
        "Ask the model to quote the passage that answers the question, then answer from it",
        _system(
            "Before answering, quote the sentence from the context that answers the "
            "question. Then give the answer using only that information."
        ),
    ),
]
