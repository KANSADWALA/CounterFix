"""Tell counterfix what the model was given (RAG) or what the right answer is.

>>> with counterfix.context(source=retrieved_text):        # doctest: +SKIP
...     response = client.chat.completions.create(...)

Calls made inside the block carry the text on ``record.metadata``. Judge checks then
tell "the source was missing" from "the source was ignored" (F4), and ``fix()`` can try
the call with the source in the prompt. ``reference`` is a known correct answer, used by
judges and by prompt regression tests. Works with ``guard()``, ``@watch`` and the
callbacks; it is per thread and per async task.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.judges.tasks import CONTEXT_KEY, REFERENCE_KEY

_current: ContextVar[dict[str, Any] | None] = ContextVar("counterfix_context", default=None)


@contextmanager
def context(
    *,
    source: str | None = None,
    reference: str | None = None,
    chunks: Any = None,
    relevant: Any = None,
) -> Iterator[None]:
    """Attach a retrieved source and/or a reference answer to the calls in this block.

    ``chunks`` are the retrieved passages in rank order (strings, dicts, LangChain
    Documents, LlamaIndex nodes or a LlamaIndex Response); they also become ``source``.
    ``relevant`` are gold labels (ids or passage texts) for the retrieval metrics.
    """
    extra: dict[str, Any] = {
        k: v for k, v in ((CONTEXT_KEY, source), (REFERENCE_KEY, reference)) if v
    }
    if chunks is not None:
        from counterfix.detect.retrieval import CHUNKS_KEY, joined, to_chunks

        found = to_chunks(chunks)
        extra[CHUNKS_KEY] = [c.model_dump() for c in found]
        if found:
            extra.setdefault(CONTEXT_KEY, joined(found))
    if relevant is not None:
        from counterfix.detect.retrieval import RELEVANT_KEY

        extra[RELEVANT_KEY] = relevant
    token = _current.set({**(_current.get() or {}), **extra})
    try:
        yield
    finally:
        _current.reset(token)


def apply(record: CallRecord) -> CallRecord:
    """Copy the active context onto a record (adapters call this)."""
    extra = _current.get()
    if extra:
        record.metadata.update(extra)
    return record
