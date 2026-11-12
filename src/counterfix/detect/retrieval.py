"""Retrieved passages ("chunks") on a call, so RAG metrics and checks can see them.

>>> with counterfix.context(chunks=docs):              # LangChain Documents, LlamaIndex
...     response = client.chat.completions.create(...)  # nodes, dicts or strings # doctest: +SKIP

A record's chunks live in ``record.metadata["chunks"]`` as a ranked list of
``{"id", "text", "score"}``; the joined text is also kept under ``"context"`` for the
judge checks. Gold labels for retrieval metrics go in ``record.metadata["relevant"]``:
a list of chunk ids or passage texts, or ``{id: grade}`` for graded relevance.

:func:`rag_messages` builds the prompt counterfix uses when an eval item gives a question
and its chunks; records built that way can be re-asked with the chunks changed (gold
passage, key passage first, fewer passages) during a fix search.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from counterfix.core.record import CallRecord, Message
from counterfix.judges.tasks import CONTEXT_KEY

CHUNKS_KEY = "chunks"
RELEVANT_KEY = "relevant"
QUESTION_KEY = "rag_question"
PROMPT_KEY = "rag_prompt"
"""Set when counterfix built the prompt from (question, chunks), so it can rebuild it."""

RAG_SYSTEM = (
    "Answer the question using the numbered context passages. If the passages do not "
    "contain the answer, say so instead of guessing."
)


class Chunk(BaseModel):
    """One retrieved passage, in retrieval order."""

    text: str
    id: str | None = None
    score: float | None = None


def _one(obj: Any) -> Chunk:
    if isinstance(obj, Chunk):
        return obj
    if isinstance(obj, str):
        return Chunk(text=obj)
    if isinstance(obj, Mapping):
        text = obj.get("text") or obj.get("page_content") or obj.get("content") or ""
        cid = obj.get("id") or (obj.get("metadata") or {}).get("id")
        score = obj.get("score")
        return Chunk(text=str(text), id=None if cid is None else str(cid), score=score)
    node = getattr(obj, "node", None)  # LlamaIndex NodeWithScore
    if node is not None and hasattr(node, "get_content"):
        return Chunk(
            text=node.get_content(),
            id=getattr(node, "node_id", None),
            score=getattr(obj, "score", None),
        )
    if hasattr(obj, "page_content"):  # LangChain Document
        meta = getattr(obj, "metadata", None) or {}
        cid = getattr(obj, "id", None) or meta.get("id") or meta.get("source")
        return Chunk(text=obj.page_content, id=None if cid is None else str(cid))
    if hasattr(obj, "get_content"):  # LlamaIndex node without score
        return Chunk(text=obj.get_content(), id=getattr(obj, "node_id", None))
    raise TypeError(f"cannot read a retrieved passage from {type(obj).__name__}")


def to_chunks(items: Any) -> list[Chunk]:
    """Chunks from strings, dicts, LangChain Documents, LlamaIndex nodes, or a LlamaIndex
    Response (its ``source_nodes``)."""
    if items is None:
        return []
    if hasattr(items, "source_nodes"):
        items = items.source_nodes
    if isinstance(items, (str, Mapping)) or not isinstance(items, Iterable):
        items = [items]
    return [_one(x) for x in items]


def joined(chunks: Iterable[Chunk]) -> str:
    """The chunks as numbered text: "[1] ...\\n\\n[2] ..."."""
    return "\n\n".join(f"[{n}] {c.text.strip()}" for n, c in enumerate(chunks, 1))


def set_chunks(record: CallRecord, chunks: Any, relevant: Any = None) -> CallRecord:
    """Attach chunks (and optional gold labels) to a record."""
    found = to_chunks(chunks)
    record.metadata[CHUNKS_KEY] = [c.model_dump() for c in found]
    if found:  # no passages: no context text, so nothing asks a judge about an empty one
        record.metadata[CONTEXT_KEY] = joined(found)
    if relevant is not None:
        record.metadata[RELEVANT_KEY] = relevant
    return record


def chunks_of(record: CallRecord) -> list[Chunk]:
    raw = record.metadata.get(CHUNKS_KEY)
    return [Chunk.model_validate(c) for c in raw] if isinstance(raw, list) else []


def rag_messages(
    question: str, chunks: Iterable[Chunk], system: str | None = None
) -> list[Message]:
    """The prompt for answering ``question`` from ``chunks`` (numbered, in order)."""
    user = f"Context:\n{joined(chunks)}\n\nQuestion: {question}"
    return [
        Message(role="system", content=system or RAG_SYSTEM),
        Message(role="user", content=user),
    ]


# --- relevance -------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def grades(record: CallRecord) -> list[float] | None:
    """Relevance of each retrieved chunk (0 = not relevant), from the gold labels, or None
    when there are no labels. Labels match a chunk by id, or by text (a gold passage
    contained in the chunk, or the chunk contained in it)."""
    relevant = record.metadata.get(RELEVANT_KEY)
    if relevant is None:
        return None
    graded: dict[str, float] = (
        {str(k): float(v) for k, v in relevant.items()}
        if isinstance(relevant, Mapping)
        else {str(k): 1.0 for k in ([relevant] if isinstance(relevant, str) else relevant)}
    )
    out = []
    for chunk in chunks_of(record):
        grade = 0.0
        if chunk.id is not None and chunk.id in graded:
            grade = graded[chunk.id]
        else:
            text = _norm(chunk.text)
            for key, g in graded.items():
                k = _norm(key)
                if len(k) > 20 and (k in text or (len(text) > 20 and text in k)):
                    grade = max(grade, g)
        out.append(grade)
    return out


def total_relevant(record: CallRecord) -> int | None:
    """How many gold passages exist (recall's denominator)."""
    relevant = record.metadata.get(RELEVANT_KEY)
    if relevant is None:
        return None
    if isinstance(relevant, str):
        return 1
    if isinstance(relevant, Mapping):
        return sum(1 for v in relevant.values() if float(v) > 0)
    return len(list(relevant))


def gold_texts(record: CallRecord) -> list[str]:
    """Gold passages given as text (for the "answer with the right passage" re-run)."""
    explicit = record.metadata.get("gold_passages")
    if isinstance(explicit, list):
        return [str(x) for x in explicit]
    relevant = record.metadata.get(RELEVANT_KEY)
    keys = (
        relevant.keys()
        if isinstance(relevant, Mapping)
        else ([relevant] if isinstance(relevant, str) else relevant or [])
    )
    chunks = chunks_of(record)
    by_id = {c.id: c.text for c in chunks if c.id is not None}
    out = []
    for k in keys:
        k = str(k)
        if k in by_id:
            out.append(by_id[k])
        elif len(k) > 20:
            out.append(k)
    return out


# --- re-asking with different passages (fix search) ------------------------------------------


def rebuildable(record: CallRecord) -> bool:
    """True when counterfix built this prompt from (question, chunks) and can rebuild it."""
    return bool(record.metadata.get(PROMPT_KEY)) and isinstance(
        record.metadata.get(QUESTION_KEY), str
    )


def with_chunks(record: CallRecord, chunks: list[Chunk]) -> tuple[list[Message], dict[str, Any]]:
    """The record's prompt rebuilt with other chunks, and the metadata that goes with it."""
    messages = rag_messages(
        str(record.metadata[QUESTION_KEY]), chunks, system=record.metadata.get("rag_system")
    )
    meta = {CHUNKS_KEY: [c.model_dump() for c in chunks], CONTEXT_KEY: joined(chunks)}
    return messages, meta


def gold_chunks(record: CallRecord) -> list[Chunk]:
    """The gold passages as chunks: those retrieved (by label) plus any given as text in
    ``metadata["gold_passages"]``. Empty when the gold text is unknown."""
    chunks = chunks_of(record)
    g = grades(record) or []
    found = [c for c, x in zip(chunks, g, strict=True) if x > 0]
    extra = record.metadata.get("gold_passages")
    if isinstance(extra, list):
        seen = {_norm(c.text) for c in found}
        found += [
            Chunk(text=str(t), id=f"gold-{n}")
            for n, t in enumerate(extra, 1)
            if _norm(str(t)) not in seen
        ]
    return found


def key_index(record: CallRecord, passage: int = 0) -> int | None:
    """0-based index of the passage holding the answer: the judge's pick (1-based
    ``passage``), else the first gold-labelled chunk; None if unknown."""
    chunks = chunks_of(record)
    if 1 <= passage <= len(chunks):
        return passage - 1
    g = grades(record) or []
    return next((i for i, x in enumerate(g) if x > 0), None)


# --- records from your own pipeline ----------------------------------------------------------


def record(
    question: str,
    answer: str,
    chunks: Any,
    *,
    relevant: Any = None,
    reference: str | None = None,
    model: str | None = None,
    latency_ms: float | None = None,
) -> CallRecord:
    """A record of one RAG answer from any pipeline, for ``diagnose()`` and ``evaluate()``:
    the question, the answer and the passages it was given (in rank order)."""
    from counterfix.judges.tasks import REFERENCE_KEY

    rec = CallRecord(
        model=model,
        messages=[Message(role="user", content=question)],
        response_text=answer,
        latency_ms=latency_ms,
        metadata={QUESTION_KEY: question},
    )
    if reference:
        rec.metadata[REFERENCE_KEY] = reference
    return set_chunks(rec, chunks, relevant)


def record_from_llamaindex(question: str, response: Any, **kwargs: Any) -> CallRecord:
    """A record from a LlamaIndex query: ``record_from_llamaindex(q, engine.query(q))``.
    The passages are the response's ``source_nodes``, with their scores."""
    return record(question, str(getattr(response, "response", response)), response, **kwargs)


def has_context(record: CallRecord) -> bool:
    """True when the call carries retrieved text (non-empty)."""
    text = record.metadata.get(CONTEXT_KEY)
    return isinstance(text, str) and bool(text.strip())


_ANSWER_KEYS = ("answer", "result", "output", "response", "text", "content")
_CONTEXT_KEYS = (
    "contexts",
    "retrieved_contexts",
    "context",
    "source_documents",
    "source_nodes",
    "documents",
    "sources",
    "chunks",
)


def _text_of(value: Any) -> str:
    """Answer text from a string, a chat message (``.content``) or a LlamaIndex response."""
    if isinstance(value, str):
        return value
    for attr in ("content", "response", "text"):
        inner = getattr(value, attr, None)
        if isinstance(inner, str):
            return inner
    return "" if value is None else str(value)


def _is_agent_run(output: Any) -> bool:
    from counterfix.detect.agents import Trajectory

    if isinstance(output, Trajectory):
        return True
    items = output.get("messages") if isinstance(output, Mapping) else output
    if not isinstance(items, list) or not items:
        return False
    for m in items:
        role = m.get("role") if isinstance(m, Mapping) else getattr(m, "type", None)
        calls = m.get("tool_calls") if isinstance(m, Mapping) else getattr(m, "tool_calls", None)
        if role in ("tool",) or calls:
            return True
    return False


def record_from_output(
    question: str, output: Any, *, latency_ms: float | None = None, model: str | None = None
) -> CallRecord:
    """A record from whatever an app or RAG pipeline returned for ``question``.

    Understands: a string; a ``CallRecord``; a response from a guarded client; a LlamaIndex
    ``Response`` (answer + ``source_nodes``); a dict such as LangChain's
    ``{"answer": ..., "context": [Document, ...]}`` or ``{"result": ...,
    "source_documents": [...]}`` (also ``contexts`` / ``retrieved_contexts``); a chat
    message with ``.content``; or an ``(answer, passages)`` pair.
    """
    if isinstance(output, CallRecord):
        output.metadata.setdefault(QUESTION_KEY, question)
        if output.latency_ms is None:
            output.latency_ms = latency_ms
        return output
    from counterfix.core.attach import get_diagnosis

    diagnosis = get_diagnosis(output) if not isinstance(output, (str, Mapping, tuple)) else None
    if diagnosis is not None and diagnosis.record is not None:
        rec = diagnosis.record
        rec.metadata.setdefault(QUESTION_KEY, question)
        return rec
    if _is_agent_run(output):  # a message list with tool calls, or a LangGraph state
        from counterfix.detect.agents import record as agent_record

        rec = agent_record(question, output)
        rec.latency_ms = latency_ms
        return rec
    answer: Any = output
    chunks: Any = None
    if isinstance(output, tuple) and len(output) == 2:
        answer, chunks = output
    elif isinstance(output, Mapping):
        answer = next((output[k] for k in _ANSWER_KEYS if output.get(k) is not None), "")
        chunks = next((output[k] for k in _CONTEXT_KEYS if output.get(k) is not None), None)
    elif hasattr(output, "source_nodes"):
        chunks = output
    rec = record(question, _text_of(answer), chunks, model=model, latency_ms=latency_ms)
    if chunks is None:  # the app gave no passages: not a RAG answer, keep no empty list
        rec.metadata.pop(CHUNKS_KEY, None)
    return rec
