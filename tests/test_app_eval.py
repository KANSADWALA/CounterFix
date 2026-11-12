"""evaluate(app=...): your own app or RAG pipeline; concurrency; wider model specs."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from counterfix.core.record import CallRecord
from counterfix.detect.retrieval import chunks_of, record_from_output
from counterfix.evals import evaluate
from counterfix.evals.data import load_callable

ROWS = [
    {"id": "a", "question": "Capital of France?", "answer": "Paris", "relevant": ["d1"]},
    {"id": "b", "question": "Capital of Japan?", "answer": "Tokyo", "relevant": ["d2"]},
    {"id": "c", "question": "Capital of Peru?", "answer": "Lima", "relevant": ["d3"]},
]
DOCS = {"France": ("d1", "Paris is the capital of France."),
        "Japan": ("d2", "Tokyo is the capital of Japan."),
        "Peru": ("d9", "Peru exports copper.")}  # fmt: skip


def rag_app(question: str) -> dict[str, Any]:
    country = question.split()[-1].rstrip("?")
    doc_id, text = DOCS[country]
    answer = text.split()[0] if "capital" in text else "I don't know"
    return {"answer": answer, "context": [{"id": doc_id, "text": text}]}


def test_app_with_retrieval_metrics():
    result = evaluate(
        ROWS, app=rag_app, input="question", reference="answer",
        metrics=["exact_match", "hit@1"],
    )  # fmt: skip
    assert result.model == "app:rag_app"
    assert [i.passed for i in result.items] == [True, True, False]
    hit = result.metric("hit@1")
    assert hit.passed == 2
    miss = result.items[2]
    assert "F10" in miss.causes  # the free label check names the retrieval miss
    assert chunks_of(miss.record)[0].id == "d9"
    assert miss.record.latency_ms is not None


def test_app_errors_become_infra():
    def broken(q: str) -> str:
        raise RuntimeError("index offline")

    result = evaluate(ROWS[:1], app=broken, input="question", reference="answer")
    assert result.items[0].causes == ["INFRA"]


def test_async_app():
    async def app(q: str) -> str:
        return "Paris"

    result = evaluate(ROWS[:1], app=app, input="question", reference="answer")
    assert result.items[0].passed


def test_model_and_app_conflict():
    with pytest.raises(ValueError):
        evaluate(ROWS, app=rag_app, model=SimpleNamespace(name="x"), input="question")


def test_concurrency_keeps_order_and_runs_in_parallel():
    active, peak, lock = [0], [0], threading.Lock()

    def slow(q: str) -> str:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return {"France?": "Paris", "Japan?": "Tokyo", "Peru?": "Lima"}[q.split()[-1]]

    result = evaluate(ROWS, app=slow, input="question", reference="answer", concurrency=3)
    assert [i.id for i in result.items] == ["a", "b", "c"]
    assert all(i.passed for i in result.items)
    assert peak[0] > 1


def test_record_from_output_shapes():
    q = "Q?"
    assert record_from_output(q, "A").response_text == "A"
    lc = record_from_output(q, {"result": "A", "source_documents": [
        SimpleNamespace(page_content="p", metadata={"id": "x"})]})  # fmt: skip
    assert lc.response_text == "A" and chunks_of(lc)[0].text == "p"
    pair = record_from_output(q, ("A", ["p1", "p2"]))
    assert len(chunks_of(pair)) == 2
    msg = record_from_output(q, SimpleNamespace(content="A"))
    assert msg.response_text == "A" and "chunks" not in msg.metadata
    rec = CallRecord(response_text="A")
    assert record_from_output(q, rec) is rec


def test_load_callable(tmp_path):
    f = tmp_path / "myapp.py"
    f.write_text("def answer(q):\n    return 'x'\n", encoding="utf-8")
    assert load_callable(f"{f}:answer")("q") == "x"
    with pytest.raises(ValueError):
        load_callable("no_colon")


def test_caller_specs(monkeypatch):
    pytest.importorskip("openai")
    from counterfix.core.presets import caller_for, pace_for

    c = caller_for("my-model@http://localhost:8000/v1")
    assert c.name.endswith("my-model") or "my-model" in c.name
    assert pace_for("my-model@http://localhost:8000/v1") == 0.0
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(SimpleNamespace and __import__("tempfile").gettempdir())
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return
    with pytest.raises(RuntimeError):
        caller_for("anthropic:claude-x")
