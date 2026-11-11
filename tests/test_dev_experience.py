"""Rough edges found by using counterfix as a new developer would (2026-11-11)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import confidence
from counterfix.cli import main
from counterfix.evals import evaluate


def test_guard_rejects_a_class_and_non_clients():
    with pytest.raises(TypeError, match=r"guard\(FakeClient\(\)\)"):
        counterfix.guard(FakeClient)
    with pytest.raises(TypeError, match="watch"):
        counterfix.guard({})


def test_did_you_mean_hints():
    with pytest.raises(ValueError, match="did you mean 'exact_match'"):
        evaluate(
            [{"id": "a", "prompt": "q", "response": "x", "answer": "x"}], metrics=["exact_mtch"]
        )
    from counterfix.benchmarks import run_benchmark

    with pytest.raises(ValueError, match="did you mean 'gsm8k'"):
        run_benchmark("gsm8", model="ollama:x")
    from counterfix.presets import caller_for

    with pytest.raises(ValueError, match="did you mean 'ollama'"):
        caller_for("olama:qwen2.5:3b")


def test_eval_exits_2_when_every_call_failed(tmp_path, monkeypatch, capsys):
    (tmp_path / "broken_app.py").write_text(
        "def ask(q):\n    raise RuntimeError('model not found')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    data = tmp_path / "d.jsonl"
    rows = [{"id": str(i), "question": "q", "answer": "a"} for i in range(2)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    code = main(["eval", str(data), "--app", "broken_app:ask", "--metric", "contains"])
    assert code == 2
    assert "no item got an answer" in capsys.readouterr().out
    sys.modules.pop("broken_app", None)


def test_app_fix_and_id_mismatch_notes():
    rows = [
        {"id": "a", "question": "Capital of France?", "answer": "Paris", "relevant": ["p1"]},
        {"id": "b", "question": "Capital of Japan?", "answer": "Tokyo", "relevant": ["p2"]},
    ]

    def app(q: str):
        return "no idea", [{"id": "doc-7", "text": "Unrelated passage about trains."}]

    result = evaluate(rows, app=app, metrics=["contains", "recall"], fix=2)
    text = result.text()
    assert "fix search skipped" in text
    assert "no retrieved passage matched" in text and "doc-7" in text and "p1" in text
    assert "Note:" in result.html()


def test_no_id_note_when_ids_match():
    rows = [
        {"id": "a", "question": "q1", "answer": "Paris", "relevant": ["d1"]},
        {"id": "b", "question": "q2", "answer": "Tokyo", "relevant": ["d2"]},
    ]

    def app(q: str):
        return "Paris", [{"id": "d1", "text": "Paris is the capital of France."}]

    result = evaluate(rows, app=app, metrics=["recall"])
    assert not any("matched" in n for n in result.notes)


def test_confidence_reads_a_guarded_response():
    resp = make_response("Tokyo")
    resp.choices[0].logprobs = SimpleNamespace(
        content=[SimpleNamespace(token="Tokyo", logprob=-0.1, top_logprobs=[])]
    )
    client = counterfix.guard(FakeClient(FakeCompletions(resp)))
    r = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert confidence.mean_probability(r) == pytest.approx(0.905, abs=0.001)
    assert "token confidence" in (confidence.describe(r) or "")
    assert confidence.of(r.diagnosis)["n"] == 1


def test_fix_explain_says_new_for_a_new_failure():
    from counterfix import FailureType, Finding
    from counterfix.core.diagnosis import Diagnosis
    from counterfix.engine import FixAttempt, FixResult

    d = Diagnosis(
        record_id="r1", findings=[Finding(failure_type=FailureType.TRUNCATION, confidence=0.95)]
    )
    attempt = FixAttempt(name="raise_max_tokens", description="", runs=2, passed=0,
                         seen=["F6 Dropped instruction", "F1 Truncation"])  # fmt: skip
    text = FixResult(record_id="r1", diagnosis=d, attempts=[attempt]).explain()
    assert "still: F1 Truncation" in text and "new: F6 Dropped instruction" in text


def test_optional_integrations_say_what_to_install():
    try:
        import litellm  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install litellm"):
            counterfix.litellm_callback()
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install langchain-core"):
            counterfix.langchain_callback()


def test_false_premise_supersedes_ambiguity():
    from counterfix import FailureType, Finding
    from counterfix.core.precedence import resolve

    found = [
        Finding(failure_type=FailureType.FALSE_PREMISE, confidence=1.0),
        Finding(failure_type=FailureType.AMBIGUITY, confidence=0.95),
    ]
    kept, suppressed = resolve(found)
    assert [f.failure_type for f in kept] == [FailureType.FALSE_PREMISE]
    assert [f.failure_type for f in suppressed] == [FailureType.AMBIGUITY]


def test_app_label_and_default_name():
    def outer():
        def app(q: str) -> str:
            return "Paris"

        return app

    result = evaluate([{"id": "a", "question": "q", "answer": "Paris"}], app=outer())
    assert result.model == "app:app" and result.name == "dataset"


def test_untested_suggestions_say_try_and_verified_ones_say_fix():
    from counterfix import FailureType, Finding
    from counterfix.core.diagnosis import Diagnosis, Prescription
    from counterfix.core.types import PrescriptionKind

    def diag(verified: bool) -> str:
        p = Prescription(kind=PrescriptionKind.PARAM_CHANGE, description="Raise max_tokens",
                         patch={"max_tokens": 160}, verified=verified)  # fmt: skip
        f = Finding(failure_type=FailureType.TRUNCATION, confidence=0.95, prescription=p)
        return Diagnosis(record_id="r", findings=[f]).explain()

    assert "  try: Raise max_tokens" in diag(False) and "fix" not in diag(False).lower()
    assert "  fix (verified): Raise max_tokens" in diag(True)


def test_rerun_caller_takes_every_documented_spec() -> None:
    import pytest

    from counterfix.cli import _rerun_caller

    caller = _rerun_caller("my-model@http://127.0.0.1:9/v1")  # no call is made here
    assert "my-model" in caller.name
    with pytest.raises(SystemExit) as stop:
        _rerun_caller("nope:model")
    assert stop.value.code == 2
