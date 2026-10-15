from __future__ import annotations

import sys
from pathlib import Path

from conftest import FakeClient, FakeCompletions, make_response
from counterfix import FailureType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import demo_guard


def test_demo_runs_and_flags_cut_off_answers(capsys: object) -> None:
    client = FakeClient(FakeCompletions(response=make_response("A long essay", "length")))
    responses = demo_guard.run(client, "fake-model")
    assert all(r.diagnosis.top.failure_type is FailureType.TRUNCATION for r in responses)


def test_demo_reports_clean_calls() -> None:
    responses = demo_guard.run(FakeClient(), "fake-model")
    assert all(r.diagnosis.ok for r in responses)


def test_env_file_loader(tmp_path: Path, monkeypatch: object) -> None:
    import os

    env = tmp_path / ".env"
    env.write_text(
        '# comment\nCF_TEST_A="from-file"\nCF_TEST_B=\nCF_TEST_C=kept\n', encoding="utf-8"
    )
    os.environ.pop("CF_TEST_A", None)
    os.environ.pop("CF_TEST_B", None)
    os.environ["CF_TEST_C"] = "already-set"
    try:
        demo_guard.load_env_file(env)
        assert os.environ["CF_TEST_A"] == "from-file"
        assert "CF_TEST_B" not in os.environ
        assert os.environ["CF_TEST_C"] == "already-set"
        demo_guard.load_env_file(tmp_path / "missing.env")  # no error
    finally:
        for key in ("CF_TEST_A", "CF_TEST_C"):
            os.environ.pop(key, None)


def test_list_models_skips_non_chat_models() -> None:
    from types import SimpleNamespace

    client = FakeClient()
    client.models = SimpleNamespace(
        list=lambda: [
            SimpleNamespace(id=i)
            for i in ("whisper-large-v3", "llama-x", "a-model", "models/gemini-x", "models/veo-3")
        ]
    )
    assert demo_guard.list_models(client) == ["a-model", "gemini-x", "llama-x"]


def test_model_not_found_detection() -> None:
    class NotFoundError(Exception):
        pass

    assert demo_guard.is_model_not_found(NotFoundError())
    error = RuntimeError()
    error.status_code = 404  # type: ignore[attr-defined]
    assert demo_guard.is_model_not_found(error)
    assert not demo_guard.is_model_not_found(RuntimeError())


def test_demo_explains_reasoning_budget_truncation() -> None:
    response = make_response(content="", finish_reason="length")
    response.choices[0].message.reasoning = "thinking..."
    client = FakeClient(FakeCompletions(response=response))
    top = demo_guard.run(client, "reasoning-model")[0].diagnosis.top
    assert "reasoning" in top.evidence[0].detail
    assert top.prescription.patch["reasoning_effort"] == "low"
