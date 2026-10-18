"""Bring your own key: how counterfix picks a judge, and diagnose()."""

from __future__ import annotations

import json
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import CallRecord, FailureType, Message, NoJudgeError, SelfJudgeWarning
from counterfix.judges import auto
from counterfix.judges.base import JudgeTask, Verdict

KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", auto.ENV_VAR)


class Stub:
    def __init__(self, name: str):
        self.name = name

    def ask(self, task: JudgeTask) -> Verdict:
        return Verdict(task=task.name, label="no_claims", confidence=0.9, judge=self.name)


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No keys, no Ollama, nothing configured; build_judge returns a named stub."""
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(auto, "ollama_models", lambda timeout=0.5: [])
    monkeypatch.setattr(auto, "_configured", None)
    real_build = auto.build_judge

    def fake_build(choice: Any) -> Any:
        if isinstance(choice, str) and "," not in choice and choice != "jev":
            return Stub(choice)
        return real_build(choice)

    monkeypatch.setattr(auto, "build_judge", fake_build)
    return monkeypatch


def test_family() -> None:
    assert auto.family("openai/gpt-oss-120b") == "gpt"
    assert auto.family("qwen/qwen3.8-27b") == "qwen"
    assert auto.family("gemma-4-31b-it") == "gemma"
    assert auto.family("my-model") == "my-model"


def test_order_passed_then_configured_then_env_then_keys(clean: pytest.MonkeyPatch) -> None:
    clean.setenv("GROQ_API_KEY", "x")
    assert auto.resolve_judge()[0].name == "groq:qwen/qwen3.8-27b"
    clean.setenv(auto.ENV_VAR, "ollama:llama3.2")
    judge, source = auto.resolve_judge()
    assert judge.name == "ollama:llama3.2" and source == f"{auto.ENV_VAR} is set"
    counterfix.configure(judge="gemini:gemini-3.5-flash-lite")
    judge, source = auto.resolve_judge()
    assert judge.name == "gemini:gemini-3.5-flash-lite" and source == "counterfix.configure()"
    judge, source = auto.resolve_judge("openai:gpt-4o-mini")
    assert judge.name == "openai:gpt-4o-mini" and source == "passed to diagnose()"
    counterfix.configure(judge=None)
    assert auto.resolve_judge()[0].name == "ollama:llama3.2"


def test_prefers_another_family_than_the_checked_model(clean: pytest.MonkeyPatch) -> None:
    clean.setenv("GROQ_API_KEY", "x")
    clean.setenv("GEMINI_API_KEY", "x")
    assert auto.detect_judges()[0][0].startswith("groq:")
    # checking a qwen model: the qwen judge goes last
    assert auto.detect_judges(avoid_model="qwen2.5:3b")[0][0].startswith("gemini:")
    clean.setattr(auto, "ollama_models", lambda timeout=0.5: ["llama3.2:3b"])
    assert ("ollama:llama3.2:3b", "Ollama is running locally") in auto.detect_judges()


def test_no_judge_explains_how_to_get_one(clean: pytest.MonkeyPatch) -> None:
    with pytest.raises(NoJudgeError, match="GROQ_API_KEY"):
        auto.resolve_judge()


def test_configure_rejects_nonsense() -> None:
    with pytest.raises(TypeError):
        counterfix.configure(judge=42)  # type: ignore[arg-type]


def test_lists_and_commas_make_a_panel(clean: pytest.MonkeyPatch) -> None:
    clean.setattr(auto, "build_judge", auto.build_judge)  # keep the panel logic real
    from counterfix.judges import JudgePanel

    clean.setattr("counterfix.presets.make_caller", lambda spec, timeout=None: _NamedCaller(spec))
    panel = auto.build_judge("groq:a, gemini:b")
    assert isinstance(panel, JudgePanel) and len(panel.judges) == 2


class _NamedCaller:
    def __init__(self, name: str):
        self.name = name

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        raise AssertionError("not called")


def judge_reply(params: dict[str, Any]) -> Any:
    return make_response(json.dumps({"label": "no_claims", "confidence": 0.9, "reason": "r"}))


def test_diagnose_a_guarded_response_falls_back_to_self_judge(clean: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions(make_response("Paris is the capital of France.", model="m"))
    client = counterfix.guard(FakeClient(completions))
    response = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "Capital of France?"}]
    )
    completions.response = make_response(
        json.dumps({"label": "no_claims", "confidence": 0.9, "reason": "r"})
    )
    with pytest.warns(SelfJudgeWarning, match="judging its own answer"):
        report = counterfix.diagnose(response)
    assert report.ok and report.judge is not None and "checked model itself" in report.judge
    assert "judge_hallucination" in report.checked
    judge_calls = completions.calls[1:]
    assert judge_calls and all(c["model"] == "m" for c in judge_calls)


def test_diagnose_a_record_with_an_explicit_judge(clean: pytest.MonkeyPatch) -> None:
    record = CallRecord(
        model="m",
        messages=[Message(role="user", content="Reply in exactly 2 sentences.")],
        response_text="One. Two. Three.",
        finish_reason="stop",
    )
    report = counterfix.diagnose(record, judge=Stub("my-judge"))
    assert [f.failure_type for f in report.findings] == [FailureType.DROPPED_INSTRUCTION]
    assert report.judge == "my-judge (passed to diagnose())"
    with pytest.raises(TypeError):
        counterfix.diagnose(object())


def test_setup_command_never_prints_key_values(
    clean: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    from counterfix.cli import main

    clean.chdir(tmp_path)
    clean.setenv("GROQ_API_KEY", "gsk_secret_value")
    assert main(["setup"]) == 0
    out = capsys.readouterr().out
    assert "gsk_secret_value" not in out
    assert "GROQ_API_KEY" in out and "will use groq:qwen/qwen3.8-27b" in out


def test_diagnose_without_a_judge_runs_live_checks_only(clean: pytest.MonkeyPatch) -> None:
    record = CallRecord(
        model="m",
        messages=[Message(role="user", content="Write an essay.")],
        response_text="Essays are",
        finish_reason="length",
        params={"max_tokens": 4},
    )
    report = counterfix.diagnose(record, judge=False)
    assert [f.failure_type for f in report.findings] == [FailureType.TRUNCATION]
    assert report.judge is None and not any(c.startswith("judge_") for c in report.checked)
    text = report.explain()
    assert text.startswith("F1 Truncation") and "why:" in text and "fix:" in text
