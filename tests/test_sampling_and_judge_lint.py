from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from counterfix import CallRecord, Config, FailureType, Message, SkipCheck, lint
from counterfix.cli import main
from counterfix.detectors.sampling import SamplingCheck, same_meaning_heuristic
from counterfix.presets import PRESETS, load_env_file, make_client, parse_spec
from test_judges import FixedJudge


class CyclingCaller:
    """Returns answers from a cycle, recording each call's params."""

    def __init__(self, answers: Sequence[str]):
        self.answers: Iterator[str] = itertools.cycle(answers)
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "fake:target"

    def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
        self.calls.append(params)
        return CallRecord(response_text=next(self.answers), finish_reason="stop")


def rec(answer: str, **params: Any) -> CallRecord:
    return CallRecord(
        messages=[Message(role="user", content="When was the bank founded?")],
        response_text=answer,
        params=params,
    )


# --- sampling ------------------------------------------------------------------------


def test_heuristic_equivalence() -> None:
    assert same_meaning_heuristic("It was founded in 1921.", "Founded 1921")
    assert not same_meaning_heuristic("It was founded in 1921.", "In 1934, I believe.")
    assert same_meaning_heuristic("Tbilisi is the capital", "the capital is Tbilisi")
    assert not same_meaning_heuristic("Tbilisi", "Atlanta")


def test_consistent_answers_pass_and_params_are_reused() -> None:
    caller = CyclingCaller(["Founded in 1921."])
    check = SamplingCheck(caller, n=4)
    assert check.check(rec("It was founded in 1921.", max_tokens=50, stream=False), Config()) == []
    assert len(caller.calls) == 4
    assert caller.calls[0] == {"temperature": 1.0, "max_tokens": 50}


def test_disagreeing_answers_flag_f4() -> None:
    caller = CyclingCaller(["In 1934.", "In 1902.", "In 1955.", "1921 I think"])
    [f] = SamplingCheck(caller, n=4).check(rec("It was founded in 1921."), Config())
    assert f.failure_type is FailureType.HALLUCINATION and f.inferred
    assert f.evidence[0].data["agreement"] == 0.4 and f.evidence[0].data["n_samples"] == 5
    assert 0 < f.evidence[0].data["entropy"] <= 1


def test_sampling_skips_long_or_empty_answers_and_errors() -> None:
    caller = CyclingCaller(["x"])
    with pytest.raises(SkipCheck):
        SamplingCheck(caller).check(rec("word " * 80), Config())
    with pytest.raises(SkipCheck):
        SamplingCheck(caller).check(rec(""), Config())

    class Broken(CyclingCaller):
        def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
            raise RuntimeError("429")

    with pytest.raises(SkipCheck):
        SamplingCheck(Broken(["x"])).check(rec("In 1921."), Config())


def test_sampling_can_use_a_judge_for_meaning() -> None:
    caller = CyclingCaller(["The bank opened its doors in nineteen twenty-one."])
    judge = FixedJudge("same")
    assert SamplingCheck(caller, n=3, judge=judge).check(rec("In 1921."), Config()) == []
    assert judge.asked and judge.asked[0].name == "same_meaning"


# --- judge lint rules ----------------------------------------------------------------


def test_judge_rules_run_only_with_a_judge() -> None:
    prompt = "Be extremely brief. Explain every detail of the French Revolution in full."
    assert [i.rule for i in lint(prompt)] == []
    issues = lint(prompt, judge=FixedJudge("conflict", conflicts=["brief vs full detail"]))
    assert [i.rule for i in issues] == ["L2"]  # the fixed judge answers "conflict" to every task
    assert "brief vs full detail" in issues[0].message


def test_l5_needs_a_system_message() -> None:
    judge = FixedJudge("contradiction", contradiction="system says English, user asks Hindi")
    only_user = lint("Reply in Hindi.", rules=["L5"], judge=judge)
    assert only_user == [] and judge.asked == []
    messages = [
        {"role": "system", "content": "Always reply in English."},
        {"role": "user", "content": "Reply in Hindi."},
    ]
    [issue] = lint(messages, rules=["L5"], judge=judge)
    assert "English" in issue.message


def test_l6_underspecified() -> None:
    [issue] = lint(
        "Who won the final last year?",
        rules=["L6"],
        judge=FixedJudge("underspecified", missing=["which sport and year"]),
    )
    assert issue.failure_types == [FailureType.AMBIGUITY] and "which sport" in issue.message


def test_judge_rule_without_judge_errors() -> None:
    with pytest.raises(ValueError):
        lint("x", rules=["L2"])


def test_cli_judge_rules_need_a_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "p.txt"
    path.write_text("Who won?", encoding="utf-8")
    assert main(["lint", str(path), "--rules", "L6"]) == 2
    assert "--judge" in capsys.readouterr().out
    assert main(["lint", str(path), "--judge", "nosuch:model"]) == 2


# --- presets -------------------------------------------------------------------------


def test_parse_spec_and_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_spec("groq:openai/gpt-oss-20b") == ("groq", "openai/gpt-oss-20b")
    assert parse_spec("ollama") == ("ollama", PRESETS["ollama"][2])
    with pytest.raises(ValueError):
        parse_spec("nosuch:model")
    env = tmp_path / ".env"
    env.write_text("CF_PRESET_TEST=yes\n", encoding="utf-8")
    monkeypatch.delenv("CF_PRESET_TEST", raising=False)
    load_env_file(env)
    import os

    assert os.environ["CF_PRESET_TEST"] == "yes"
    monkeypatch.delenv("CF_PRESET_TEST")


def test_make_client_needs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_client("groq")
    with pytest.raises(ValueError):
        make_client("nosuch")
