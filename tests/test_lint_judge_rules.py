"""Prompt lint rules that need a judge (L2, L5, L6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from counterfix import FailureType, lint
from counterfix.cli import main
from test_judges import FixedJudge


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
