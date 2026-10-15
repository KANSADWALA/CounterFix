from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfix import FailureType, lint
from counterfix.cli import main


def rules(prompt: object, **kw: object) -> list[str]:
    return [i.rule for i in lint(prompt, **kw)]  # type: ignore[arg-type]


# --- L1 buried instruction ----------------------------------------------------------

FILLER = "Our brand voice is warm and practical and we sell to students. " * 12


def test_l1_flags_constraint_buried_in_long_prompt() -> None:
    prompt = FILLER + "You must write exactly three sentences. " + FILLER + "Describe the bottle."
    [issue] = lint(prompt, rules=["L1"])
    assert issue.failure_types == [FailureType.DROPPED_INSTRUCTION]
    assert "exactly three sentences" in (issue.snippet or "")


def test_l1_quiet_when_constraint_repeated_at_end_or_prompt_short() -> None:
    repeated = (
        FILLER
        + "You must write exactly three sentences. "
        + FILLER
        + "Describe the bottle. You must write exactly three sentences."
    )
    assert rules(repeated, rules=["L1"]) == []
    assert rules("You must write exactly three sentences about bottles.", rules=["L1"]) == []


# --- L3 no abstain option -----------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Why did Marie Curie win three Nobel Prizes?",
        "In which year did India host the 2016 Summer Olympics?",
        "What is the return window in the Nimbus 2026 refund policy?",
    ],
)
def test_l3_flags_factual_questions(question: str) -> None:
    [issue] = lint(question, rules=["L3"])
    assert FailureType.HALLUCINATION in issue.failure_types


@pytest.mark.parametrize(
    "prompt",
    [
        "What is 17 to the power of 5?",
        "How many times does the letter 'r' appear in 'strawberry'?",
        "What is 48,763 multiplied by 9,284?",
        "Write a poem about rain.",
        "Explain how transformers work.",
    ],
)
def test_l3_skips_computation_and_non_questions(prompt: str) -> None:
    assert rules(prompt, rules=["L3"]) == []


def test_l3_quiet_when_abstaining_is_allowed() -> None:
    messages = [
        {"role": "system", "content": "If you don't know the answer, say so."},
        {"role": "user", "content": "Who won the 1987 Ranji Trophy?"},
    ]
    assert rules(messages, rules=["L3"]) == []


# --- L4 vague format ----------------------------------------------------------------


def test_l4_flags_format_described_in_words_only() -> None:
    [issue] = lint("Return ONLY JSON with keys name and age for: Priya, 29.", rules=["L4"])
    assert issue.failure_types == [FailureType.FORMAT_BREAK] and "JSON" in issue.message


def test_l4_quiet_with_example_or_json_mode() -> None:
    assert rules('Answer as JSON like {"name": "...", "age": 0}.', rules=["L4"]) == []
    assert (
        rules(
            "Return JSON with keys name and age.",
            params={"response_format": {"type": "json_object"}},
            rules=["L4"],
        )
        == []
    )
    assert rules("What is JSON?", rules=["L4"]) == []


def test_l4_other_formats() -> None:
    [issue] = lint("Output the results as YAML please.", rules=["L4"])
    assert "YAML" in issue.message


# --- L7 undelimited input -----------------------------------------------------------

DOC = "The quarterly results were strong across all regions and teams. " * 5


def test_l7_flags_pasted_text_without_delimiters() -> None:
    [issue] = lint(f"Summarise this.\nDocument: {DOC}", rules=["L7"])
    assert issue.severity == "warning" and "injection" in issue.message


def test_l7_quiet_with_delimiters_or_short_text() -> None:
    assert rules(f"Summarise this.\n<document>\n{DOC}\n</document>", rules=["L7"]) == []
    assert rules(f'Summarise this.\nDocument: """{DOC}"""', rules=["L7"]) == []
    assert rules("Summarise this.\nDocument: a short note.", rules=["L7"]) == []


def test_clean_prompt_has_no_issues() -> None:
    assert lint("Write a haiku about the monsoon.") == []


def test_unknown_rule_raises() -> None:
    with pytest.raises(KeyError):
        lint("hi", rules=["L99"])


# --- CLI ----------------------------------------------------------------------------


def test_cli_lints_text_json_and_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "p.txt").write_text("Return ONLY JSON with keys a and b.", encoding="utf-8")
    (tmp_path / "p.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Why is the sky green?"}]}),
        encoding="utf-8",
    )
    (tmp_path / "p.jsonl").write_text(
        json.dumps({"id": "x1", "messages": [{"role": "user", "content": "Write a haiku."}]})
        + "\n",
        encoding="utf-8",
    )
    code = main(
        ["lint", str(tmp_path / "p.txt"), str(tmp_path / "p.json"), str(tmp_path / "p.jsonl")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "L4" in out and "L3" in out and "p.jsonl:x1: ok" in out
    assert "3 prompt(s) checked, 2 with issues, 1 warning(s)." in out


def test_cli_fail_on_warning_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "p.txt"
    path.write_text("Return ONLY JSON with keys a and b.", encoding="utf-8")
    assert main(["lint", str(path), "--fail-on-warning"]) == 1
    assert main(["lint", str(path), "--rules", "L99"]) == 2
    assert main(["lint", str(tmp_path / "missing.txt")]) == 2


def test_cli_runs_on_the_seed_set(capsys: pytest.CaptureFixture[str]) -> None:
    seed = Path(__file__).resolve().parents[1] / "bench" / "seed_failures.jsonl"
    assert main(["lint", str(seed), "--quiet"]) == 0
    assert "54 prompt(s) checked" in capsys.readouterr().out
