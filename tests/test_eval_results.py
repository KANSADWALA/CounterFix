"""Reading DeepEval, Ragas and Promptfoo results. The files in tests/data/eval_results
were written by the tools themselves (DeepEval 4.3.4, Ragas 0.4.3, Promptfoo 0.125.1)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from counterfix import CallRecord, Message
from counterfix.cli import main
from counterfix.integrations.eval_results import _provider, detect_tool, read_results
from counterfix.judges.tasks import CONTEXT_KEY, REFERENCE_KEY

DATA = Path(__file__).parent / "data" / "eval_results"


class Scripted:
    name = "fake:m"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        self.calls += 1
        msgs = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
        if (params.get("max_tokens") or 0) > 64:
            text, finish = "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.", "stop"
        else:
            text, finish = "Mercury\nVenus", "length"
        return CallRecord(
            model="m", messages=msgs, params=params, response_text=text, finish_reason=finish
        )


# ------------------------------------------------------------------------------ Promptfoo


def test_promptfoo_keeps_the_call_as_it_was_sent() -> None:
    cases = read_results(DATA / "promptfoo.json")
    assert [c.tool for c in cases] == ["promptfoo"] * 4
    assert [c.failed for c in cases] == [True, True, False, False]
    first = cases[0]
    assert [m.role for m in first.record.messages] == ["system", "user"]
    assert first.record.messages[1].text == "List the planets"
    assert first.record.params == {"temperature": 0.2, "max_tokens": 64}
    assert first.record.finish_reason == "length"
    assert first.record.model == "gpt-4o-mini" and first.record.provider == "openai"
    assert first.rerun == "gpt-4o-mini@http://127.0.0.1:8765/v1"  # its own apiBaseUrl
    assert "neptune" in first.reason and "names two planets" in first.reason


def test_promptfoo_model_graded_assertions_are_not_faked() -> None:
    first = read_results(DATA / "promptfoo.json")[0]
    assert first.check is None  # llm-rubric needs Promptfoo's grader
    assert any("llm-rubric" in n for n in first.notes)


def test_promptfoo_deterministic_assertions_are_rechecked() -> None:
    cases = read_results(DATA / "promptfoo_custom.json")
    failed, passed = cases[0], cases[2]
    assert failed.check is not None and passed.check is not None
    assert failed.check(failed.record) == (False, "contains Neptune failed")
    assert passed.check(passed.record) == (True, "")  # rendered the same way it ran
    fixed = failed.record.model_copy(update={"response_text": "... Neptune."})
    assert failed.check(fixed)[0]
    assert failed.rerun is None and any("--caller" in n for n in failed.notes)
    assert [m.role for m in cases[1].record.messages] == ["user"]  # a plain-text prompt


def test_promptfoo_assertion_kinds(tmp_path: Path) -> None:
    data = json.loads((DATA / "promptfoo_custom.json").read_text(encoding="utf-8"))
    row = data["results"]["results"][0]
    row["vars"]["word"] = "venus"

    def check_for(*assertions: dict[str, Any]) -> Any:
        row["gradingResult"]["componentResults"] = [
            {"pass": False, "assertion": a} for a in assertions
        ]
        path = tmp_path / "r.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return read_results(path)[0].check

    out = row["response"]["output"]  # "Mercury\nVenus"
    record = CallRecord(response_text=out)
    assert check_for({"type": "icontains", "value": "{{word}}"})(record)[0]
    assert check_for({"type": "not-contains", "value": "Pluto"})(record)[0]
    assert not check_for({"type": "not-icontains", "value": "MERCURY"})(record)[0]
    assert check_for({"type": "contains-all", "value": ["Mercury", "Venus"]})(record)[0]
    assert not check_for({"type": "contains-any", "value": ["Mars", "Pluto"]})(record)[0]
    assert check_for({"type": "regex", "value": "^Mercury"})(record)[0]
    assert check_for({"type": "starts-with", "value": "Merc"})(record)[0]
    assert not check_for({"type": "is-json"})(record)[0]
    assert check_for({"type": "equals", "value": {"a": 1}})(CallRecord(response_text='{"a": 1}'))[0]
    assert check_for({"type": "javascript", "value": "output.length > 3"}) is None
    assert check_for({"type": "contains", "value": "{{ word | upper }}"}) is None


@pytest.mark.parametrize(
    ("pid", "config", "spec"),
    [
        ("openai:chat:gpt-4o-mini", {}, "openai:gpt-4o-mini"),
        ("openai:gpt-4o-mini", {}, "openai:gpt-4o-mini"),
        ("groq:llama-3.1-8b-instant", {}, "groq:llama-3.1-8b-instant"),
        ("ollama:chat:qwen2.5:3b", {}, "ollama:qwen2.5:3b"),
        ("ollama:qwen2.5:3b", {}, "ollama:qwen2.5:3b"),
        ("anthropic:messages:claude-x", {}, "anthropic:claude-x"),
        ("google:gemini-3.5-flash-lite", {}, "gemini:gemini-3.5-flash-lite"),
        ("openrouter:meta/llama", {}, "openrouter:meta/llama"),
        ("openai:chat:m", {"apiBaseUrl": "http://h:8000/v1"}, "m@http://h:8000/v1"),
        ("azure:chat:my-deployment", {}, None),
        ("file://provider.js", {}, None),
        ("python:provider.py", {}, None),
    ],
)
def test_promptfoo_provider_ids(pid: str, config: dict[str, Any], spec: str | None) -> None:
    assert _provider(pid, config)[2] == spec


# ------------------------------------------------------------------------------- DeepEval


def test_deepeval_results_folder_file() -> None:
    cases = read_results(DATA / "deepeval_test_run.json")
    assert [c.tool for c in cases] == ["deepeval", "deepeval"]
    bad, good = cases
    assert bad.failed and not good.failed and good.name == "ok-case"
    assert bad.reason == "Has Neptune 0.00 (threshold 1.0): no Neptune"
    assert bad.record.messages == [Message(role="user", content="List the planets")]
    assert bad.record.response_text == "Mercury\nVenus" and bad.record.finish_reason is None
    assert bad.record.metadata[CONTEXT_KEY] and bad.record.metadata[REFERENCE_KEY] == "All eight"
    assert bad.rerun is None and bad.check is None
    assert any("--system" in n for n in bad.notes)


def test_deepeval_latest_run_and_hyperparameters() -> None:
    for name in ("deepeval_latest_test_run.json", "deepeval_hyperparameters.json"):
        (case,) = read_results(DATA / name)
        assert case.failed and case.record.model == "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------------- Ragas


@pytest.mark.parametrize("name", ["ragas.csv", "ragas.json"])
def test_ragas_needs_and_uses_thresholds(name: str) -> None:
    with pytest.raises(ValueError, match="exact_match, string_present"):
        read_results(DATA / name)
    with pytest.raises(ValueError, match="no column faithfulness"):
        read_results(DATA / name, thresholds={"faithfulness": 0.8})
    good, bad = read_results(DATA / name, thresholds={"exact_match": 1})
    assert not good.failed and bad.failed and bad.reason == "exact_match 0.00 (>= 1.0 needed)"
    assert bad.record.metadata[CONTEXT_KEY] == "[1] There are eight planets."  # cell parsed
    assert bad.record.metadata[REFERENCE_KEY] == "Neptune"
    assert bad.record.response_text == "Mercury\nVenus"  # a multi-line cell
    # lower is better
    good, bad = read_results(DATA / name, thresholds={"exact_match": "<=0.5"})
    assert good.failed and not bad.failed


def test_ragas_csv_with_windows_line_endings(tmp_path: Path) -> None:
    text = (DATA / "ragas.csv").read_text(encoding="utf-8").replace("\n", "\r\n")
    path = tmp_path / "r.csv"
    path.write_bytes(text.encode("utf-8"))
    _, bad = read_results(path, thresholds={"exact_match": 1})
    assert bad.failed and bad.record.response_text == "Mercury\nVenus"


def test_ragas_missing_score_is_not_a_failure(tmp_path: Path) -> None:
    rows = [{"user_input": "q", "response": "a", "faithfulness": math.nan}]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rows).replace("NaN", "null"), encoding="utf-8")
    (case,) = read_results(path, thresholds={"faithfulness": 0.8})
    assert not case.failed and any("not scored" in n for n in case.notes)


def test_a_counterfix_log_is_not_mistaken_for_results(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(json.dumps({"record": CallRecord(model="m").model_dump()}) + "\n")
    assert detect_tool(log) is None
    assert detect_tool(DATA / "ragas.csv") == "ragas"


# ------------------------------------------------------------------------------------ CLI


def test_cli_diagnose_detects_the_tool(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["diagnose", str(DATA / "promptfoo.json")]) == 0
    out = capsys.readouterr().out
    assert "as promptfoo results" in out and "F1 Truncation" in out
    assert "4 test case(s), 2 failed in the tool, 2 with a cause counterfix found." in out


def test_cli_diagnose_names_the_judge_plainly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from counterfix.judges.base import JudgeTask, Verdict

    class Judge:
        name = "fake:judge"

        def ask(self, task: JudgeTask) -> Verdict:  # the first label of every task
            return Verdict(task=task.name, label=task.labels[0], confidence=0.9, reason="r")

    import counterfix.cli.results as results

    monkeypatch.setattr(results, "_cli_judge", lambda choice: Judge())
    assert main(["diagnose", str(DATA / "deepeval_test_run.json"), "--judge"]) == 0
    out = capsys.readouterr().out
    assert "judged by fake:judge\n" in out and "passed to diagnose()" not in out


def test_cli_fix_reruns_the_promptfoo_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # keep only the icontains assertion, so the tool's own check decides each re-run
    data = json.loads((DATA / "promptfoo.json").read_text(encoding="utf-8"))
    rows = data["results"]["results"]
    for row in rows:
        parts = row["gradingResult"]["componentResults"]
        row["gradingResult"]["componentResults"] = [
            p for p in parts if p["assertion"]["type"] != "llm-rubric"
        ]
    data["results"]["results"] = rows[:1]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    import counterfix.cli.results as results

    seen: list[str] = []
    monkeypatch.setattr(results, "_rerun_caller", lambda spec: seen.append(spec) or Scripted())
    out = tmp_path / "fixes.jsonl"
    assert main(["fix", str(path), "--from", "promptfoo", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert seen == ["gpt-4o-mini@http://127.0.0.1:8765/v1"]
    assert "pass test: your check (the Promptfoo assertions)" in text
    assert "1 failed test case(s), 1 with a verified fix." in text
    (row,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert row["fix"]["name"] == "raise_max_tokens" and row["case"] == "test 0, prompt 0"


def test_cli_fix_without_a_model_says_what_to_add(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fix", str(DATA / "deepeval_test_run.json")]) == 2
    assert "Add --caller provider:model" in capsys.readouterr().out


def test_cli_fix_adds_the_system_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import counterfix.cli.results as results

    sent: list[Any] = []

    class Recorder(Scripted):
        def complete(self, messages: Any, **params: Any) -> CallRecord:
            sent.append(messages)
            return super().complete(messages, **params)

    monkeypatch.setattr(results, "_rerun_caller", lambda spec: Recorder())
    check = DATA.parent.parent / "test_fix_check.py"  # any file with a function will do
    code = main(
        [
            "fix",
            str(DATA / "deepeval_test_run.json"),
            "--caller",
            "fake:m",
            "--system",
            "Be brief.",
            "--check",
            f"{check}:has_neptune",
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    # no cause found by the live checks: nothing is re-run, and it says so
    assert sent == [] and "found no cause it can target" in text


def test_cli_system_file_typo_is_caught(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["fix", str(DATA / "deepeval_test_run.json"), "--caller", "fake:m"]
    assert main([*args, "--system", "prompts/sytem.txt"]) == 2
    assert "no such file" in capsys.readouterr().out


def test_cli_ragas_without_thresholds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["diagnose", str(DATA / "ragas.csv")]) == 2
    assert "--threshold exact_match=0.8" in capsys.readouterr().out
    assert main(["diagnose", str(DATA / "ragas.csv"), "--threshold", "exact_match=1"]) == 0
