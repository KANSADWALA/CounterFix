from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from bench import eval_judges, label

from counterfix import CallRecord
from counterfix.core.caller import RateLimitedCaller


def test_parse_answer() -> None:
    assert label.parse_answer("") == "" and label.parse_answer("?") == "?"
    assert label.parse_answer("n") == []
    assert label.parse_answer(" f4, F5 ") == ["F4", "F5"]
    assert label.parse_answer("F5 F4 F4") == ["F4", "F5"]
    assert label.parse_answer("q") == "q" and label.parse_answer("u") == "u"
    assert label.parse_answer("F10") is None and label.parse_answer("maybe") is None


def test_labels_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    data = {
        "r:F4-001": {
            "key": "r:F4-001",
            "run": "r",
            "id": "F4-001",
            "model": "m",
            "labels": ["F4"],
            "source": "human",
        }
    }
    label.save_labels(data, path)
    assert label.load_labels(path) == data


def test_human_labels_win_when_files_are_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(label, "LABEL_DIR", tmp_path)
    base = {"key": "r:F5-001", "run": "r", "id": "F5-001", "model": "m"}

    def write(name: str, labels: list[str], source: str) -> None:
        line = json.dumps({**base, "labels": labels, "source": source})
        (tmp_path / name).write_text(line + "\n", encoding="utf-8")

    write("a_claude.jsonl", ["F5"], "claude-reference")
    write("labels.jsonl", [], "human")
    write("z_other.jsonl", ["F4"], "claude-reference")
    merged = label.load_labels()
    assert merged["r:F5-001"]["labels"] == [] and merged["r:F5-001"]["source"] == "human"
    extra = {**base, "key": "x", "labels": [], "source": "claude-reference"}
    label.save_labels({**merged, "x": extra}, tmp_path / "own.jsonl")
    assert "claude-reference" not in (tmp_path / "own.jsonl").read_text(encoding="utf-8")


def test_score() -> None:
    truth = {"a": {"F4"}, "b": set(), "c": {"F4", "F5"}, "d": set()}
    predicted = {"a": {"F4"}, "b": {"F4"}, "c": set(), "d": set()}
    s = eval_judges.score(predicted, truth)
    assert s["F4"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "positives": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert s["F5"]["recall"] == 0.0 and s["F5"]["precision"] is None
    assert s["F9"]["positives"] == 0 and s["F9"]["recall"] is None


def test_labelled_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results"
    (results / "phase2").mkdir(parents=True)
    run = {
        "provider": "x",
        "model": "m",
        "rows": [{"id": "F2-001", "answer_full": '{"name": "A"}', "finish_reason": "stop"}],
    }
    (results / "phase2" / "run1.json").write_text(json.dumps(run), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    label.save_labels(
        {
            "run1:F2-001": {
                "key": "run1:F2-001",
                "run": "run1",
                "id": "F2-001",
                "model": "m",
                "labels": ["F2"],
            }
        },
        labels,
    )
    monkeypatch.setattr(eval_judges, "RESULTS", results)
    monkeypatch.setattr(eval_judges, "load_labels", lambda: label.load_labels(labels))
    [(record, codes, key)] = eval_judges.labelled_records(with_reference=True)
    assert key == "run1:F2-001" and codes == {"F2"}
    assert record.response_text == '{"name": "A"}' and "expected_schema" in record.metadata
    assert record.prompt_text.startswith("Return ONLY a JSON")


class Flaky:
    def __init__(self, fail_times: int, error: Exception):
        self.fail_times, self.error, self.calls = fail_times, error, 0

    @property
    def name(self) -> str:
        return "flaky"

    def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return CallRecord(response_text="ok")


def test_rate_limited_caller_retries_429_only() -> None:
    caller = RateLimitedCaller(Flaky(2, RuntimeError("Error code: 429")), rpm=0, backoff=0.01)
    assert caller.complete([]).response_text == "ok" and caller.name == "flaky"
    other = RateLimitedCaller(Flaky(1, ValueError("400 bad request")), rpm=0, backoff=0.01)
    with pytest.raises(ValueError):
        other.complete([])
    exhausted = RateLimitedCaller(Flaky(9, RuntimeError("429")), rpm=0, retries=1, backoff=0.01)
    with pytest.raises(RuntimeError):
        exhausted.complete([])


def test_rate_limited_caller_spaces_calls() -> None:
    caller = RateLimitedCaller(Flaky(0, RuntimeError()), rpm=600)  # one call per 0.1 s
    start = time.monotonic()
    for _ in range(3):
        caller.complete([])
    assert time.monotonic() - start >= 0.17  # 2 gaps of 0.1 s, less one Windows clock tick


def test_suggestions_are_objective_only() -> None:
    from bench.seed import load_seed

    seed = {i.id: i for i in load_seed()}
    cut = {"answer_full": "", "finish_reason": "length", "reasoning_chars": 300}
    assert label.suggest(cut, seed["F1-001"])[0] == ["F1"]
    done = {"answer_full": "A full essay.", "finish_reason": "stop"}
    codes, reasons = label.suggest(done, seed["F1-001"])
    assert codes == [] and "passed" in reasons[0]
    bad_json = {"answer_full": 'Sure! {"name": "P"}', "finish_reason": "stop"}
    assert label.suggest(bad_json, seed["F2-001"])[0] == ["F2"]
    good_json = {
        "answer_full": '{"name": "Priya", "age": 29, "skills": ["Python"]}',
        "finish_reason": "stop",
    }
    assert label.suggest(good_json, seed["F2-001"]) == (
        [],
        ["F2 check passed: valid JSON that matches the schema"],
    )
    assert label.suggest(
        {"answer_full": "There are 2 r's.", "finish_reason": "stop"}, seed["F8-001"]
    )[0] == ["F8"]
    assert (
        label.suggest({"answer_full": "Three r's.", "finish_reason": "stop"}, seed["F8-001"])[0]
        == []
    )
    # judgement-based prompts get no suggestion at all
    assert label.suggest(
        {"answer_full": "She won three...", "finish_reason": "stop"}, seed["F5-001"]
    ) == ([], [])


def test_legend_lists_every_code() -> None:
    text = label.legend()
    assert all(code in text for code in [f"F{i}" for i in range(1, 10)] + ["n"])


def test_default_judges_follow_available_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert eval_judges.default_judges() == []
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert eval_judges.default_judges() == [
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemma-4-31b-it",
    ]
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert len(eval_judges.default_judges()) == 3


def test_daily_limit_fails_fast_instead_of_waiting() -> None:
    import pytest

    from counterfix.core.caller import DailyLimitError

    class RateLimitError(Exception):
        pass

    groq = RateLimitError(
        "Error code: 429 - Rate limit reached for model `qwen/qwen3.8-27b` on tokens per day "
        "(TPD): Limit 200000, Used 199934. Please try again in 2m32.9s."
    )
    flaky = Flaky(5, groq)
    start = time.monotonic()
    with pytest.raises(DailyLimitError, match="daily limit"):
        RateLimitedCaller(flaky, rpm=600).complete([])
    assert flaky.calls == 1 and time.monotonic() - start < 1
    minute = RateLimitError("429 Rate limit reached on tokens per minute. Try again in 0.01s")
    assert RateLimitedCaller(Flaky(1, minute), rpm=600).complete([]).response_text == "ok"
