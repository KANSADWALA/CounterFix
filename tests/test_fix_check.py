"""fix(check=...): the developer's own test decides which re-runs pass."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import counterfix
from counterfix import CallRecord, Message
from counterfix.cli import main
from counterfix.evals.metrics.base import Metric
from counterfix.fixes.engine import CheckError


class Scripted:
    """A fake model: ``answer(messages, params)`` decides each reply."""

    name = "fake:m"

    def __init__(self, answer: Callable[[list[Message], dict[str, Any]], tuple[str, str]]):
        self.answer = answer
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        msgs = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
        self.calls.append((msgs, params))
        text, finish = self.answer(msgs, params)
        return CallRecord(
            model="m", messages=msgs, params=params, response_text=text, finish_reason=finish
        )


def cut_off() -> CallRecord:
    return CallRecord(
        model="m",
        messages=[Message(role="user", content="List the planets, one per line.")],
        params={"max_tokens": 20},
        response_text="Mercury\nVenus",
        finish_reason="length",
    )


def has_neptune(record: CallRecord) -> bool:
    return "Neptune" in (record.response_text or "")


def full_list(m: list[Message], p: dict) -> tuple[str, str]:
    return ("Mercury ... Neptune.", "stop") if p["max_tokens"] > 20 else ("Mercury", "length")


def test_a_check_that_passes_on_the_call_means_no_reruns() -> None:
    model = Scripted(full_list)
    record = cut_off().model_copy(update={"response_text": "Mercury ... Neptune."})
    result = counterfix.fix(record, judge=False, caller=model, check=has_neptune)
    assert model.calls == [] and result.calls == 0 and result.fix is None
    assert result.stopped == "your check passes on this call: nothing to fix"
    assert result.check == "has_neptune"


def test_your_check_decides_and_the_cause_picks_the_change() -> None:
    model = Scripted(full_list)
    result = counterfix.fix(cut_off(), judge=False, caller=model, check=has_neptune)
    assert result.fix is not None and result.fix.name == "raise_max_tokens"
    assert "pass test: your check has_neptune" in result.explain()


def test_the_check_is_the_authority_on_each_run() -> None:
    # every re-run is complete (no cut-off), but the user's check never passes
    model = Scripted(lambda m, p: ("Mercury and Venus.", "stop"))
    result = counterfix.fix(cut_off(), judge=False, caller=model, check=has_neptune)
    assert result.fix is None and result.retry is not None and result.retry.passed == 0
    assert all(a.passed == 0 for a in result.attempts)
    assert "your check failed" in result.explain()


def test_a_new_failure_blocks_a_pass_even_when_the_check_passes() -> None:
    # the answer has Neptune, but a re-run that is cut off is a new failure
    record = CallRecord(
        model="m",
        messages=[Message(role="user", content="Return JSON with the planets.")],
        params={"max_tokens": 20},
        response_text="not json",
        finish_reason="stop",
    )
    model = Scripted(lambda m, p: ("Neptune", "length"))
    result = counterfix.fix(record, judge=False, caller=model, check=has_neptune)
    tried = ([result.retry] if result.retry else []) + result.attempts
    assert tried and all(a.passed == 0 for a in tried)


def test_no_cause_found_is_said_honestly_and_nothing_runs() -> None:
    record = CallRecord(
        model="m",
        messages=[Message(role="user", content="Name the outer planets.")],
        response_text="Mars and Venus.",
        finish_reason="stop",
    )
    model = Scripted(full_list)
    result = counterfix.fix(record, judge=False, caller=model, check=has_neptune)
    assert model.calls == [] and result.fix is None
    assert "your check fails, but counterfix found no cause" in result.stopped
    assert "judge" in result.stopped


def test_reason_from_a_pair_is_shown() -> None:
    record = CallRecord(model="m", response_text="Mars.", finish_reason="stop")
    result = counterfix.fix(
        record, judge=False, caller=Scripted(full_list), check=lambda r: (False, "no Neptune")
    )
    assert result.stopped.startswith("your check fails (no Neptune)")


def test_a_check_error_on_the_call_is_raised_plainly() -> None:
    def broken(record: CallRecord) -> bool:
        raise KeyError("contexts")

    with pytest.raises(CheckError, match="raised KeyError") as info:
        counterfix.fix(cut_off(), judge=False, caller=Scripted(full_list), check=broken)
    assert isinstance(info.value.__cause__, KeyError)


def test_a_score_instead_of_a_verdict_is_refused() -> None:
    with pytest.raises(CheckError, match="return True or"):
        counterfix.fix(cut_off(), judge=False, caller=Scripted(full_list), check=lambda r: 0.9)


def test_a_check_error_on_a_rerun_counts_as_a_fail() -> None:
    def picky(record: CallRecord) -> bool:
        if record.finish_reason == "stop":
            raise RuntimeError("judge timed out")
        return False

    result = counterfix.fix(cut_off(), judge=False, caller=Scripted(full_list), check=picky)
    assert result.fix is None and result.unreachable is None
    assert any("your check raised RuntimeError" in s for a in result.attempts for s in a.seen)


class Mentions(Metric):
    name = "mentions_neptune"
    default_threshold = 1.0
    needs_reference = False

    def compute(self, record: CallRecord, reference: object = None) -> tuple[float, str]:
        return float("Neptune" in (record.response_text or "")), ""


def test_a_counterfix_metric_is_accepted() -> None:
    result = counterfix.fix(cut_off(), judge=False, caller=Scripted(full_list), check=Mentions())
    assert result.fix is not None and result.check == "mentions_neptune"


def test_a_metric_without_a_verdict_says_why() -> None:
    class Graded(Mentions):
        default_threshold = None

    with pytest.raises(CheckError, match="gave no pass/fail"):
        counterfix.fix(cut_off(), judge=False, caller=Scripted(full_list), check=Graded())


def test_cli_check_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    log = tmp_path / "calls.jsonl"
    good = cut_off().model_copy(update={"response_text": "Mercury ... Neptune."})
    log.write_text(
        "\n".join(r.model_dump_json() for r in (good, cut_off())) + "\n", encoding="utf-8"
    )
    (tmp_path / "mycheck.py").write_text(
        "def ok(record):\n    return 'Neptune' in (record.response_text or '')\n",
        encoding="utf-8",
    )
    import counterfix.cli.calls as calls

    monkeypatch.setattr(calls, "_rerun_caller", lambda spec: Scripted(full_list))
    out = tmp_path / "fixes.jsonl"
    code = main(
        [
            "fix",
            str(log),
            "--caller",
            "fake:m",
            "--check",
            f"{tmp_path / 'mycheck.py'}:ok",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "1 flagged call(s), 1 with a verified fix." in text
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["fix"]["name"] == "raise_max_tokens"


def test_cli_bad_check_spec(tmp_path: Path, capsys) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(cut_off().model_dump_json() + "\n", encoding="utf-8")
    assert main(["fix", str(log), "--check", "nope"]) == 2
    assert "Cannot load the check" in capsys.readouterr().out
