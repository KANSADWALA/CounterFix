"""The eval runner: score, explain misses by cause, verify fixes, resume, suites, CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import counterfix
from counterfix import CallRecord, Message, Usage
from counterfix.cli import main
from counterfix.core.caller import DailyLimitError
from counterfix.evals import bootstrap_ci, evaluate, make_items
from counterfix.testing import assert_metric

ROWS = [
    {"id": "q1", "question": "Capital of France?", "answer": "Paris"},
    {"id": "q2", "question": "Capital of Japan?", "answer": "Tokyo"},
    {"id": "q3", "question": "Capital of Peru?", "answer": "Lima"},
    {"id": "q4", "question": "Write a long essay on Peru.", "answer": "Lima"},
]


class Scripted:
    """A fake model: right for France and Japan, wrong for Peru, cut off on essays
    unless max_tokens is large."""

    name = "fake:model"

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_after = fail_after

    def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> CallRecord:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise DailyLimitError("daily limit used up")
        self.calls.append(params)
        msgs = [m if isinstance(m, Message) else Message(**m) for m in messages]
        prompt = msgs[-1].text
        finish, text = "stop", "I am not sure."
        if "France" in prompt:
            text = "Paris"
        elif "Japan" in prompt:
            text = "Tokyo."
        elif "Peru?" in prompt:
            text = "Cusco"
        elif "essay" in prompt:
            if (params.get("max_tokens") or 0) >= 1000:
                text = "Peru's capital is Lima"
            else:
                text, finish = "Peru is a country in", "length"
        return CallRecord(
            model=self.name, messages=msgs, params=params, response_text=text,
            finish_reason=finish, latency_ms=100.0 + len(self.calls),
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )  # fmt: skip


def test_make_items_field_rules() -> None:
    items = make_items(
        [{"question": "Q?", "answer": "A", "params": {"temperature": 0}}],
        system="Be brief.",
        params={"max_tokens": 50},
    )
    item = items[0]
    assert [m.role for m in item.messages] == ["system", "user"]
    assert item.reference == "A" and item.params == {"max_tokens": 50, "temperature": 0}
    templated = make_items([{"q": "2+2"}], template="Compute {q}.", reference="q")
    assert templated[0].messages[0].text == "Compute 2+2." and templated[0].reference == "2+2"
    with pytest.raises(ValueError, match="no prompt"):
        make_items([{"answer": "x"}])
    with pytest.raises(ValueError, match="template field"):
        make_items([{"a": 1}], template="{b}")


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    values = [1.0] * 7 + [0.0] * 3
    low, high = bootstrap_ci(values)
    assert low is not None and high is not None and low <= 0.7 <= high
    assert bootstrap_ci(values) == (low, high)
    assert bootstrap_ci([1.0]) == (None, None)


def test_evaluate_scores_groups_misses_and_verifies_a_fix(tmp_path: Path) -> None:
    rows = [dict(r, params={"max_tokens": 20}) if r["id"] == "q4" else r for r in ROWS]
    result = evaluate(rows, model=Scripted(), metrics=["contains"], fix=2, name="capitals")
    assert [i.passed for i in result.items] == [True, True, False, False]
    m = result.metric("contains")
    assert (m.passed, m.failed, m.mean) == (2, 2, 0.5)
    assert result.by_cause == {"F1": 1, "none found": 1}
    q4 = result.items[3]
    assert q4.fix is not None and q4.fix.fix is not None
    assert q4.fix.fix.name == "raise_max_tokens_more"
    assert result.fixes == {"raise_max_tokens_more": 1, "no verified fix": 1}
    text = result.text()
    assert "Misses by cause" in text and "F1 Truncation" in text and "Verified fixes" in text
    assert "<h1>eval capitals</h1>" in result.html()
    assert result.speed.calls == 4
    saved = result.save(tmp_path / "r.json")
    assert json.loads(saved.read_text())["name"] == "capitals"


def test_metric_failure_type_becomes_a_cause() -> None:
    rows = [{"id": "j", "prompt": "Give JSON", "response": '{"a": 1'}]
    result = evaluate(rows, metrics=["json_valid"])
    assert result.by_cause == {"F2": 1}


def test_recorded_answers_need_no_model() -> None:
    rows = [{"prompt": "p", "response": "Paris", "answer": "Paris"}]
    assert evaluate(rows).metric("exact_match").passed == 1
    with pytest.raises(ValueError, match="no model given"):
        evaluate([{"prompt": "p", "answer": "x"}])


def test_resume_reuses_kept_answers_and_stops_cleanly_at_a_daily_limit(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    first = evaluate(ROWS, model=Scripted(fail_after=2), metrics=["contains"], out=out)
    assert len(first.items) == 2 and "daily limit" in first.stopped
    model = Scripted()
    second = evaluate(ROWS, model=model, metrics=["contains"], out=out)
    assert len(second.items) == 4 and len(model.calls) == 2  # q1, q2 came from the file


def test_suite_file_and_assert_metric(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ROWS[:3]) + "\n", encoding="utf-8"
    )
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"name": "caps", "data": "data.jsonl", "metrics": ["contains"]}))
    result = evaluate(suite, model=Scripted())
    assert result.name == "caps" and len(result.items) == 3
    assert assert_metric(result, "contains", ">=", 0.6) == pytest.approx(2 / 3)
    with pytest.raises(AssertionError, match="contains pass_rate"):
        assert_metric(result, "contains", ">=", 0.9)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"data": "x.jsonl", "modle": "typo"}))
    with pytest.raises(ValueError, match="unknown suite keys"):
        evaluate(bad, model=Scripted())


def test_cli_eval_on_recorded_answers_with_a_ci_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "answers.jsonl"
    rows = [
        {"id": "a", "prompt": "2+2?", "response": "The answer is 4.", "answer": "4"},
        {"id": "b", "prompt": "3*3?", "response": "It is 6.", "answer": "9"},
    ]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    html = tmp_path / "r.html"
    code = main(["eval", str(data), "--metric", "number", "--html", str(html)])
    out = capsys.readouterr().out
    assert code == 0 and "number" in out and "1/2" in out and html.exists()
    assert main(["eval", str(data), "--metric", "number", "--fail-under", "number=0.9"]) == 1
    assert "FAILED: number 0.5 < 0.9" in capsys.readouterr().out
    assert main(["eval", str(data), "--metric", "nope"]) == 2


def test_top_level_exports() -> None:
    assert counterfix.evaluate is evaluate
    assert counterfix.metrics.get_metric("f1").name == "f1"


def test_a_missing_judge_fails_before_any_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import counterfix.judges.auto as auto

    def no_judge(*args: Any, **kwargs: Any) -> Any:
        raise auto.NoJudgeError("no judge available")

    monkeypatch.setattr(auto, "resolve_judge", no_judge)
    model = Scripted()
    with pytest.raises(auto.NoJudgeError):
        evaluate(ROWS, model=model, metrics=["contains"], judge=True)
    assert model.calls == []


# --- regressions from the Wave 1 review ------------------------------------------------------


def test_csv_with_quoted_newlines_and_an_excel_bom(tmp_path: Path) -> None:
    from counterfix.evals import read_rows

    path = tmp_path / "data.csv"
    path.write_bytes('﻿id,question,answer\n1,"Line one\nline two?",yes\n'.encode())
    rows = read_rows(path)
    assert rows == [{"id": "1", "question": "Line one\nline two?", "answer": "yes"}]


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="used twice"):
        make_items([{"id": 1, "prompt": "a"}, {"id": 1, "prompt": "b"}])


def test_a_changed_prompt_is_asked_again_not_read_from_the_kept_answers(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    evaluate(ROWS[:2], model=Scripted(), metrics=["contains"], out=out)
    model = Scripted()
    evaluate(ROWS[:2], model=model, metrics=["contains"], out=out, system="Be brief.")
    assert len(model.calls) == 2  # different messages: the kept answers do not apply


def test_failed_calls_count_as_misses_and_unscored_items_are_shown() -> None:
    class Flaky(Scripted):
        def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> Any:
            if "Japan" in str(messages[-1]):
                raise TimeoutError("read timed out")
            return super().complete(messages, **params)

    rows = [*ROWS[:2], {"id": "noref", "question": "Capital of France?"}]
    result = evaluate(rows, model=Flaky(), metrics=["contains"])
    m = result.metric("contains")
    assert (m.passed, m.failed, m.not_scored) == (1, 1, 1)
    assert m.mean == 1.0 and m.pass_rate == 0.5  # mean over scored items; the error is a miss
    assert "(1 not scored)" in result.text()
    assert result.by_cause == {"INFRA": 1}


def test_suite_max_runs_caps_the_fix_search(tmp_path: Path) -> None:
    (tmp_path / "d.jsonl").write_text(json.dumps(dict(ROWS[3], params={"max_tokens": 20})) + "\n")
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"data": "d.jsonl", "metrics": ["contains"], "fix": 1, "max_runs": 2})
    )
    result = evaluate(suite, model=Scripted())
    fix = result.items[0].fix
    assert fix is not None and fix.calls == 2 and fix.fix is None


def test_judge_errors_are_reported_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    import counterfix.judges.auto as auto
    from counterfix.core.diagnosis import Diagnosis

    class BrokenJudge:
        name = "fake:judge"

    monkeypatch.setattr(auto, "resolve_judge", lambda *a, **k: (BrokenJudge(), "test"))
    import sys

    diag_mod = sys.modules["counterfix.diagnose"]

    def judged(record: CallRecord, *, judge: Any = None, **kw: Any) -> Diagnosis:
        d = Diagnosis(record_id=record.id)
        if judge is not False:
            d.errors["judge:F4"] = "DailyLimitError: the daily limit is used up"
        return d

    monkeypatch.setattr(diag_mod, "diagnose", judged)
    result = evaluate(ROWS[:3], model=Scripted(), metrics=["contains"], judge=True)
    assert result.judge_errors == ["q3: DailyLimitError: the daily limit is used up"]
    assert "could not be judged" in result.text()


def test_a_cut_off_answer_that_meets_a_latency_gate_is_flagged_not_hidden() -> None:
    rows = [ROWS[0], dict(ROWS[3], params={"max_tokens": 20})]
    result = evaluate(rows, model=Scripted(), metrics=["latency:5000"])
    assert result.failed == [] and [i.id for i in result.flagged] == ["q4"]
    assert result.flagged_by_cause == {"F1": 1} and result.items[1].causes == ["F1"]
    text = result.text()
    assert "0 missed, 1 flagged" in text and "Flagged by live checks" in text
    assert "<h2>Flagged by live checks</h2>" in result.html()


def test_cli_fail_on_flagged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import counterfix.evals.runner as runner

    monkeypatch.setattr(runner, "_caller", lambda model: Scripted())
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(dict(ROWS[3], params={"max_tokens": 20})) + "\n")
    args = ["eval", str(data), "--model", "fake:model", "--metric", "latency:5000"]
    assert main(args) == 0
    assert main([*args, "--fail-on-flagged"]) == 1
    assert "1 item(s) flagged by live checks (F1 1)" in capsys.readouterr().out
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"data": "d.jsonl", "metrics": ["latency:5000"], "fail_on_flagged": True})
    )
    assert main(["eval", str(suite), "--model", "fake:model"]) == 1


def test_fix_search_covers_flagged_items_after_misses() -> None:
    rows = [ROWS[2], dict(ROWS[3], params={"max_tokens": 20})]  # q3 missed, q4 flagged
    result = evaluate(rows, model=Scripted(), metrics=["latency:5000", "contains:0"], fix=5)
    assert [i.id for i in result.failed] == [] and [i.id for i in result.flagged] == ["q4"]
    q4 = result.items[1]
    assert q4.fix is not None and q4.fix.fix is not None
    assert q4.fix.fix.name == "raise_max_tokens_more"
    assert result.fix_tried == 1 and result.fixes == {"raise_max_tokens_more": 1}
    assert "verified fix: Raise max_tokens" in result.text()


def test_misses_get_the_fix_budget_before_flagged_items() -> None:
    flagged = dict(ROWS[3], answer="Peru", params={"max_tokens": 20})  # cut off, but has "Peru"
    rows = [flagged, ROWS[2]]  # flagged first, missed second
    result = evaluate(rows, model=Scripted(), metrics=["contains"], fix=1)
    assert [i.id for i in result.flagged] == ["q4"] and [i.id for i in result.failed] == ["q3"]
    assert result.fix_tried == 1
    assert result.items[1].fix is not None and result.items[0].fix is None


def test_fix_search_reports_progress_on_every_re_run() -> None:
    seen: list[str] = []
    rows = [dict(ROWS[3], params={"max_tokens": 20})]
    evaluate(
        rows, model=Scripted(), metrics=["latency:5000"], fix=1,
        progress=lambda n, total, label: seen.append(label),
    )  # fmt: skip
    runs = [s for s in seen if s.startswith("fix search q4: re-run")]
    assert runs and runs[0] == "fix search q4: re-run 1/30 (plain_retry)"
    assert any("raise_max_tokens_more" in s for s in runs)


def test_a_failed_fix_search_says_why_the_runs_did_not_pass() -> None:
    class SlowWhenLong(Scripted):
        def complete(self, messages: Sequence[Message | dict[str, Any]], **params: Any) -> Any:
            r = super().complete(messages, **params)
            if (params.get("max_tokens") or 0) >= 1000:
                r.latency_ms = 9000.0  # the complete answer is too slow for the gate
            return r

    rows = [dict(ROWS[3], params={"max_tokens": 20})]
    result = evaluate(rows, model=SlowWhenLong(), metrics=["latency:5000"], fix=1)
    fix = result.items[0].fix
    assert fix is not None and fix.fix is None
    more = next(t for t in fix.tried if t.name == "raise_max_tokens_more")
    assert more.reasons() == f"latency over 5000 in {more.runs}/{more.runs}"
    assert "closest was raise_max_tokens_more" in result.text()
    assert "latency over 5000" in result.text()
