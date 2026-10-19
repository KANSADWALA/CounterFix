from __future__ import annotations

import json
from pathlib import Path

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import CallRecord, Config, Message
from counterfix.cli import main
from counterfix.report import html_report, text_report


def test_log_to_writes_every_guarded_call_and_load_log_reads_it(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    client = counterfix.guard(
        FakeClient(FakeCompletions(make_response("Hi <b>there</b>", model="m"))),
        config=Config(on_diagnosis=counterfix.log_to(log)),
    )
    for _ in range(2):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hi"}])
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and set(json.loads(lines[0])) == {"record", "diagnosis"}
    records = counterfix.load_log(log)
    assert [r.model for r in records] == ["m", "m"] and records[0].prompt_text == "Hi"


def test_load_log_names_the_bad_line(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text('{"model": "m"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        counterfix.load_log(log)


def _cut_off() -> CallRecord:
    return CallRecord(
        model="m",
        messages=[Message(role="user", content="Write an essay <please>.")],
        response_text="Essays are",
        finish_reason="length",
        params={"max_tokens": 4},
    )


def test_reports_count_failures_and_escape_html() -> None:
    fine = CallRecord(messages=[Message(role="user", content="Hi")], response_text="Hello!")
    reports = [counterfix.diagnose(r, judge=False) for r in (_cut_off(), fine)]
    text = text_report(reports)
    assert text.startswith("2 call(s) checked, 1 with a failure.")
    assert "F1 Truncation" in text and "fix:" in text and "Hello!" not in text
    page = html_report(reports)
    assert page.startswith("<!doctype html>") and page.count("<details") == 2
    assert "&lt;please&gt;" in page and "<please>" not in page


def test_cli_diagnose_writes_json_and_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(_cut_off().model_dump_json() + "\n", encoding="utf-8")
    out_json, out_html = tmp_path / "out.jsonl", tmp_path / "r.html"
    code = main(["diagnose", str(log), "--json", str(out_json), "--html", str(out_html)])
    assert code == 0 and "1 with a failure" in capsys.readouterr().out
    assert json.loads(out_json.read_text(encoding="utf-8"))["diagnosis"]["findings"]
    assert "F1" in out_html.read_text(encoding="utf-8")
    assert main(["diagnose", str(log), "--fail-on-failure"]) == 1


def test_context_block_attaches_source_and_reference(tmp_path: Path) -> None:
    from counterfix.judges.tasks import CONTEXT_KEY, REFERENCE_KEY

    seen: list[CallRecord] = []
    client = counterfix.guard(
        FakeClient(FakeCompletions(make_response("21 days."))),
        config=Config(on_diagnosis=lambda r, d: seen.append(r)),
    )
    with counterfix.context(source="Refunds within 21 days."), counterfix.context(reference="21"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "?"}])
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "?"}])
    assert seen[0].metadata[CONTEXT_KEY] == "Refunds within 21 days."
    assert seen[0].metadata[REFERENCE_KEY] == "21"
    assert CONTEXT_KEY not in seen[1].metadata


def test_cli_fix_without_a_provider_names_the_caller_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "calls.jsonl"
    log.write_text(_cut_off().model_dump_json() + "\n", encoding="utf-8")
    assert main(["fix", str(log)]) == 2
    out = capsys.readouterr().out
    assert "--caller provider:model" in out and "caller=" not in out
