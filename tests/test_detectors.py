from __future__ import annotations

import json
from typing import Any

import pytest

from counterfix import CallRecord, Config, FailureType, Message, SkipCheck, Usage
from counterfix.detectors import (
    EmptyOutputDetector,
    FormatDetector,
    ProviderFilterDetector,
    RefusalDetector,
    TruncationDetector,
    expected_format,
)

CFG = Config()


def rec(
    text: str | None = "ok",
    *,
    finish: str | None = "stop",
    prompt: str = "hi",
    system: str | None = None,
    params: dict[str, Any] | None = None,
    **extra: Any,
) -> CallRecord:
    messages = [Message(role="system", content=system)] if system else []
    messages.append(Message(role="user", content=prompt))
    return CallRecord(
        messages=messages, response_text=text, finish_reason=finish, params=params or {}, **extra
    )


def types(findings: list[Any]) -> list[FailureType]:
    return [f.failure_type for f in findings]


# --- provider filter and empty output ----------------------------------------------


@pytest.mark.parametrize("reason", ["content_filter", "SAFETY", "recitation"])
def test_provider_filter(reason: str) -> None:
    assert types(ProviderFilterDetector().check(rec("", finish=reason), CFG)) == [
        FailureType.PROVIDER_FILTER
    ]
    assert ProviderFilterDetector().check(rec("hello"), CFG) == []


def test_empty_output() -> None:
    d = EmptyOutputDetector()
    assert types(d.check(rec(""), CFG)) == [FailureType.EMPTY_OUTPUT]
    assert types(d.check(rec("<think>only thinking</think>"), CFG)) == [FailureType.EMPTY_OUTPUT]
    assert "reasoning only" in d.check(rec("<think>x</think>"), CFG)[0].evidence[0].detail
    assert d.check(rec("", finish="length"), CFG) == []  # truncation explains it
    assert d.check(rec("", finish="content_filter"), CFG) == []
    assert d.check(rec(None, tool_calls=[{"id": "t"}]), CFG) == []
    assert d.check(rec("hello"), CFG) == []


# --- F1 truncation ------------------------------------------------------------------


def test_truncation_from_stop_reason() -> None:
    [f] = TruncationDetector().check(
        rec("An essay", finish="length", params={"max_tokens": 40}), CFG
    )
    assert f.failure_type is FailureType.TRUNCATION and not f.inferred
    assert f.prescription is not None and f.prescription.patch == {"max_tokens": 160}


def test_truncation_reasoning_budget() -> None:
    record = rec("", finish="length", params={"max_tokens": 50}, reasoning_text="thinking")
    [f] = TruncationDetector().check(record, CFG)
    assert "reasoning" in f.evidence[0].detail
    assert f.prescription is not None and f.prescription.patch == {"max_tokens": 400}


def test_truncation_limit_is_capped() -> None:
    [f] = TruncationDetector().check(rec("x", finish="length", params={"max_tokens": 20000}), CFG)
    assert f.prescription is not None and f.prescription.patch == {"max_tokens": 32768}


def test_no_truncation_on_normal_stop_or_tool_calls() -> None:
    d = TruncationDetector()
    assert d.check(rec("Done.", params={"max_tokens": 40}), CFG) == []
    assert d.check(rec(None, finish="tool_calls", params={"max_tokens": 40}), CFG) == []


def test_truncation_inferred_from_token_count_without_stop_reason() -> None:
    record = rec(
        "Some text", finish=None, params={"max_tokens": 100}, usage=Usage(completion_tokens=100)
    )
    [f] = TruncationDetector().check(record, CFG)
    assert f.inferred and f.confidence == 0.8


def test_truncation_inferred_from_mid_sentence_stop() -> None:
    cut = "word " * 40 + "and then the"  # ~52 tokens of text for a limit of 50
    [f] = TruncationDetector().check(rec(cut, finish=None, params={"max_tokens": 50}), CFG)
    assert f.inferred and f.confidence == 0.6
    finished = "word " * 40 + "the end."
    assert (
        TruncationDetector().check(rec(finished, finish=None, params={"max_tokens": 50}), CFG) == []
    )
    assert TruncationDetector().check(rec(cut, finish=None), CFG) == []  # no limit known


# --- F2 format ----------------------------------------------------------------------

JSON_PROMPT = "Return ONLY a JSON object with keys name and age."
SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}


def test_expected_format_sources() -> None:
    assert expected_format(rec(prompt=JSON_PROMPT))[0]
    assert expected_format(rec(prompt="Answer in valid JSON."))[0]
    assert not expected_format(rec(prompt="What is JSON used for?"))[0]
    assert expected_format(rec(params={"response_format": {"type": "json_object"}}))[0]
    wants, schema, _ = expected_format(
        rec(params={"response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}})
    )
    assert wants and schema == SCHEMA
    assert expected_format(rec(metadata={"expected_schema": SCHEMA}))[1] == SCHEMA


def test_valid_json_passes() -> None:
    assert (
        FormatDetector().check(rec('{"name": "Priya", "age": 29}', prompt=JSON_PROMPT), CFG) == []
    )


def test_broken_json_is_flagged_with_json_mode_fix() -> None:
    [f] = FormatDetector().check(rec('Sure! {"name": "Priya", "age": 29}', prompt=JSON_PROMPT), CFG)
    assert f.failure_type is FailureType.FORMAT_BREAK and f.confidence == 0.9
    assert f.prescription is not None
    assert f.prescription.patch == {"response_format": {"type": "json_object"}}
    assert f.evidence[0].data["repairable"] is True


def test_code_fence_is_a_soft_finding() -> None:
    [f] = FormatDetector().check(
        rec('```json\n{"name": "A", "age": 1}\n```', prompt=JSON_PROMPT), CFG
    )
    assert f.confidence == 0.5 and "code fence" in f.evidence[0].detail


def test_schema_mismatch() -> None:
    record = rec('{"name": "A", "age": "29"}', metadata={"expected_schema": SCHEMA})
    [f] = FormatDetector().check(record, CFG)
    assert "age" in f.evidence[0].detail
    assert f.prescription is not None
    assert f.prescription.patch["response_format"]["type"] == "json_schema"


def test_think_blocks_are_ignored_for_json() -> None:
    text = '<think>the user wants JSON</think>{"name": "A", "age": 1}'
    assert FormatDetector().check(rec(text, prompt=JSON_PROMPT), CFG) == []


def test_format_skips_truncated_answers() -> None:
    with pytest.raises(SkipCheck):
        FormatDetector().check(rec('{"name": "A",', finish="length", prompt=JSON_PROMPT), CFG)


def test_no_json_expected_no_check() -> None:
    assert FormatDetector().check(rec("plain text answer", prompt="Tell me a joke"), CFG) == []


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def call(name: str, args: str) -> dict[str, Any]:
    return {"id": "1", "type": "function", "function": {"name": name, "arguments": args}}


def test_tool_call_arguments() -> None:
    d = FormatDetector()
    ok = rec(
        None,
        finish="tool_calls",
        params={"tools": TOOLS},
        tool_calls=[call("get_weather", json.dumps({"city": "Pune"}))],
    )
    assert d.check(ok, CFG) == []
    bad_json = rec(
        None,
        finish="tool_calls",
        params={"tools": TOOLS},
        tool_calls=[call("get_weather", "{city: Pune")],
    )
    assert "not JSON" in d.check(bad_json, CFG)[0].evidence[0].detail
    bad_schema = rec(
        None,
        finish="tool_calls",
        params={"tools": TOOLS},
        tool_calls=[call("get_weather", json.dumps({"town": "Pune"}))],
    )
    assert "schema" in d.check(bad_schema, CFG)[0].evidence[0].detail
    unknown = rec(
        None, finish="tool_calls", params={"tools": TOOLS}, tool_calls=[call("send_email", "{}")]
    )
    assert "not offered" in d.check(unknown, CFG)[0].evidence[0].detail


# --- F3 refusal ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I cannot assist with this request.",
        "I'm not able to provide instructions on that.",
        "I must decline this request.",
        "As an AI language model, I cannot write threats.",
        "Sorry, that goes against my guidelines.",
        "I\u2019m sorry, but I can\u2019t help with that.",
        "I am unable to answer that.",
        # seen live from qwen2.5:3b on the seed set (F3-003)
        "I am Qwen. Creating material with threatening content is not aligned with my programming.",
        "I don't feel comfortable writing that.",
    ],
)
def test_refusals_are_flagged(text: str) -> None:
    [f] = RefusalDetector().check(rec(text), CFG)
    assert f.failure_type is FailureType.OVER_REFUSAL and f.confidence == 0.7
    assert "judge" in f.evidence[0].detail


@pytest.mark.parametrize(
    "text",
    [
        "To kill a hanging process, run `kill -9 <pid>`.",
        "Here is the villain's monologue: 'You cannot stop me!'",
        "I can help with that. First, open the terminal.",
        "",
    ],
)
def test_normal_answers_are_not_refusals(text: str) -> None:
    assert RefusalDetector().check(rec(text), CFG) == []


def test_refusal_late_in_long_answer_is_ignored() -> None:
    text = "Here is a full guide. " * 40 + "Note: I can't help with illegal uses."
    assert RefusalDetector().check(rec(text), CFG) == []


def test_premise_correction_is_not_treated_as_a_confident_refusal() -> None:
    # seen live from qwen2.5:3b on the seed set (F5-003)
    text = (
        "Alas, I cannot provide information about when William Shakespeare wrote 'The Old Man "
        "and the Sea' because this piece of fiction is not his work. It was written by Hemingway."
    )
    [f] = RefusalDetector().check(rec(text), CFG)
    assert f.confidence == 0.3 and f.evidence[0].data["premise_correction"] is True
    assert "false premise" in f.evidence[0].detail
