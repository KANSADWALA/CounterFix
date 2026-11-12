"""Wave 5: agent trajectories, F14 tool-use checks, the tool_call metric, F14 fixes."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from counterfix import Config
from counterfix.benchmarks import load_benchmark, run_benchmark
from counterfix.core.detector import SkipCheck, run_detectors
from counterfix.core.diagnosis import Finding
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType
from counterfix.detect import AgentTaskCheck, ToolUseCheck, agents
from counterfix.evals import evaluate
from counterfix.fixes.engine import INTERVENTIONS
from counterfix.judges.base import Verdict
from counterfix.metrics import get_metric
from counterfix.metrics.tools import check_call

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}  # fmt: skip
SEARCH = {"type": "function", "function": {"name": "search_web", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}  # fmt: skip


def _call(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}  # fmt: skip


def _result(cid: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": cid, "content": text}


LOOP = [
    {"role": "user", "content": "Weather in Pune?"},
    _call("1", "get_weather", {"city": "Pune"}), _result("1", "error: timeout"),
    _call("2", "get_weather", {"city": "Pune"}), _result("2", "error: timeout"),
    _call("3", "get_weather", {"city": "Pune"}), _result("3", "error: timeout"),
    {"role": "assistant", "content": "Sorry, I could not get the weather."},
]  # fmt: skip


def test_trajectory_from_openai_messages() -> None:
    t = agents.from_messages(LOOP, tools=[WEATHER])
    assert t.task == "Weather in Pune?" and len(t.calls) == 3 and t.turns == 4
    assert t.calls[0].result == "error: timeout" and t.final_answer.startswith("Sorry")
    assert "[step 0] call get_weather" in t.text()


def test_trajectory_from_langchain_and_anthropic_shapes() -> None:
    ai = SimpleNamespace(
        type="ai",
        content="",
        tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "a"}],
    )
    tool = SimpleNamespace(type="tool", content="found it", tool_call_id="a")
    human = SimpleNamespace(type="human", content="find x")
    final = SimpleNamespace(type="ai", content="x is y", tool_calls=[])
    t = agents.trajectory({"messages": [human, ai, tool, final]})
    assert t.calls[0].name == "search_web" and t.calls[0].result == "found it"
    assert t.final_answer == "x is y" and t.task == "find x"
    anth = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "3 C"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "3 C in Oslo"}]},
    ]  # fmt: skip
    t = agents.trajectory(anth)
    assert t.calls[0].arguments == {"city": "Oslo"} and t.calls[0].result == "3 C"
    assert t.final_answer == "3 C in Oslo"


def test_trajectory_from_otel_spans() -> None:
    spans = [
        {"gen_ai.operation.name": "chat", "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": "Weather?"}]}])},
        {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather", "gen_ai.tool.call.arguments": '{"city": "Pune"}', "gen_ai.tool.call.result": "29 C"},
        {"gen_ai.operation.name": "chat", "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": "29 C"}]}])},
    ]  # fmt: skip
    t = agents.trajectory(spans)
    assert [c.name for c in t.calls] == ["get_weather"] and t.calls[0].result == "29 C"
    assert t.turns == 2


def test_live_checks_find_loops_unknown_and_missing_tools() -> None:
    t = agents.from_messages([*LOOP[:-1], _call("4", "send_sms", {}), LOOP[-1]], tools=[WEATHER])
    causes = [p["cause"] for p in agents.problems(t, expect=["search_web"], max_steps=3)]
    assert causes == ["unknown_tool", "loop", "missing_call", "too_many_steps"]
    rec = agents.record("Weather in Pune?", LOOP, tools=[WEATHER])
    d = run_detectors(rec, [ToolUseCheck()], Config())
    assert d.findings[0].failure_type is FailureType.TOOL_MISUSE
    assert d.findings[0].evidence[0].data["cause"] == "loop"
    with pytest.raises(SkipCheck):
        ToolUseCheck().check(CallRecord(response_text="hi"), Config())


@pytest.mark.parametrize(
    ("calls", "reference", "cause"),
    [
        ([("get_weather", {"city": "pune "})], {"name": "get_weather", "arguments": {"city": "Pune"}}, None),
        ([], {"name": "get_weather"}, "no_call"),
        ([("get_weather", {"city": "Pune"})], "no_call", "extra_call"),
        ([], "no_call", None),
        ([("search_web", {"query": "pune"})], {"name": "get_weather"}, "wrong_tool"),
        ([("send_sms", {})], {"name": "get_weather"}, "unknown_tool"),
        ([("get_weather", {"city": "Mumbai"})], {"name": "get_weather", "arguments": {"city": "Pune"}}, "bad_arguments"),
        ([("get_weather", {})], {"name": "get_weather", "arguments": {"city": "Pune"}}, "bad_arguments"),
        ([("get_weather", "{city: Pune")], {"name": "get_weather"}, "bad_arguments"),
        ([("get_weather", {"city": "Pune, India"})], {"name": "get_weather", "arguments": {"city": {"any_of": ["Pune", "Pune, India"]}}}, None),
        ([("convert", {"amount": "250"})], {"name": "convert", "arguments": {"amount": 250}}, None),
        ([("a", {}), ("b", {})], [{"name": "b"}, {"name": "a"}], None),
    ],
)  # fmt: skip
def test_tool_call_metric(calls: list[Any], reference: Any, cause: str | None) -> None:
    tool_calls = [{"id": str(i), "type": "function", "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                  for i, (n, a) in enumerate(calls)]  # fmt: skip
    rec = CallRecord(
        params={"tools": [WEATHER, SEARCH]}, tool_calls=tool_calls or None, response_text=""
    )
    ok, why, got, _ = check_call(rec, reference)
    assert got == cause, why
    assert ok is (cause is None)
    r = get_metric("tool_call").score(rec, reference)
    assert r.passed is (cause is None)
    if cause:
        assert (
            r.failure_type is FailureType.TOOL_MISUSE
            and rec.metadata["tool_check"]["cause"] == cause
        )


def _iv(name: str) -> Any:
    return next(i for i in INTERVENTIONS if i.name == name)


def test_tool_fixes() -> None:
    f = Finding(failure_type=FailureType.TOOL_MISUSE, confidence=1.0)
    rec = CallRecord(messages=[Message(role="user", content="q")], params={"tools": [WEATHER]},
                     metadata={"tool_check": {"cause": "wrong_tool", "expected": {"name": "get_weather"}}})  # fmt: skip
    _, params = _iv("tool_choice").apply(rec, f)
    assert params["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    rec.provider = "anthropic"
    assert _iv("tool_choice").apply(rec, f)[1]["tool_choice"] == {
        "type": "tool",
        "name": "get_weather",
    }
    msgs, _ = _iv("name_the_tool").apply(rec, f)
    assert "`get_weather`" in msgs[0].text
    rec.metadata["tool_check"] = {"cause": "bad_arguments"}
    _, params = _iv("describe_arguments").apply(rec, f)
    assert "city (string; required)" in params["tools"][0]["function"]["description"]
    assert "description" not in WEATHER["function"]  # the original tools are not changed
    rec.metadata["tool_check"] = {"cause": "loop"}
    assert "same arguments" in _iv("no_repeat").apply(rec, f)[0][0].text


class _Picker:
    """Calls get_weather for everything, unless tool_choice names another tool."""

    name = "fake:picker"

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        choice = params.get("tool_choice")
        name = choice["function"]["name"] if isinstance(choice, dict) else "get_weather"
        spec = next(t["function"] for t in params["tools"] if t["function"]["name"] == name)
        props = spec["parameters"]["properties"]
        args = {k: (1 if props[k].get("type") in ("number", "integer") else "x")
                for k in spec["parameters"]["required"]}  # fmt: skip
        if name == "get_weather":
            args = {"city": "Pune"}
        call = {
            "id": "1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        return CallRecord(params=params, tool_calls=[call], finish_reason="tool_calls")


def test_tools_benchmark_finds_and_fixes() -> None:
    assert len(load_benchmark("tools")) == 20
    result = run_benchmark("tools", model=_Picker(), fix=2, max_runs=9)
    missed = [i for i in result.items if not i.passed]
    assert missed and missed[0].causes[0] == "F14"
    fixed = [i for i in missed if i.fix and i.fix.fix]
    assert fixed and fixed[0].fix.fix.name == "tool_choice"


class _Judge:
    name = "fake:judge"

    def ask(self, task: Any) -> Verdict:
        assert "get_weather" in task.material["run"]
        return Verdict(task=task.name, label="not_completed", confidence=0.9, reason="r",
                       extra={"cause": "ignored_result"})  # fmt: skip


def test_agent_task_judge_and_app_runs() -> None:
    rec = agents.record("Weather in Pune?", LOOP, tools=[WEATHER])
    found = AgentTaskCheck(_Judge()).check(rec, Config())
    assert found[0].evidence[0].data["cause"] == "ignored_result"

    def app(question: str) -> list[dict[str, Any]]:
        return LOOP

    rows = [{"question": "Weather in Pune?", "expect_tools": ["search_web"]}]
    out = evaluate(rows, app=app, metrics=["contains"])
    item = out.items[0]
    assert agents.of(item.record) is not None
    assert item.flagged and item.causes == ["F14"]  # the loop and the missing search
