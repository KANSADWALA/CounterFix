"""Agents and tool calls: read an agent run as a trajectory and check how it used tools.

A trajectory is the run's tool calls in order, with their results, the tools offered and
the final answer. Build one from what your framework already gives you:

* an OpenAI-style message list (``assistant`` messages with ``tool_calls``, ``tool``
  messages with results): plain loops, the OpenAI SDK, LiteLLM, Groq, Ollama;
* LangChain / LangGraph messages (``AIMessage.tool_calls``, ``ToolMessage``), or a
  LangGraph state ``{"messages": [...]}``;
* Anthropic messages (``tool_use`` / ``tool_result`` blocks);
* OpenTelemetry GenAI spans (``execute_tool`` spans with ``gen_ai.tool.*``);
* a list of guarded responses or ``CallRecord`` objects, one per model call.

``counterfix.agents.record(task, run)`` turns a run into a record for ``diagnose()`` and
``evaluate(app=...)``; the live check :class:`ToolUseCheck` then reports F14 when a call
names a tool that was not offered, when the same call repeats (a loop), or when a tool
the item expects was never called. Free, no extra calls.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.record import CallRecord, Message

TRAJECTORY_KEY = "trajectory"
EXPECT_KEY = "expect_tools"
"""Record metadata: tool names the run must call (missing ones are F14)."""


class ToolCall(BaseModel):
    """One tool call, with its result when the run has it."""

    name: str
    arguments: Any = None
    id: str | None = None
    result: str | None = None
    step: int = 0
    """Which model turn made the call (0-based)."""

    @property
    def key(self) -> str:
        """Name and arguments, normalised: equal keys are the same call."""
        return f"{self.name}:{_canonical(self.arguments)}"


class Trajectory(BaseModel):
    task: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = None
    turns: int = 0
    """Model turns (assistant messages or model calls)."""

    @property
    def tool_names(self) -> list[str]:
        return [_tool_name(t) for t in self.tools if _tool_name(t)]

    def text(self, limit: int = 12_000) -> str:
        """The run as readable lines, for a judge or a report."""
        lines = [f"user: {self.task}"] if self.task else []
        for c in self.calls:
            lines.append(f"[step {c.step}] call {c.name}({_canonical(c.arguments)})")
            if c.result is not None:
                lines.append(f"  -> {c.result[:600]}")
        if self.final_answer:
            lines.append(f"final answer: {self.final_answer}")
        out = "\n".join(lines)
        return out if len(out) <= limit else out[: limit // 2] + "\n...\n" + out[-limit // 2 :]


# --- reading runs --------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _canonical(args: Any) -> str:
    if isinstance(args, str):
        text = args
        try:
            args = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text.strip()
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


def _tool_name(tool: Any) -> str:
    fn = _get(tool, "function")
    return str(_get(fn, "name") or _get(tool, "name") or "")


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif _get(p, "type") in ("text", "output_text") and isinstance(_get(p, "text"), str):
                parts.append(_get(p, "text"))
        return "\n".join(parts)
    return str(content)


def _role(m: Any) -> str:
    role = _get(m, "role")
    if role:
        return str(role)
    kind = str(_get(m, "type") or type(m).__name__).lower()  # LangChain message objects
    if "human" in kind:
        return "user"
    if kind in ("ai", "aimessage", "aimessagechunk") or kind.startswith("ai"):
        return "assistant"
    if "tool" in kind:
        return "tool"
    if "system" in kind:
        return "system"
    return kind


def from_messages(
    messages: Sequence[Any], tools: Sequence[Any] | None = None, task: str | None = None
) -> Trajectory:
    """A trajectory from a message list (OpenAI, LangChain or Anthropic shapes)."""
    calls: list[ToolCall] = []
    by_id: dict[str, ToolCall] = {}
    turns, final, first_user = 0, None, None
    for m in messages:
        role = _role(m)
        content = _get(m, "content")
        if role == "user":
            if isinstance(content, list) and any(_get(b, "type") == "tool_result" for b in content):
                for b in content:  # Anthropic: tool results come back in a user message
                    if _get(b, "type") == "tool_result" and (
                        c := by_id.get(str(_get(b, "tool_use_id")))
                    ):
                        c.result = _text(_get(b, "content"))
                continue
            first_user = first_user if first_user is not None else _text(content)
        elif role == "assistant":
            made = _calls_of(m, turns)
            for c in made:
                calls.append(c)
                if c.id:
                    by_id[c.id] = c
            text = _text(content)
            if not made and text.strip():
                final = text
            elif text.strip():
                final = None  # it said something, then called a tool: not the final answer
            turns += 1
        elif role == "tool":
            cid = str(_get(m, "tool_call_id") or "")
            target = by_id.get(cid) or next((c for c in reversed(calls) if c.result is None), None)
            if target is not None:
                target.result = _text(content)
    return Trajectory(
        task=task if task is not None else (first_user or ""),
        tools=[_plain(t) for t in tools or []],
        calls=calls,
        final_answer=final,
        turns=turns,
    )


def _plain(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else {"name": str(_get(obj, "name") or obj)}


def _calls_of(message: Any, step: int) -> list[ToolCall]:
    """Tool calls on an assistant message (OpenAI, LangChain or Anthropic shape)."""
    out: list[ToolCall] = []
    for c in _get(message, "tool_calls") or []:
        fn = _get(c, "function")
        if fn is not None:  # OpenAI: {"id", "function": {"name", "arguments"}}
            out.append(ToolCall(name=str(_get(fn, "name")), arguments=_get(fn, "arguments"),
                                id=_get(c, "id"), step=step))  # fmt: skip
        else:  # LangChain: {"name", "args", "id"}
            out.append(ToolCall(name=str(_get(c, "name")), arguments=_get(c, "args"),
                                id=_get(c, "id"), step=step))  # fmt: skip
    content = _get(message, "content")
    if isinstance(content, list):  # Anthropic tool_use blocks
        for b in content:
            if _get(b, "type") == "tool_use":
                out.append(ToolCall(name=str(_get(b, "name")), arguments=_get(b, "input"),
                                    id=_get(b, "id"), step=step))  # fmt: skip
    return out


def from_records(records: Iterable[Any], task: str | None = None) -> Trajectory:
    """A trajectory from one record (or guarded response) per model call, in order."""
    from counterfix.adapters import get_diagnosis

    recs: list[CallRecord] = []
    for r in records:
        if isinstance(r, CallRecord):
            recs.append(r)
        elif (d := get_diagnosis(r)) is not None and d.record is not None:
            recs.append(d.record)
    if not recs:
        return Trajectory(task=task or "")
    last = recs[-1]
    messages: list[Any] = [m.model_dump(exclude_none=True) for m in last.messages]
    messages.append({"role": "assistant", "content": last.response_text,
                     "tool_calls": last.tool_calls})  # fmt: skip
    return from_messages(messages, tools=last.params.get("tools") or [], task=task)


def from_spans(spans: Iterable[Mapping[str, Any]], task: str | None = None) -> Trajectory:
    """A trajectory from OpenTelemetry GenAI span attributes, in start order: ``chat``
    spans are model turns, ``execute_tool`` spans are tool calls (``gen_ai.tool.name``,
    ``gen_ai.tool.call.arguments``, ``gen_ai.tool.call.result``, opt-in in the
    conventions, so arguments and results may be missing)."""
    from counterfix.otel import record_from_span

    calls: list[ToolCall] = []
    turns, final, first = 0, None, task
    for attrs in spans:
        op = attrs.get("gen_ai.operation.name")
        if op == "execute_tool":
            calls.append(
                ToolCall(
                    name=str(attrs.get("gen_ai.tool.name", "?")),
                    arguments=attrs.get("gen_ai.tool.call.arguments"),
                    id=attrs.get("gen_ai.tool.call.id"),
                    result=_text(attrs.get("gen_ai.tool.call.result")) or None,
                    step=max(turns - 1, 0),
                )
            )
        elif op in ("chat", "text_completion", "generate_content"):
            rec = record_from_span(attrs)
            if first is None:
                first = next((m.text for m in rec.messages if m.role == "user"), "")
            final = rec.response_text or final
            turns += 1
    return Trajectory(task=first or "", calls=calls, final_answer=final, turns=turns)


def trajectory(
    run: Any, *, tools: Sequence[Any] | None = None, task: str | None = None
) -> Trajectory:
    """A trajectory from any supported run shape (see the module docstring)."""
    if isinstance(run, Trajectory):
        return run
    if isinstance(run, Mapping) and isinstance(run.get("messages"), list):  # LangGraph state
        return from_messages(run["messages"], tools=tools, task=task)
    items = list(run) if isinstance(run, (list, tuple)) else [run]
    if items and all(isinstance(i, Mapping) and "gen_ai.operation.name" in i for i in items):
        return from_spans(items, task=task)
    if items and all(isinstance(i, CallRecord) or hasattr(i, "choices") for i in items):
        return from_records(items, task=task)
    return from_messages(items, tools=tools, task=task)


def record(
    task: str,
    run: Any,
    *,
    tools: Sequence[Any] | None = None,
    expect_tools: Sequence[str] | None = None,
    max_steps: int | None = None,
) -> CallRecord:
    """A record of one agent run, for ``diagnose()`` and ``evaluate(app=...)``: the task,
    the final answer, and the trajectory in ``metadata["trajectory"]``."""
    traj = trajectory(run, tools=tools, task=task)
    rec = CallRecord(
        messages=[Message(role="user", content=task)],
        response_text=traj.final_answer,
        params={"tools": traj.tools} if traj.tools else {},
        metadata={TRAJECTORY_KEY: traj.model_dump(mode="json")},
    )
    if expect_tools:
        rec.metadata[EXPECT_KEY] = list(expect_tools)
    if max_steps is not None:
        rec.metadata["max_steps"] = max_steps
    return rec


def of(rec: CallRecord) -> Trajectory | None:
    """The trajectory on a record, if any."""
    raw = rec.metadata.get(TRAJECTORY_KEY)
    return Trajectory.model_validate(raw) if isinstance(raw, dict) else None


# --- checks --------------------------------------------------------------------------------


def _issue(cause: str, detail: str, **data: Any) -> dict[str, Any]:
    return {"cause": cause, "detail": detail, **data}


def problems(
    traj: Trajectory,
    *,
    expect: Sequence[str] | None = None,
    max_steps: int | None = None,
    repeat_limit: int = 3,
) -> list[dict[str, Any]]:
    """What went wrong with tool use in a run, as {"cause", "detail", ...} dicts:

    * ``unknown_tool``: a call names a tool that was not offered (when tools are known);
    * ``loop``: the same call (name and arguments) ``repeat_limit`` times, or twice in a
      row with the same result;
    * ``missing_call``: a tool in ``expect`` was never called;
    * ``too_many_steps``: more model turns than ``max_steps``.
    """
    found: list[dict[str, Any]] = []
    offered = set(traj.tool_names)
    if offered:
        near = ", ".join(sorted(offered))
        for c in traj.calls:
            if c.name not in offered:
                why = f"called {c.name!r}, which was not offered ({near})"
                found.append(_issue("unknown_tool", why, tool=c.name, step=c.step))
    counts = Counter(c.key for c in traj.calls)
    flagged: set[str] = set()
    for c in traj.calls:
        if counts[c.key] >= repeat_limit and c.key not in flagged:
            flagged.add(c.key)
            why = f"called {c.name} with the same arguments {counts[c.key]} times"
            found.append(_issue("loop", why, tool=c.name, step=c.step))
    for a, b in zip(traj.calls, traj.calls[1:], strict=False):
        if a.key == b.key and a.result == b.result and a.key not in flagged:
            flagged.add(a.key)
            why = f"repeated {a.name} with the same arguments and got the same result"
            found.append(_issue("loop", why, tool=a.name, step=b.step))
    called = {c.name for c in traj.calls}
    for name in expect or []:
        if name not in called:
            what = "no tool at all" if not called else ", ".join(sorted(called))
            found.append(_issue("missing_call", f"never called {name} (called: {what})", tool=name))
    if max_steps is not None and traj.turns > max_steps:
        why = f"{traj.turns} model turns, over the limit of {max_steps}"
        found.append(_issue("too_many_steps", why))
    return found
