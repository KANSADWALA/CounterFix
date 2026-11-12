"""Interventions for tool misuse (F14): tool_choice, naming the tool, describing arguments."""

from __future__ import annotations

import copy
from typing import Any

from counterfix.core.diagnosis import Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import PrescriptionKind
from counterfix.fixes.interventions.base import (
    Change,
    F,
    Intervention,
    _system,
)


def _tool_check(r: CallRecord, f: Finding) -> dict[str, Any]:
    check = r.metadata.get("tool_check")
    if isinstance(check, dict) and check.get("cause"):
        return check
    for e in f.evidence:  # live F14 findings carry the cause in their evidence
        if e.data.get("cause"):
            return {"cause": e.data["cause"], "expected": {"name": e.data.get("tool")}}
    return {}


def _expected_name(check: dict[str, Any]) -> str | None:
    want = check.get("expected")
    name = want.get("name") if isinstance(want, dict) else None
    return str(name) if name else None


def _tool_choice(r: CallRecord, f: Finding) -> Change | None:
    """Force the call: the expected tool when the right tool is known, else any tool;
    no tool when none was needed."""
    if not r.params.get("tools"):
        return None
    check = _tool_check(r, f)
    cause, name = check.get("cause"), _expected_name(check)
    anthropic = r.provider == "anthropic"
    if cause == "extra_call":
        choice: Any = {"type": "none"} if anthropic else "none"
    elif cause in ("no_call", "wrong_tool", "missing_call") and name:
        choice = (
            {"type": "tool", "name": name}
            if anthropic
            else {"type": "function", "function": {"name": name}}
        )
    elif cause in ("no_call", "missing_call"):
        choice = {"type": "any"} if anthropic else "required"
    else:
        return None
    return list(r.messages), {**r.params, "tool_choice": choice}


def _name_the_tool(r: CallRecord, f: Finding) -> Change | None:
    check = _tool_check(r, f)
    name = _expected_name(check)
    if check.get("cause") == "extra_call":
        rule = (
            "Answer directly when no tool is needed; call a tool only when the request needs "
            "data or an action only that tool can give."
        )
        return _system(rule)(r, f)
    if not name or check.get("cause") not in (
        "no_call",
        "wrong_tool",
        "unknown_tool",
        "missing_call",
    ):
        return None
    return _system(f"For this request, call the `{name}` tool.")(r, f)


def _describe_arguments(r: CallRecord, f: Finding) -> Change | None:
    """Spell out each tool's arguments (type, required, allowed values) in its description."""
    tools = r.params.get("tools")
    if not tools or _tool_check(r, f).get("cause") not in ("bad_arguments", "wrong_tool", None):
        return None
    out = []
    changed = False
    for tool in copy.deepcopy(tools):
        fn = tool.get("function") if isinstance(tool, dict) and "function" in tool else tool
        schema = (fn.get("parameters") or fn.get("input_schema")) if isinstance(fn, dict) else None
        props = (schema or {}).get("properties") or {}
        if isinstance(fn, dict) and props:
            required = set((schema or {}).get("required") or [])
            parts = []
            for key, spec in props.items():
                spec = spec if isinstance(spec, dict) else {}
                bits = [str(spec.get("type", "any"))]
                bits.append("required" if key in required else "optional")
                if spec.get("enum"):
                    bits.append("one of " + ", ".join(map(str, spec["enum"])))
                if spec.get("format"):
                    bits.append(f"format {spec['format']}")
                text = f"{key} ({'; '.join(bits)})"
                if spec.get("description"):
                    text += f": {spec['description']}"
                parts.append(text)
            note = "Arguments: " + ". ".join(parts) + "."
            if note not in str(fn.get("description", "")):
                fn["description"] = f"{fn.get('description', '').rstrip()} {note}".strip()
                changed = True
        out.append(tool)
    return (list(r.messages), {**r.params, "tools": out}) if changed else None


def _no_repeat(r: CallRecord, f: Finding) -> Change | None:
    if _tool_check(r, f).get("cause") not in ("loop", "too_many_steps"):
        return None
    return _system(
        "Never call a tool again with the same arguments: if a result did not help, change "
        "the arguments, try another tool, or answer with what you have."
    )(r, f)


TOOLS: list[Intervention] = [
    Intervention(
        "tool_choice",
        F.TOOL_MISUSE,
        PrescriptionKind.PARAM_CHANGE,
        "Set tool_choice: the right tool when known, any tool when one was needed, none when "
        "no tool was needed",
        _tool_choice,
    ),
    Intervention(
        "name_the_tool",
        F.TOOL_MISUSE,
        PrescriptionKind.PROMPT_CHANGE,
        "Say in the system prompt which tool the request needs (or that it needs none)",
        _name_the_tool,
    ),
    Intervention(
        "describe_arguments",
        F.TOOL_MISUSE,
        PrescriptionKind.PARAM_CHANGE,
        "Spell out each tool's arguments (type, required, allowed values) in its description",
        _describe_arguments,
    ),
    Intervention(
        "no_repeat",
        F.TOOL_MISUSE,
        PrescriptionKind.PROMPT_CHANGE,
        "Tell the model never to repeat a tool call with the same arguments",
        _no_repeat,
    ),
]
