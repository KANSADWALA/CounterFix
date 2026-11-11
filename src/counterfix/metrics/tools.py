"""Tool-call metric: did the model call the expected tool with the right arguments?

The reference is the expected call ``{"name": "get_weather", "arguments": {"city":
"Pune"}}`` (arguments optional), a list of calls (all must be made, in any order: parallel
calls), or ``"no_call"`` when the right answer uses no tool. An argument may list accepted
values: ``{"city": {"any_of": ["Pune", "Pune, India"]}}``. Strings compare without case
and outer spaces, numbers by value. Arguments the reference does not name are allowed.
"""

from __future__ import annotations

import json
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.metrics.base import Metric

TOOL_KEY = "tool_check"
NO_CALL = ("no_call", "none", "no tool")


def _same(want: Any, got: Any) -> bool:
    if isinstance(want, dict) and set(want) == {"any_of"}:
        return any(_same(w, got) for w in list(want["any_of"]))
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(want == got)
    if isinstance(want, (int, float)) and isinstance(got, (int, float, str)):
        try:
            return abs(float(want) - float(got)) < 1e-9
        except ValueError:
            return False
    if isinstance(want, str) and isinstance(got, (str, int, float)):
        return want.strip().casefold() == str(got).strip().casefold()
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(_same(a, b) for a, b in zip(want, got, strict=True))
    if isinstance(want, dict) and isinstance(got, dict):
        return all(k in got and _same(v, got[k]) for k, v in want.items())
    return bool(want == got)


def _args(raw: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(raw, dict):
        return raw, ""
    if raw in (None, ""):
        return {}, ""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, f"arguments are not JSON ({exc})"
    return (value, "") if isinstance(value, dict) else (None, "arguments are not an object")


def _calls(record: CallRecord) -> list[tuple[str, Any]]:
    out = []
    for c in record.tool_calls or []:
        fn = c.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        out.append(
            (str(name or c.get("name") or "?"), fn.get("arguments") if fn else c.get("input"))
        )
    return out


def _offered(record: CallRecord) -> set[str]:
    names = set()
    for t in record.params.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        names.add(str((fn or {}).get("name") or (t.get("name") if isinstance(t, dict) else "")))
    return {n for n in names if n}


def check_call(record: CallRecord, reference: Any) -> tuple[bool, str, str | None, dict[str, Any]]:
    """(passed, why, cause or None, details) for the record's tool calls against the
    expected call(s)."""
    calls = _calls(record)
    if isinstance(reference, str) and reference.strip().lower() in NO_CALL:
        if not calls:
            return True, "answered without a tool, as expected", None, {}
        names = ", ".join(n for n, _ in calls)
        return False, f"called {names}; no tool was needed", "extra_call", {}
    expected = reference if isinstance(reference, list) else [reference]
    expected = [e for e in expected if isinstance(e, dict) and e.get("name")]
    if not expected:
        return False, "no expected call in the reference", None, {}
    if not calls:
        return (
            False,
            f"no tool call; expected {expected[0]['name']}",
            "no_call",
            {"expected": expected[0]},
        )
    offered = _offered(record)
    used: set[int] = set()
    for want in expected:
        match = next(
            (i for i, (n, _) in enumerate(calls) if n == want["name"] and i not in used), None
        )
        if match is None:
            got = ", ".join(n for n, _ in calls)
            unknown = offered and all(n not in offered for n, _ in calls)
            cause = "unknown_tool" if unknown else "wrong_tool"
            return False, f"called {got}; expected {want['name']}", cause, {"expected": want}
        used.add(match)
        args, problem = _args(calls[match][1])
        if args is None:
            return False, f"{want['name']}: {problem}", "bad_arguments", {"expected": want}
        for key, value in (want.get("arguments") or {}).items():
            if key not in args:
                return (
                    False,
                    f"{want['name']}: missing argument {key!r}",
                    "bad_arguments",
                    {"expected": want},
                )
            if not _same(value, args[key]):
                why = f"{want['name']}: {key}={args[key]!r}, expected {value!r}"
                return False, why, "bad_arguments", {"expected": want}
    extra = len(calls) - len(used)
    tail = f" (and {extra} more call(s))" if extra else ""
    return True, f"called {', '.join(w['name'] for w in expected)} correctly{tail}", None, {}


class ToolCallMatch(Metric):
    """1.0 when the model made the expected tool call(s); a miss is F14 with its cause
    (no_call, extra_call, wrong_tool, unknown_tool, bad_arguments)."""

    name = "tool_call"
    default_threshold = 1.0
    failure_type = FailureType.TOOL_MISUSE

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        ok, why, cause, details = check_call(record, reference)
        record.metadata[TOOL_KEY] = {"cause": cause, "detail": why, **details}
        if cause is None and not ok:
            return None, why
        return (1.0 if ok else 0.0), (f"[{cause}] {why}" if cause else why)
