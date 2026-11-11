"""Structured output: is the answer valid JSON (and valid against a schema), and do its
fields match the expected ones. A miss points to F2 (format break)."""

from __future__ import annotations

import json
import re
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.metrics.base import Metric

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json(text: str) -> Any:
    """Parse the answer as JSON, allowing one surrounding ```json fence. Raises ValueError."""
    if m := _FENCE.match(text):
        text = m.group(1)
    return json.loads(text)


class JsonValid(Metric):
    """1.0 when the answer parses as JSON and, if ``schema=`` is given, matches it."""

    name = "json_valid"
    needs_reference = False
    default_threshold = 1.0
    failure_type = FailureType.FORMAT_BREAK

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        try:
            data = parse_json(record.answer_text)
        except ValueError as exc:
            return 0.0, f"not valid JSON: {exc}"
        schema = self.options.get("schema")
        if schema:
            import jsonschema

            errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(data), key=str)
            if errors:
                return 0.0, f"schema: {errors[0].message[:120]}"
        return 1.0, "valid JSON"


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): data}


class JsonMatch(Metric):
    """Share of the reference's fields (nested keys flattened) that the answer gets right."""

    name = "json_match"
    default_threshold = 1.0
    failure_type = FailureType.FORMAT_BREAK

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        want = reference
        if isinstance(want, str):
            try:
                want = parse_json(want)
            except ValueError:
                return None, "the reference is not JSON"
        try:
            got = parse_json(record.answer_text)
        except ValueError as exc:
            return 0.0, f"not valid JSON: {exc}"
        want_f, got_f = _flatten(want), _flatten(got)
        if not want_f:
            return None, "the reference has no fields"
        wrong = [k for k, v in want_f.items() if got_f.get(k) != v]
        share = 1 - len(wrong) / len(want_f)
        return (
            share,
            "all fields match" if not wrong else f"wrong or missing: {', '.join(wrong[:5])}",
        )
