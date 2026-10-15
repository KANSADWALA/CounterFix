"""F2: the answer breaks the requested format (JSON, a JSON schema, or tool-call arguments)."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
from json_repair import repair_json

from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detectors.basic import LENGTH_REASONS

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_ASKS_FOR_JSON = re.compile(
    r"\b(?:only|valid|return|output|respond|reply|answer|format(?:ted)?|as)\b[^.\n]{0,40}\bjson\b"
    r"|\bjson\b[^.\n]{0,30}\b(?:only|object|array|list|format)\b",
    re.IGNORECASE,
)

SCHEMA_METADATA_KEY = "expected_schema"


def expected_format(record: CallRecord) -> tuple[bool, dict[str, Any] | None, str]:
    """Return (expects JSON, schema or None, where the expectation came from)."""
    schema = record.metadata.get(SCHEMA_METADATA_KEY)
    if isinstance(schema, dict):
        return True, schema, "schema given to counterfix"
    response_format = record.params.get("response_format")
    if isinstance(response_format, dict):
        kind = response_format.get("type")
        if kind == "json_schema":
            spec = response_format.get("json_schema") or {}
            inner = spec.get("schema") if isinstance(spec, dict) else None
            return True, inner if isinstance(inner, dict) else None, "response_format json_schema"
        if kind == "json_object":
            return True, None, "response_format json_object"
    text = f"{record.system_text}\n{record.prompt_text}"
    if _ASKS_FOR_JSON.search(text):
        return True, None, "the prompt asks for JSON"
    return False, None, ""


def _json_mode_fix() -> Prescription:
    return Prescription(
        kind=PrescriptionKind.PARAM_CHANGE,
        description="Turn on JSON mode or structured output, and put the schema in the prompt",
        patch={"response_format": {"type": "json_object"}},
    )


class FormatDetector:
    """Checks JSON answers and tool-call arguments. Runs on the visible answer, with any
    inline reasoning removed, so <think> blocks are not mistaken for broken JSON."""

    name = "format"
    covers = (FailureType.FORMAT_BREAK,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        if (record.finish_reason or "").lower() in LENGTH_REASONS:
            raise SkipCheck("answer was truncated; see F1")
        findings = self._check_tool_calls(record)
        wants_json, schema, source = expected_format(record)
        if wants_json and not record.tool_calls:
            finding = self._check_json(record.answer_text, schema, source)
            if finding is not None:
                findings.append(finding)
        return findings

    def _check_json(
        self, answer: str, schema: dict[str, Any] | None, source: str
    ) -> Finding | None:
        if not answer:
            return None  # empty output is reported by the empty-output check
        evidence: list[Evidence] = []
        text, fenced = answer, False
        match = _FENCE.match(answer)
        if match:
            text, fenced = match.group(1), True
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            repaired = repair_json(text, return_objects=True)
            repairable = repaired not in ("", None)
            detail = f"expected JSON ({source}) but it does not parse: {exc.msg} at char {exc.pos}"
            if repairable:
                detail += "; a tolerant parser can repair it"
            return Finding(
                failure_type=FailureType.FORMAT_BREAK,
                confidence=0.9,
                evidence=[
                    Evidence(detector=self.name, detail=detail, data={"repairable": repairable})
                ],
                prescription=_json_mode_fix(),
            )
        if fenced:
            evidence.append(
                Evidence(
                    detector=self.name, detail="valid JSON, but wrapped in a markdown code fence"
                )
            )
        if schema is not None:
            errors = sorted(
                jsonschema.Draft202012Validator(schema).iter_errors(value), key=lambda e: e.path
            )
            if errors:
                first = errors[0]
                where = "/".join(str(p) for p in first.absolute_path) or "(root)"
                evidence.append(
                    Evidence(
                        detector=self.name,
                        detail=f"JSON does not match the schema at {where}: {first.message}",
                        data={"n_errors": len(errors)},
                    )
                )
                return Finding(
                    failure_type=FailureType.FORMAT_BREAK,
                    confidence=0.9,
                    evidence=evidence,
                    prescription=Prescription(
                        kind=PrescriptionKind.PARAM_CHANGE,
                        description="Use structured output with this schema (json_schema mode)",
                        patch={
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {"name": "answer", "schema": schema},
                            }
                        },
                    ),
                )
        if fenced:
            return Finding(
                failure_type=FailureType.FORMAT_BREAK,
                confidence=0.5,
                evidence=evidence,
                prescription=_json_mode_fix(),
            )
        return None

    def _check_tool_calls(self, record: CallRecord) -> list[Finding]:
        if not record.tool_calls:
            return []
        schemas: dict[str, Any] = {}
        for tool in record.params.get("tools") or []:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            if isinstance(function, dict) and "name" in function:
                schemas[function["name"]] = function.get("parameters")
        problems: list[Evidence] = []
        for call in record.tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "?")
            raw = function.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as exc:
                problems.append(
                    Evidence(
                        detector=self.name,
                        detail=f"tool {name!r}: arguments are not JSON ({exc.msg})",
                    )
                )
                continue
            schema = schemas.get(name)
            if name not in schemas and schemas:
                problems.append(
                    Evidence(detector=self.name, detail=f"tool {name!r} was not offered")
                )
            elif isinstance(schema, dict):
                error = next(iter(jsonschema.Draft202012Validator(schema).iter_errors(args)), None)
                if error is not None:
                    problems.append(
                        Evidence(
                            detector=self.name,
                            detail=f"tool {name!r}: arguments break its schema: {error.message}",
                        )
                    )
        if not problems:
            return []
        return [
            Finding(
                failure_type=FailureType.FORMAT_BREAK,
                confidence=0.9,
                evidence=problems,
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=(
                        "Tighten the tool descriptions and parameter schemas; "
                        "use strict tool mode if available"
                    ),
                ),
            )
        ]
