"""Code metric: does the answer's code pass the tests? Runs the code, so it needs code
execution turned on (see :mod:`counterfix.detect.code.sandbox`)."""

from __future__ import annotations

from typing import Any

from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.metrics.base import Metric, MetricResult

CODE_KEY = "code_check"
"""Record metadata: the last code check (cause, detail, signature) for fixes and reports."""


class TestsPass(Metric):
    """1.0 when the code in the answer passes every test (a miss points at F12, or F1 when
    the code was cut off). The reference holds the tests: a HumanEval/MBPP/EvalPlus row,
    ``{"tests": [...], "entry_point": ...}``, a list of asserts or test code.

    Options: ``timeout`` (seconds, default 10), ``sandbox`` ("auto", "subprocess",
    "docker"), ``allowed_imports`` (modules beyond the standard library), ``allow``.
    """

    __test__ = False  # not a pytest test class
    name = "tests"
    default_threshold = 1.0
    failure_type = FailureType.CODE_FAILURE
    needs_code_execution = True

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        raise NotImplementedError  # score() does the work: it needs the record's metadata

    def score(self, record: CallRecord, reference: Any = None) -> MetricResult:
        from counterfix.detect.code import check_code
        from counterfix.detect.code.check import as_tests

        result = MetricResult(name=self.name, threshold=self.threshold)
        if record.error is not None:
            result.passed = False
            result.reason = f"the call failed: {record.error[:120]}"
            return result
        if reference is None:
            result.reason = "no tests to run"
            return result
        spec = as_tests(reference)
        checked = check_code(
            code_answer(record),
            spec,
            finish_reason=record.finish_reason,
            allowed_imports=self.options.get("allowed_imports"),
            timeout=float(self.options.get("timeout", 10.0)),
            sandbox=self.options.get("sandbox"),
            allow=self.options.get("allow"),
        )
        record.metadata[CODE_KEY] = {
            "cause": checked.cause,
            "detail": checked.detail,
            "passed": checked.tests_passed,
            "total": checked.tests_total,
            "entry_point": spec.entry_point,
            "signature": _signature(spec.prompt, spec.entry_point),
        }
        result.value = 1.0 if checked.passed else 0.0
        result.passed = checked.passed
        tag = f"[{checked.cause}] " if checked.cause else ""
        result.reason = tag + checked.why
        if not checked.passed:
            cut = checked.cause == "cut_off"
            result.failure_type = FailureType.TRUNCATION if cut else FailureType.CODE_FAILURE
        return result


def code_answer(record: CallRecord) -> str:
    """The answer with its leading indentation kept (a bare function body must stay
    indented to continue a prompt); inline reasoning is still removed."""
    answer = record.answer_text
    raw = record.response_text or ""
    return raw.strip("\n").rstrip() if raw.strip() == answer.strip() else answer


def _signature(prompt: str | None, entry_point: str | None) -> str | None:
    """The ``def name(...)`` line the prompt asks for, if any."""
    if not prompt or not entry_point:
        return None
    for line in prompt.splitlines():
        if line.lstrip().startswith(f"def {entry_point}("):
            return line.strip().rstrip(":") + ":"
    return None
