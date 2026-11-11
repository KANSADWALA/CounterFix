"""HumanEval+ and MBPP+ scored by EvalPlus itself, so the numbers are EvalPlus's numbers.

The ``evalplus`` metric hands the model's code to ``evalplus.evaluate.check_correctness``
with EvalPlus's expected outputs (base tests, plus the extra "plus" tests), exactly as
``evalplus.evaluate`` does. When the code fails, counterfix's own checks name the cause
(F12: syntax, name, signature, runtime, wrong output, timeout). Needs the ``code`` extra
(``pip install "counterfix[code]"``) and code execution turned on. EvalPlus runs code in
a child process with its own guard, not in counterfix's Docker sandbox: for code you do
not trust, run counterfix itself inside a container.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.metrics.base import Metric, MetricResult
from counterfix.metrics.code import CODE_KEY, _signature, code_answer

_TRUTH: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
_LOCK = threading.Lock()


def _evalplus() -> Any:
    try:
        import evalplus.data
        import evalplus.evaluate
    except ImportError as exc:
        raise ImportError(
            'HumanEval+ / MBPP+ need EvalPlus: pip install "counterfix[code]"'
        ) from exc
    return evalplus


def problems(dataset: str) -> dict[str, Any]:
    """EvalPlus problems ("humaneval" or "mbpp"), downloaded and cached by EvalPlus. Does
    not compute expected outputs (that needs EvalPlus's Unix-only runner)."""
    if dataset in _TRUTH:
        return _TRUTH[dataset][0]
    ep = _evalplus()
    if dataset == "humaneval":
        return dict(ep.data.get_human_eval_plus())
    if dataset == "mbpp":
        return dict(ep.data.get_mbpp_plus())
    raise ValueError(f"unknown EvalPlus dataset {dataset!r} (humaneval or mbpp)")


def truth(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(problems, expected outputs) for a dataset; the expected outputs are computed once
    by EvalPlus from the reference solutions and cached on disk by EvalPlus."""
    with _LOCK:
        if dataset not in _TRUTH:
            ep = _evalplus()
            if dataset == "humaneval":
                probs = ep.data.get_human_eval_plus()
                digest = ep.data.get_human_eval_plus_hash()
                expected = ep.evaluate.get_groundtruth(probs, digest, [])
            elif dataset == "mbpp":
                probs = ep.data.get_mbpp_plus()
                digest = ep.data.get_mbpp_plus_hash()
                expected = ep.evaluate.get_groundtruth(
                    probs, digest, ep.evaluate.MBPP_OUTPUT_NOT_NONE_TASKS
                )
            else:
                raise ValueError(f"unknown EvalPlus dataset {dataset!r} (humaneval or mbpp)")
            _TRUTH[dataset] = (probs, expected)
        return _TRUTH[dataset]


def base_tests(problem: dict[str, Any]) -> dict[str, Any]:
    """The problem's original tests in a form :func:`counterfix.code.check_code` reads."""
    if problem.get("test"):  # HumanEval
        return {k: problem[k] for k in ("prompt", "entry_point", "test")}
    from counterfix.code.check import _split_asserts

    asserts = _split_asserts(problem.get("assertion") or "")  # MBPP+
    return {"tests": asserts, "entry_point": problem.get("entry_point")}


class EvalPlusTests(Metric):
    """1.0 when the code passes EvalPlus's tests (base and plus; ``plus=False`` for the
    original tests only). The reference is an EvalPlus problem with a ``dataset`` key."""

    __test__ = False
    name = "evalplus"
    default_threshold = 1.0
    failure_type = FailureType.CODE_FAILURE
    needs_code_execution = True

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        raise NotImplementedError

    def score(self, record: CallRecord, reference: Any = None) -> MetricResult:
        from counterfix.code import check_code
        from counterfix.code.check import as_tests, build_program
        from counterfix.code.sandbox import CodeExecutionDisabled, execution_allowed

        result = MetricResult(name=self.name, threshold=self.threshold)
        if record.error is not None:
            result.passed = False
            result.reason = f"the call failed: {record.error[:120]}"
            return result
        if not isinstance(reference, dict) or "task_id" not in reference:
            result.reason = "no EvalPlus problem to score against"
            return result
        if not execution_allowed(self.options.get("allow")):
            raise CodeExecutionDisabled()
        if os.name == "nt":
            raise RuntimeError(
                "EvalPlus's scorer needs Linux or macOS (it uses Unix signals and resource "
                "limits). On Windows run it in WSL, or use the 'humaneval' / 'mbpp' "
                "benchmarks: the original tests, scored by counterfix's checker"
            )
        dataset = str(reference.get("dataset") or _dataset_of(reference["task_id"]))
        probs, expected = truth(dataset)
        problem = probs[reference["task_id"]]
        spec = as_tests({"prompt": problem["prompt"], "entry_point": problem["entry_point"],
                         "tests": ["pass"]})  # fmt: skip
        if dataset == "mbpp":
            spec.prompt = None  # MBPP+ prompts are a description, not a code stub
        answer = code_answer(record)
        solution, _ = build_program(answer, spec)
        plus = bool(self.options.get("plus", True))
        ep = _evalplus()
        out = ep.evaluate.check_correctness(
            dataset, 0, problem, solution, expected[reference["task_id"]],
            not plus, True, record.id,
        )  # fmt: skip
        base_ok = out["base"][0] == "pass"
        plus_ok = (not plus) or out.get("plus", ("fail", []))[0] == "pass"
        passed = base_ok and plus_ok
        result.value, result.passed = float(passed), passed
        if passed:
            result.reason = "passed EvalPlus " + ("base and plus tests" if plus else "base tests")
            record.metadata[CODE_KEY] = {"cause": None}
            return result
        # Why: counterfix's own checks on the original tests give the cause.
        why = check_code(
            answer, base_tests(problem), finish_reason=record.finish_reason,
            timeout=float(self.options.get("timeout", 10.0)), allow=True,
        )  # fmt: skip
        cause, detail = why.cause, why.detail
        if why.passed:  # passes the original tests, fails EvalPlus's extra ones
            cause, detail = (
                "wrong_output",
                "passes the original tests, fails EvalPlus's extra tests",
            )
        if out["base"][0] == "timeout" or out.get("plus", ("",))[0] == "timeout":
            cause = cause if cause in ("syntax", "name", "signature") else "timeout"
        record.metadata[CODE_KEY] = {
            "cause": cause, "detail": detail, "entry_point": problem["entry_point"],
            "signature": _signature(problem["prompt"], problem["entry_point"]),
        }  # fmt: skip
        from counterfix.code import CAUSE_TEXT

        result.reason = f"[{cause}] {CAUSE_TEXT.get(cause or '', 'failed')}" + (
            f": {detail}" if detail else ""
        )
        cut = cause == "cut_off"
        result.failure_type = FailureType.TRUNCATION if cut else FailureType.CODE_FAILURE
        return result


def _dataset_of(task_id: str) -> str:
    return "mbpp" if task_id.lower().startswith("mbpp") else "humaneval"
