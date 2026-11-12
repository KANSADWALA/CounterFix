"""Code evaluation: sandbox, checks and their causes, pass@k, the tests metric, F12 fixes."""

from __future__ import annotations

from typing import Any

import pytest

from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType
from counterfix.detect.code import (
    CodeExecutionDisabled,
    as_tests,
    check_code,
    mean_pass_at_k,
    pass_at_k,
    run_python,
)
from counterfix.evals import evaluate
from counterfix.metrics import get_metric

HUMANEVAL = {
    "task_id": "HumanEval/0",
    "prompt": ('from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n'),
    "entry_point": "add",
    "test": (
        "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n"
    ),
}
MBPP = {"test_list": ["assert sq(2) == 4", "assert sq(3) == 9", "assert sq(-1) == 1"]}


@pytest.fixture(autouse=True)
def _allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COUNTERFIX_ALLOW_CODE", "1")
    monkeypatch.setenv("COUNTERFIX_SANDBOX", "subprocess")


def _check(answer: str, ref: Any = HUMANEVAL, **kw: Any):
    return check_code(answer, ref, timeout=5, **kw)


@pytest.mark.parametrize(
    ("answer", "cause"),
    [
        ("    return a + b\n", None),  # a body that continues the prompt
        ("```python\ndef add(a, b):\n    return a + b\n```", None),
        ("```python\ndef add(a):\n    return a\n```", "signature"),
        ("```python\ndef plus(a, b):\n    return a + b\n```", "name"),
        ("```python\ndef add(a, b):\n    return a - b\n```", "wrong_output"),
        ("```python\ndef add(a, b):\n    return 1 / 0\n```", "runtime"),
        ("```python\ndef add(a, b):\n    while True:\n        pass\n```", "timeout"),
        ("```python\ndef add(a, b:\n```", "syntax"),
        ("I cannot help with that.", "no_code"),
        ("```python\nraise SystemExit(3)\ndef add(a, b): return a + b\n```", "runtime"),
    ],
)
def test_causes(answer: str, cause: str | None) -> None:
    result = _check(answer)
    assert result.cause == cause, result.why
    assert result.passed is (cause is None)


def test_cut_off_code_is_f1_not_syntax() -> None:
    assert (
        _check("```python\ndef add(a, b):\n    return (a +", finish_reason="length").cause
        == "cut_off"
    )


def test_mbpp_counts_each_assert() -> None:
    result = _check("def sq(x): return x * x if x > 0 else 0", MBPP)
    assert (result.tests_passed, result.tests_total, result.cause) == (2, 3, "wrong_output")
    assert "sq(-1)" in result.detail


def test_failed_assert_line_named_for_humaneval() -> None:
    assert "candidate(1, 2) == 3" in _check("    return a - b\n").detail


def test_imports_allow_list() -> None:
    code = "```python\nimport numpy\ndef sq(x): return x * x\n```"
    assert _check(code, MBPP, allowed_imports=[]).cause == "import"


def test_test_formats() -> None:
    assert as_tests(["assert f() == 1"]).tests == ["assert f() == 1"]
    assert as_tests("assert f() == 1\nassert f() != 2").tests == [
        "assert f() == 1",
        "assert f() != 2",
    ]
    he = as_tests(HUMANEVAL)
    assert he.tests == ["check(add)"] and "def check" in he.setup and he.entry_point == "add"
    mb = as_tests({"test_list": ["assert g()"], "test_setup_code": "X = 1"})
    assert mb.setup == "X = 1"


def test_execution_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COUNTERFIX_ALLOW_CODE")
    with pytest.raises(CodeExecutionDisabled):
        run_python("print(1)")
    with pytest.raises(CodeExecutionDisabled):
        evaluate(
            [{"prompt": "x", "response": "y", "reference": ["assert True"]}], metrics=["tests"]
        )
    assert run_python("print(1)", allow=True).stdout.strip() == "1"


def test_sandbox_hides_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    run = run_python("import os; print(os.environ.get('GROQ_API_KEY'))")
    assert run.stdout.strip() == "None"


def test_pass_at_k_matches_the_paper() -> None:
    import math

    def naive(n: int, c: int, k: int) -> float:
        return 1 - math.comb(n - c, k) / math.comb(n, k)

    for n, c, k in [(10, 3, 1), (10, 3, 5), (20, 0, 5), (5, 5, 1), (200, 17, 100)]:
        assert pass_at_k(n, c, k) == pytest.approx(naive(n, c, k))
    assert mean_pass_at_k([(10, 3), (10, 0)], 1) == pytest.approx(0.15)
    assert mean_pass_at_k([(3, 1)], 5) is None
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_tests_metric_and_f12() -> None:
    m = get_metric("tests")
    rec = CallRecord(response_text="```python\ndef add(a, b):\n    return a - b\n```")
    r = m.score(rec, HUMANEVAL)
    assert r.passed is False and r.failure_type is FailureType.CODE_FAILURE
    assert r.reason.startswith("[wrong_output]")
    assert rec.metadata["code_check"]["signature"] == "def add(a: int, b: int) -> int:"
    cut = CallRecord(
        response_text="```python\ndef add(a, b):\n    return (", finish_reason="length"
    )
    assert m.score(cut, HUMANEVAL).failure_type is FailureType.TRUNCATION


class _Coder:
    """Writes wrong code first; right code once it sees its error."""

    name = "fake:coder"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Any], **params: Any) -> CallRecord:
        self.calls += 1
        msgs = [Message.model_validate(m) if isinstance(m, dict) else m for m in messages]
        saw_error = any("Your code failed" in m.text for m in msgs if m.role == "user")
        body = "a + b" if saw_error else "a - b"
        return CallRecord(
            model="coder", messages=msgs, params=params, finish_reason="stop",
            response_text=f"```python\ndef add(a, b):\n    return {body}\n```",
        )  # fmt: skip


def test_eval_finds_f12_and_verifies_error_feedback() -> None:
    rows = [{"id": "he0", "prompt": HUMANEVAL["prompt"], "reference": HUMANEVAL}]
    result = evaluate(rows, model=_Coder(), metrics=["tests"], fix=1, max_runs=20)
    item = result.items[0]
    assert not item.passed and item.causes[0] == "F12"
    assert result.code_causes == {"wrong_output": 1}
    assert item.fix is not None and item.fix.fix is not None
    assert item.fix.fix.name == "error_feedback"
    assert "code that failed its tests" in result.text()


def test_samples_report_pass_at_k() -> None:
    state = {"n": 0}

    class Flaky:
        name = "fake:flaky"

        def complete(self, messages: list[Any], **params: Any) -> CallRecord:
            state["n"] += 1
            body = "a + b" if state["n"] % 2 else "a - b"
            return CallRecord(response_text=f"```python\ndef add(a, b):\n    return {body}\n```")

    rows = [{"id": "he0", "prompt": "add", "reference": HUMANEVAL}]
    result = evaluate(rows, model=Flaky(), metrics=["tests"], samples=4)
    assert [i.id for i in result.items] == ["he0#1", "he0#2", "he0#3", "he0#4"]
    assert result.pass_at_k["tests"][1] == pytest.approx(0.5)
    assert "pass@1 0.500" in result.text()
