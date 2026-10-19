from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import counterfix
from counterfix import CallRecord, Message
from counterfix.testing import assert_ok, export_tests, load_cases, run_case
from test_engine import Scripted, cut_off

pytest_plugins = ["pytester"]


def test_assert_ok_passes_good_calls_and_explains_bad_ones() -> None:
    good = CallRecord(messages=[Message(role="user", content="Hi")], response_text="Hello!")
    assert assert_ok(good).ok
    with pytest.raises(AssertionError, match="F1 Truncation"):
        assert_ok(cut_off())
    assert_ok(cut_off(), allow=("F1",))


def _fixed_rows() -> list[dict[str, Any]]:
    model = Scripted(
        lambda m, p: (
            "A, B." if p["max_tokens"] > 20 else "A",
            "stop" if p["max_tokens"] > 20 else "length",
        )
    )
    record = cut_off()
    record.provider = "groq"
    result = counterfix.fix(record, judge=False, caller=model)
    assert result.fix is not None
    return [
        {"record": record.model_dump(mode="json"), "fix": result.fix.model_dump(mode="json")},
        {"record": record.model_dump(mode="json"), "fix": None},  # unfixed rows are skipped
    ]


def test_export_tests_writes_cases_with_the_fix_applied(tmp_path: Path) -> None:
    out = tmp_path / "test_regressions.py"
    assert export_tests(_fixed_rows(), out) == 1
    case = load_cases(out.with_suffix(".json"))[0]
    assert case["spec"] == "groq:m" and case["failure"] == "F1"
    assert case["params"]["max_tokens"] == 80
    assert "run_case" in out.read_text(encoding="utf-8")

    good = Scripted(lambda m, p: ("A, B.", "stop"))
    run_case(case, caller=good)
    assert len(good.calls) == 2  # settled after 2 passes
    with pytest.raises(AssertionError, match="passed 0 of 2"):
        run_case(case, caller=Scripted(lambda m, p: ("A", "length")))


def test_generated_file_runs_under_pytest(pytester: pytest.Pytester) -> None:
    out = pytester.path / "test_regressions.py"
    export_tests(_fixed_rows(), out)
    cases = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    cases[0]["spec"] = None  # no provider: skipped unless a caller fixture is given
    out.with_suffix(".json").write_text(json.dumps(cases), encoding="utf-8")
    pytester.makeconftest(
        """
import pytest
from test_engine import Scripted

@pytest.fixture
def counterfix_caller():
    return Scripted(lambda m, p: ("A, B.", "stop"))
"""
    )
    pytester.syspathinsert(Path(__file__).parent)
    result = pytester.runpytest("--counterfix-runs", "1")  # plugin loads from its entry point
    result.assert_outcomes(passed=1)
