"""Prompt regression tests: keep fixed failures fixed.

In your own tests:

>>> from counterfix.testing import assert_ok                         # doctest: +SKIP
>>> assert_ok(client.chat.completions.create(...))                   # doctest: +SKIP

From verified fixes (``counterfix fix calls.jsonl --out fixes.jsonl``)::

    counterfix export-tests fixes.jsonl --out tests/test_prompt_regressions.py

writes a pytest file and a ``.json`` file of cases beside it. Each case is a call that
once failed, with its verified fix applied; it passes while the failure stays gone in
2 of 3 runs (``--counterfix-runs``), so a prompt or model change cannot quietly bring it
back. Cases re-run on the provider named in the case (keys from the environment or a
``.env`` file). To run them on another model or your own server, pass
``--counterfix-caller SPEC`` (or set ``COUNTERFIX_CALLER``): ``groq:qwen/qwen3.8-27b``,
``openai:MODEL`` (honours ``OPENAI_BASE_URL``), or ``MODEL@http://host:8000/v1`` for vLLM
and other OpenAI-compatible servers; or override the ``counterfix_caller`` fixture. A
case that cannot run is skipped on your machine but fails in CI (when ``CI`` is set, or
with ``--counterfix-strict``), so a green CI run means the cases really ran. Pass
``--counterfix-judge [SPEC]`` to add judge checks (live checks only by default).

This module is also a pytest plugin (registered on install); it only adds those two
options and fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from counterfix.core.caller import ModelCaller
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord, Message


def assert_ok(target: Any, *, judge: Any = False, allow: tuple[str, ...] = ()) -> Diagnosis:
    """Fail the test if counterfix finds a model failure (F1 to F9) in this call.

    ``target`` is a guarded response, a Diagnosis or a CallRecord; ``judge`` as in
    :func:`counterfix.diagnose` (live checks only by default); ``allow`` lists failure
    codes to ignore, e.g. ``("F9",)``.
    """
    from counterfix.diagnose import diagnose

    report = target if isinstance(target, Diagnosis) and judge is False else None
    report = report or diagnose(target, judge=judge)
    bad = [f for f in report.model_failures if f.failure_type.value not in allow]
    if bad:
        raise AssertionError("counterfix found a failure:\n" + report.explain())
    return report


def case_from_fix(row: dict[str, Any]) -> dict[str, Any] | None:
    """A test case from one ``counterfix fix --out`` row with a verified fix, else None."""
    fix = row.get("fix")
    if not fix:
        return None
    record = CallRecord.model_validate(row["record"])
    messages = fix.get("messages") or [m.model_dump(exclude_none=True) for m in record.messages]
    spec = f"{record.provider}:{record.model}" if record.provider and record.model else None
    return {
        "id": f"{fix.get('failure_type', 'F?')}-{fix['name']}-{record.id[:8]}",
        "spec": spec,
        "model": record.model,
        "messages": messages,
        "params": fix.get("params") or record.params,
        "failure": fix.get("failure_type"),
        "fix": fix.get("description"),
        "metadata": record.metadata,
    }


TEST_FILE = '''"""Prompt regression tests written by `counterfix export-tests`.

Each case is a call that once failed and was fixed; it fails if the failure returns.
Cases live in {cases}. Run: pytest {name} [--counterfix-caller SPEC] [--counterfix-judge]
Without a key for a case's provider it is skipped locally and fails in CI (CI=1).
"""

from pathlib import Path

import pytest

from counterfix.testing import load_cases, run_case

CASES = load_cases(Path(__file__).with_name("{cases}"))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_prompt_case(
    case, counterfix_caller, counterfix_judge, counterfix_runs, counterfix_strict
):
    run_case(
        case,
        caller=counterfix_caller,
        judge=counterfix_judge,
        runs=counterfix_runs,
        strict=counterfix_strict,
    )
'''


def export_tests(rows: list[dict[str, Any]], out: Path) -> int:
    """Write ``out`` (a pytest file) and its cases file. Returns the number of cases."""
    cases = [c for c in (case_from_fix(r) for r in rows) if c is not None]
    cases_path = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", "utf-8")
    out.write_text(TEST_FILE.format(cases=cases_path.name, name=out.name), encoding="utf-8")
    return len(cases)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    return cases


def caller_from_spec(spec: str) -> ModelCaller:
    """A caller from ``provider:model`` or ``MODEL@http(s)://base-url`` (OpenAI-compatible;
    the key comes from ``OPENAI_API_KEY`` if set)."""
    import os

    from counterfix.presets import load_env_file, make_caller

    load_env_file(Path.cwd() / ".env")
    if "@http" in spec:
        from openai import OpenAI

        from counterfix.core.caller import OpenAICompatibleCaller

        model, base_url = spec.split("@", 1)
        client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "none")
        return OpenAICompatibleCaller(client, model, provider=base_url)
    return make_caller(spec)


def _default_caller(case: dict[str, Any]) -> ModelCaller | None:
    import os

    from counterfix.presets import PRESETS, load_env_file, make_caller

    load_env_file(Path.cwd() / ".env")
    spec = case.get("spec")
    if not spec or spec.split(":", 1)[0] not in PRESETS:
        return None
    key_var = PRESETS[spec.split(":", 1)[0]][1]
    if key_var and not os.environ.get(key_var):
        return None
    return make_caller(spec)


def run_case(
    case: dict[str, Any],
    *,
    caller: ModelCaller | None = None,
    judge: Any = False,
    runs: int = 3,
    need: int | None = None,
    strict: bool = False,
) -> None:
    """Re-run one case; fail unless no model failure shows in ``need`` of ``runs`` runs
    (default: a majority, 2 of 3)."""
    import pytest

    from counterfix.diagnose import diagnose

    caller = caller or _default_caller(case)
    if caller is None:
        import os

        why = (
            f"cannot run case {case['id']} on {case.get('spec') or case.get('model')}: no key "
            "for its provider, or it ran on a custom server. Pass --counterfix-caller SPEC "
            "(e.g. groq:qwen/qwen3.8-27b, or MODEL@http://host:8000/v1) or set COUNTERFIX_CALLER"
        )
        if strict or os.environ.get("CI"):
            pytest.fail(why)
        pytest.skip(why)
    need = need or runs // 2 + 1
    messages = [Message.model_validate(m) for m in case["messages"]]
    passed, reports = 0, []
    for _ in range(runs):
        record = caller.complete(messages, **case.get("params", {}))
        record.metadata = {**case.get("metadata", {}), **record.metadata}
        report = diagnose(record, judge=judge)
        if report.model_failures:
            reports.append(report.explain())
        else:
            passed += 1
        if passed >= need or len(reports) > runs - need:
            break
    if passed < need:
        was = f" (it was fixed by: {case['fix']})" if case.get("fix") else ""
        raise AssertionError(
            f"{case['id']}: passed {passed} of {passed + len(reports)} runs{was}\n" + reports[-1]
        )


# --- pytest plugin ----------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("counterfix")
    group.addoption(
        "--counterfix-judge",
        nargs="?",
        const="auto",
        default=None,
        help="add judge checks to counterfix cases (alone: pick a judge from your keys)",
    )
    group.addoption(
        "--counterfix-runs", type=int, default=3, help="runs per counterfix case (default 3)"
    )
    group.addoption(
        "--counterfix-caller",
        default=None,
        help="model to run counterfix cases on: provider:model, or MODEL@http://host/v1 "
        "(default: COUNTERFIX_CALLER, else each case's own provider)",
    )
    group.addoption(
        "--counterfix-strict",
        action="store_true",
        help="fail cases that cannot run instead of skipping (on by default when CI is set)",
    )


try:
    import pytest as _pytest
except ImportError:  # pragma: no cover - pytest is only needed when running tests
    _pytest = None  # type: ignore[assignment]

if _pytest is not None:

    @_pytest.fixture
    def counterfix_judge(request: Any) -> Any:
        """The judge for counterfix cases: False (live checks only) unless asked for."""
        choice = request.config.getoption("--counterfix-judge")
        return False if choice is None else (None if choice == "auto" else choice)

    @_pytest.fixture
    def counterfix_caller(request: Any) -> ModelCaller | None:
        """``--counterfix-caller`` / ``COUNTERFIX_CALLER``, else None (each case's own
        provider). Override it to run cases through your own client."""
        import os

        spec = request.config.getoption("--counterfix-caller") or os.environ.get(
            "COUNTERFIX_CALLER"
        )
        return caller_from_spec(spec) if spec else None

    @_pytest.fixture
    def counterfix_strict(request: Any) -> bool:
        strict: bool = request.config.getoption("--counterfix-strict")
        return strict

    @_pytest.fixture
    def counterfix_runs(request: Any) -> int:
        runs: int = request.config.getoption("--counterfix-runs")
        return runs
