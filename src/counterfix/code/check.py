"""Checking a model's code against tests, cheapest check first.

1. extract: take the code block from the answer (``no_code`` when there is none);
2. parse: ``ast.parse`` (``syntax``; ``cut_off`` when the answer stopped at the limit);
3. name and signature: the function the tests call exists with a compatible signature
   (``name``, ``signature``);
4. imports: only allowed modules, when an allow-list is given (``import``);
5. run the tests in the sandbox: ``timeout``, ``runtime`` (an exception that is not a
   failed assert) or ``wrong_output`` (a failed assert).

Test formats: HumanEval (``test`` code defining ``check(candidate)`` + ``entry_point``),
MBPP (a list of ``assert`` lines, optional setup), or plain test code / assert lines.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from counterfix.code.sandbox import RunResult, Sandbox, run_python
from counterfix.metrics.extract import extract_code

Cause = Literal[
    "no_code", "syntax", "cut_off", "name", "signature", "import", "runtime",
    "wrong_output", "timeout",
]  # fmt: skip
CAUSE_TEXT: dict[str, str] = {
    "no_code": "no code in the answer",
    "syntax": "the code does not parse",
    "cut_off": "the code stops mid-way (the answer hit its token limit)",
    "name": "the function the tests call is missing or misnamed",
    "signature": "the function takes different arguments than asked",
    "import": "the code imports a module that is not allowed",
    "runtime": "the code raises an error when run",
    "wrong_output": "the code runs but gives wrong results",
    "timeout": "the code did not finish in time",
}
_MARK = "<<COUNTERFIX-RESULTS>>"
_LENGTH = frozenset({"length", "max_tokens", "max_output_tokens"})


class CodeTests(BaseModel):
    """What to check code against."""

    tests: list[str]
    """Test snippets, each run on its own (an assert line, or ``check(f)``)."""
    setup: str = ""
    """Code run once before the tests (helpers, imports, HumanEval's ``check``)."""
    entry_point: str | None = None
    prompt: str | None = None
    """The code prompt (HumanEval): a completion that is only the body is appended to it."""


class CodeResult(BaseModel):
    """The verdict on one piece of code."""

    passed: bool
    cause: Cause | None = None
    detail: str = ""
    tests_passed: int = 0
    tests_total: int = 0
    code: str = ""
    run: RunResult | None = None

    @property
    def why(self) -> str:
        if self.passed:
            return f"passed {self.tests_passed}/{self.tests_total} tests"
        text = CAUSE_TEXT.get(self.cause or "", "failed")
        return f"{text}: {self.detail}" if self.detail else text


def as_tests(reference: Any) -> CodeTests:
    """Read tests from a HumanEval/MBPP/EvalPlus row, a dict, a list of asserts or code."""
    if isinstance(reference, CodeTests):
        return reference
    if isinstance(reference, str):
        return CodeTests(tests=_split_asserts(reference) or [reference])
    if isinstance(reference, Sequence):
        return CodeTests(tests=[str(t) for t in reference])
    if not isinstance(reference, Mapping):
        raise ValueError("code tests: give test code, a list of asserts, or a dict")
    entry = reference.get("entry_point")
    setup_parts = [
        str(reference.get(k) or "") for k in ("test_imports", "test_setup_code", "setup")
    ]
    if isinstance(reference.get("test_imports"), list):
        setup_parts[0] = "\n".join(reference["test_imports"])
    tests: list[str]
    raw = reference.get("test_list") or reference.get("tests")
    if isinstance(raw, list):
        tests = [str(t) for t in raw]
        extra = reference.get("challenge_test_list")
        if isinstance(extra, list):
            tests += [str(t) for t in extra]
    elif isinstance(raw, str) and entry and "def check" in raw:
        setup_parts.append(raw)
        tests = [f"check({entry})"]
    elif isinstance(reference.get("test"), str):
        code = reference["test"]
        if entry and "def check" in code:  # HumanEval / EvalPlus
            setup_parts.append(code)
            tests = [f"check({entry})"]
        else:
            tests = _split_asserts(code) or [code]
    elif isinstance(raw, str):
        tests = _split_asserts(raw) or [raw]
    else:
        raise ValueError("code tests: no 'test', 'tests' or 'test_list' field")
    return CodeTests(
        tests=tests,
        setup="\n".join(p for p in setup_parts if p.strip()),
        entry_point=entry,
        prompt=reference.get("prompt") if isinstance(reference.get("prompt"), str) else None,
    )


def _split_asserts(code: str) -> list[str]:
    """Top-level ``assert`` statements as separate tests, when the code is only asserts."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    if not tree.body or not all(isinstance(s, ast.Assert) for s in tree.body):
        return []
    return [ast.get_source_segment(code, s) or "" for s in tree.body]


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _arity(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int | None]:
    """(required positional arguments, maximum or None for *args)."""
    args = fn.args
    positional = len(args.posonlyargs) + len(args.args)
    required = positional - len(args.defaults)
    return required, None if args.vararg else positional


def build_program(answer: str, spec: CodeTests) -> tuple[str, str]:
    """(the solution code, the full program with the test harness)."""
    code = answer_code(answer)
    solution = code
    if spec.prompt and spec.entry_point:
        whole = f"def {spec.entry_point}(" in code or _top_level(code)
        if not whole:  # only the body (indented): it continues the prompt
            solution = spec.prompt + code
        else:  # a whole function: keep the prompt's imports and helpers above it
            solution = _prompt_header(spec.prompt, spec.entry_point) + code
    harness = (
        "\n\n# --- counterfix test harness ---\n"
        "import json as _cf_json, linecache as _cf_lc, traceback as _cf_tb\n"
        "def _cf_compile(_cf_text, _cf_name):\n"
        "    _cf_lc.cache[_cf_name] = (len(_cf_text), None, _cf_text.splitlines(True), _cf_name)\n"
        "    return compile(_cf_text, _cf_name, 'exec')\n"
        f"exec(_cf_compile({spec.setup!r}, '<setup>'), globals())\n"
        "def _cf_actual(_cf_line, _cf_trace):\n"
        "    # 'assert f(x) == y' failed: what did f(x) give? (re-evaluated in the test's frame)\n"
        "    import ast as _cf_ast\n"
        "    try:\n"
        "        _cf_node = _cf_ast.parse(_cf_line.strip()).body[0]\n"
        "        _cf_test = _cf_node.test\n"
        "        if not isinstance(_cf_test, _cf_ast.Compare):\n"
        "            return ''\n"
        "        while _cf_trace.tb_next is not None:\n"
        "            _cf_trace = _cf_trace.tb_next\n"
        "        _cf_frame = _cf_trace.tb_frame\n"
        "        _cf_expr = _cf_ast.unparse(_cf_test.left)\n"
        "        _cf_value = eval(_cf_expr, _cf_frame.f_globals, dict(_cf_frame.f_locals))\n"
        "        return ' (got ' + repr(_cf_value)[:200] + ')'\n"
        "    except Exception:\n"
        "        return ''\n"
        "_cf_results = []\n"
        f"for _cf_i, _cf_src in enumerate({spec.tests!r}):\n"
        "    try:\n"
        "        exec(_cf_compile(_cf_src, f'<test {_cf_i + 1}>'), globals())\n"
        "        _cf_results.append(['pass', ''])\n"
        "    except AssertionError as _cf_e:\n"
        "        _cf_at = _cf_tb.extract_tb(_cf_e.__traceback__)[-1].line or _cf_src\n"
        "        _cf_got = _cf_actual(_cf_at, _cf_e.__traceback__)\n"
        "        _cf_said = (_cf_at.strip() + ' ' + str(_cf_e))[:300]\n"
        "        _cf_results.append(['assert', _cf_said + _cf_got])\n"
        "    except Exception as _cf_e:\n"
        "        _cf_lines = _cf_tb.format_exception(type(_cf_e), _cf_e, _cf_e.__traceback__)\n"
        "        _cf_results.append(['error', ''.join(_cf_lines[-3:])[-800:]])\n"
        f"print({_MARK!r} + _cf_json.dumps(_cf_results))\n"
    )
    return solution, solution + harness


def answer_code(answer: str) -> str:
    """The code in an answer: the fenced block, else the text as it is (indentation kept,
    so a bare function body still continues a prompt)."""
    if "```" in answer:
        return extract_code(answer)
    lines = answer.rstrip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _top_level(code: str) -> bool:
    """True when the code has statements at column 0 (a module, not a function body)."""
    return any(ln and not ln[0].isspace() and not ln.startswith("#") for ln in code.splitlines())


_CODE_HINT = re.compile(
    r"^\s*(def |class |return\b|import |from \w+ import|for |while |if )|=", re.M
)


def _prompt_header(prompt: str, entry_point: str) -> str:
    """The prompt's imports and helper functions, without the stub being solved."""
    try:
        tree = ast.parse(prompt + "\n    pass\n")
    except SyntaxError:
        return ""
    keep = [
        ast.get_source_segment(prompt + "\n    pass\n", n) or ""
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != entry_point)
    ]
    return "\n".join(keep) + ("\n\n" if keep else "")


def _imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def check_code(
    answer: str,
    reference: Any,
    *,
    finish_reason: str | None = None,
    allowed_imports: Sequence[str] | None = None,
    timeout: float = 10.0,
    sandbox: Sandbox | None = None,
    allow: bool | None = None,
) -> CodeResult:
    """Check an answer's code against tests (see the module docstring for the steps).

    ``allowed_imports``: modules allowed beyond the standard library; None skips the
    import check. Raises :class:`~counterfix.code.sandbox.CodeExecutionDisabled` when the
    tests would run and execution is off.
    """
    spec = as_tests(reference)
    total = len(spec.tests)
    if not answer_code(answer).strip():
        return CodeResult(passed=False, cause="no_code", tests_total=total)
    solution, program = build_program(answer, spec)
    cut = (finish_reason or "") in _LENGTH
    try:
        tree = ast.parse(solution)
    except SyntaxError as exc:
        if "```" not in answer and not _CODE_HINT.search(answer):
            return CodeResult(passed=False, cause="no_code", tests_total=total)
        where = f"line {exc.lineno}: {exc.msg}"
        if cut:
            where = "the answer stopped at its token limit before the code was complete"
        return CodeResult(
            passed=False, cause="cut_off" if cut else "syntax", detail=where,
            tests_total=total, code=solution,
        )  # fmt: skip
    if spec.entry_point:
        fns = _functions(tree)
        fn = fns.get(spec.entry_point)
        if fn is None and not _assigned(tree, spec.entry_point):
            near = ", ".join(sorted(fns)) or "none"
            return CodeResult(
                passed=False, cause="name", tests_total=total, code=solution,
                detail=f"no function {spec.entry_point}() (defined: {near})",
            )  # fmt: skip
        want = _wanted(spec)
        if fn is not None and want is not None:
            got = _arity(fn)
            if not _compatible(got, want):
                return CodeResult(
                    passed=False, cause="signature", tests_total=total, code=solution,
                    detail=f"{spec.entry_point}() takes {_say(got)}, the prompt asks for "
                    f"{_say(want)}",
                )  # fmt: skip
    if allowed_imports is not None:
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        bad = sorted(_imports(tree) - stdlib - set(allowed_imports))
        if bad:
            return CodeResult(
                passed=False, cause="import", detail=", ".join(bad), tests_total=total,
                code=solution,
            )  # fmt: skip
    run = run_python(program, timeout=timeout, sandbox=sandbox, allow=allow)
    return _verdict(run, total, solution)


def _assigned(tree: ast.Module, name: str) -> bool:
    """``name = lambda ...`` or a class also count as defining it."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return True
    return False


def _wanted(spec: CodeTests) -> tuple[int, int | None] | None:
    if not spec.prompt or not spec.entry_point:
        return None
    try:
        fn = _functions(ast.parse(spec.prompt + "\n    pass\n")).get(spec.entry_point)
    except SyntaxError:
        return None
    return _arity(fn) if fn is not None else None


def _compatible(got: tuple[int, int | None], want: tuple[int, int | None]) -> bool:
    """The tests call with between want's required and maximum arguments."""
    lo, hi = got
    need_lo, need_hi = want
    top = need_hi if need_hi is not None else need_lo
    return lo <= need_lo and (hi is None or hi >= top)


def _say(arity: tuple[int, int | None]) -> str:
    lo, hi = arity
    if hi is None:
        return f"{lo}+ arguments"
    return f"{lo} argument(s)" if lo == hi else f"{lo} to {hi} arguments"


def _verdict(run: RunResult, total: int, solution: str) -> CodeResult:
    if run.status == "timeout":
        return CodeResult(
            passed=False, cause="timeout", detail=f"over {run.duration_ms / 1000:.0f}s",
            tests_total=total, code=solution, run=run,
        )  # fmt: skip
    line = next((ln for ln in reversed(run.stdout.splitlines()) if ln.startswith(_MARK)), None)
    if line is None:  # the program died before the tests ran: its own top-level code
        tail = run.stderr.strip().splitlines()[-1:] or [f"exit code {run.returncode}"]
        cause: Cause = "timeout" if "TimeoutError: time limit" in run.stderr else "runtime"
        if "MemoryError" in run.stderr:
            tail = ["MemoryError (over the memory limit)"]
        return CodeResult(
            passed=False, cause=cause, detail=tail[0][:300], tests_total=total,
            code=solution, run=run,
        )  # fmt: skip
    results = json.loads(line[len(_MARK) :])
    passed = sum(1 for kind, _ in results if kind == "pass")
    first_error = next((d for kind, d in results if kind == "error"), None)
    first_assert = next((d for kind, d in results if kind == "assert"), None)
    if passed == total:
        return CodeResult(
            passed=True, tests_passed=passed, tests_total=total, code=solution, run=run
        )
    if first_error is not None and _is_assert(first_error):
        first_assert, first_error = first_error, None
    cause = "runtime" if first_error is not None else "wrong_output"
    detail = (first_error or first_assert or "").strip().splitlines()
    return CodeResult(
        passed=False, cause=cause, detail=(detail[-1] if detail else "")[:300],
        tests_passed=passed, tests_total=total, code=solution, run=run,
    )  # fmt: skip


def _is_assert(text: str) -> bool:
    return "AssertionError" in text.strip().splitlines()[-1] if text.strip() else False
