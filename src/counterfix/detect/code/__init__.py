"""Code evaluation: run a model's code against tests in a sandbox, find why it failed
(F12 and its sub-causes), and estimate pass@k.

>>> from counterfix.detect.code import check_code
>>> check_code(answer, {"test_list": ["assert add(1, 2) == 3"]}, allow=True)  # doctest: +SKIP
CodeResult(passed=True, ...)

Running model-written code is off until you turn it on; see :mod:`counterfix.detect.code.sandbox`.
"""

from counterfix.detect.code.check import (
    CAUSE_TEXT,
    CodeResult,
    CodeTests,
    as_tests,
    build_program,
    check_code,
)
from counterfix.detect.code.passk import mean_pass_at_k, pass_at_k
from counterfix.detect.code.sandbox import (
    ALLOW_ENV,
    CodeExecutionDisabled,
    RunResult,
    docker_available,
    execution_allowed,
    run_python,
)

__all__ = [
    "ALLOW_ENV",
    "CAUSE_TEXT",
    "CodeExecutionDisabled",
    "CodeResult",
    "CodeTests",
    "RunResult",
    "as_tests",
    "build_program",
    "check_code",
    "docker_available",
    "execution_allowed",
    "mean_pass_at_k",
    "pass_at_k",
    "run_python",
]
