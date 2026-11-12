"""counterfix: find out why an LLM call failed, and prove the fix works.

The everyday API is the names in ``__all__``: wrap a client with :func:`guard` (or a
function with :func:`watch`), read the :class:`Diagnosis` on each response, and call
:func:`fix` for a change that was re-run and beat a plain retry.

Everything else lives in a subpackage (``counterfix.core``, ``counterfix.adapters``,
``counterfix.evals``, ...). The names that 0.2 exported here still import from
``counterfix`` for compatibility; they are loaded on first use, so ``import counterfix``
stays light.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from counterfix import _compat

_compat.install()  # old 0.2 module paths keep importing (see counterfix._compat)

# The core loop, loaded eagerly. ``context``, ``diagnose`` and ``lint`` are also the names of
# submodules, so they must be bound here before any submodule import can shadow them.
from counterfix.adapters import guard, watch  # noqa: E402
from counterfix.context import context  # noqa: E402
from counterfix.core import CallRecord, Config, Diagnosis, FailureType, Message  # noqa: E402
from counterfix.diagnose import diagnose  # noqa: E402
from counterfix.fixes.engine import FixResult, fix  # noqa: E402
from counterfix.judges.auto import configure  # noqa: E402
from counterfix.lint import lint  # noqa: E402
from counterfix.report import log_to  # noqa: E402

__version__ = "0.2.1"

__all__ = [
    "CallRecord",
    "Config",
    "Diagnosis",
    "FailureType",
    "FixResult",
    "Message",
    "__version__",
    "configure",
    "diagnose",
    "fix",
    "guard",
    "log_to",
    "watch",
]

# Exported here in 0.2; still importable from ``counterfix`` but loaded lazily.
# name -> (module, attribute); attribute None means the module itself.
_LAZY: dict[str, tuple[str, str | None]] = {
    # integrations
    "GuardedClient": ("counterfix.adapters", "GuardedClient"),
    "get_diagnosis": ("counterfix.adapters", "get_diagnosis"),
    "langchain_callback": ("counterfix.adapters", "langchain_callback"),
    "litellm_callback": ("counterfix.adapters", "litellm_callback"),
    "record_from_openai": ("counterfix.adapters", "record_from_openai"),
    "record_from_responses": ("counterfix.adapters", "record_from_responses"),
    # core building blocks
    "Detector": ("counterfix.core", "Detector"),
    "Evidence": ("counterfix.core", "Evidence"),
    "Finding": ("counterfix.core", "Finding"),
    "Prescription": ("counterfix.core", "Prescription"),
    "PrescriptionKind": ("counterfix.core", "PrescriptionKind"),
    "RetryAttempt": ("counterfix.core", "RetryAttempt"),
    "SkipCheck": ("counterfix.core", "SkipCheck"),
    "Timing": ("counterfix.core", "Timing"),
    "Usage": ("counterfix.core", "Usage"),
    "hooks": ("counterfix.core", "hooks"),
    "run_detectors": ("counterfix.core", "run_detectors"),
    # fixes
    "FixAttempt": ("counterfix.fixes.engine", "FixAttempt"),
    "FixRule": ("counterfix.fixes", "FixRule"),
    "RuleStore": ("counterfix.fixes", "RuleStore"),
    # judges, lint, runtime
    "NoJudgeError": ("counterfix.judges.auto", "NoJudgeError"),
    "SelfJudgeWarning": ("counterfix.judges.auto", "SelfJudgeWarning"),
    "LintIssue": ("counterfix.lint", "LintIssue"),
    "load_log": ("counterfix.report", "load_log"),
    # evaluation
    "EvalResult": ("counterfix.evals", "EvalResult"),
    "evaluate": ("counterfix.evals", "evaluate"),
    # modules
    "background": ("counterfix.background", None),
    "metrics": ("counterfix.metrics", None),
    "otel": ("counterfix.otel", None),
    "perf": ("counterfix.perf", None),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'counterfix' has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value  # later lookups skip this function
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


if TYPE_CHECKING:
    from counterfix import background, metrics, otel, perf
    from counterfix.adapters import (
        GuardedClient,
        get_diagnosis,
        langchain_callback,
        litellm_callback,
        record_from_openai,
        record_from_responses,
    )
    from counterfix.core import (
        Detector,
        Evidence,
        Finding,
        Prescription,
        PrescriptionKind,
        RetryAttempt,
        SkipCheck,
        Timing,
        Usage,
        hooks,
        run_detectors,
    )
    from counterfix.evals import EvalResult, evaluate
    from counterfix.fixes import FixRule, RuleStore
    from counterfix.fixes.engine import FixAttempt
    from counterfix.judges.auto import NoJudgeError, SelfJudgeWarning
    from counterfix.lint import LintIssue
    from counterfix.report import load_log
