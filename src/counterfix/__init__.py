"""counterfix: find why an LLM call failed and get a fix that was tested to work."""

from counterfix import background, metrics, otel, perf
from counterfix.adapters import (
    GuardedClient,
    get_diagnosis,
    guard,
    langchain_callback,
    litellm_callback,
    record_from_openai,
    record_from_responses,
    watch,
)
from counterfix.context import context
from counterfix.core import (
    CallRecord,
    Config,
    Detector,
    Diagnosis,
    Evidence,
    FailureType,
    Finding,
    Message,
    Prescription,
    PrescriptionKind,
    RetryAttempt,
    SkipCheck,
    Timing,
    Usage,
    hooks,
    run_detectors,
)
from counterfix.diagnose import diagnose
from counterfix.engine import FixAttempt, FixResult, fix
from counterfix.evals import EvalResult, evaluate
from counterfix.judges.auto import NoJudgeError, SelfJudgeWarning, configure
from counterfix.lint import LintIssue, lint
from counterfix.report import load_log, log_to
from counterfix.rules import FixRule, RuleStore

__version__ = "0.2.0"

__all__ = [
    "CallRecord",
    "Config",
    "Detector",
    "Diagnosis",
    "EvalResult",
    "Evidence",
    "FailureType",
    "Finding",
    "FixAttempt",
    "FixResult",
    "FixRule",
    "GuardedClient",
    "LintIssue",
    "Message",
    "NoJudgeError",
    "Prescription",
    "PrescriptionKind",
    "RetryAttempt",
    "RuleStore",
    "SelfJudgeWarning",
    "SkipCheck",
    "Timing",
    "Usage",
    "__version__",
    "background",
    "configure",
    "context",
    "diagnose",
    "evaluate",
    "fix",
    "get_diagnosis",
    "guard",
    "hooks",
    "langchain_callback",
    "lint",
    "litellm_callback",
    "load_log",
    "log_to",
    "metrics",
    "otel",
    "perf",
    "record_from_openai",
    "record_from_responses",
    "run_detectors",
    "watch",
]
