"""counterfix: find why an LLM call failed and get a fix that was tested to work."""

from counterfix import background
from counterfix.adapters import GuardedClient, get_diagnosis, guard, record_from_openai, watch
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
    Usage,
    run_detectors,
)
from counterfix.diagnose import diagnose
from counterfix.judges.auto import NoJudgeError, SelfJudgeWarning, configure
from counterfix.lint import LintIssue, lint
from counterfix.report import load_log, log_to
from counterfix.rules import FixRule, RuleStore

__version__ = "0.1.0"

__all__ = [
    "CallRecord",
    "Config",
    "Detector",
    "Diagnosis",
    "Evidence",
    "FailureType",
    "Finding",
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
    "Usage",
    "__version__",
    "background",
    "configure",
    "diagnose",
    "get_diagnosis",
    "guard",
    "lint",
    "load_log",
    "log_to",
    "record_from_openai",
    "run_detectors",
    "watch",
]
