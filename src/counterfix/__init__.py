"""counterfix: find why an LLM call failed and get a fix that was tested to work."""

from counterfix.adapters import GuardedClient, get_diagnosis, guard, record_from_openai
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
from counterfix.rules import FixRule, RuleStore

__version__ = "0.0.1"

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
    "configure",
    "diagnose",
    "get_diagnosis",
    "guard",
    "lint",
    "record_from_openai",
    "run_detectors",
]
