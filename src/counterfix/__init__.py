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
    SkipCheck,
    Usage,
    run_detectors,
)

__version__ = "0.0.1"

__all__ = [
    "CallRecord",
    "Config",
    "Detector",
    "Diagnosis",
    "Evidence",
    "FailureType",
    "Finding",
    "GuardedClient",
    "Message",
    "Prescription",
    "PrescriptionKind",
    "SkipCheck",
    "Usage",
    "__version__",
    "get_diagnosis",
    "guard",
    "record_from_openai",
    "run_detectors",
]
