"""Built-in detectors.

Live detectors are cheap: they make no LLM calls and run on every guarded call.
"""

from __future__ import annotations

from counterfix.core.detector import Detector
from counterfix.detectors.basic import EmptyOutputDetector, ProviderFilterDetector
from counterfix.detectors.fmt import FormatDetector, expected_format
from counterfix.detectors.instructions import InstructionCheck, extract_constraints
from counterfix.detectors.judged import (
    AmbiguityCheck,
    ComputationCheck,
    DroppedInstructionCheck,
    FalsePremiseCheck,
    HallucinationCheck,
    OverRefusalCheck,
    judge_detectors,
)
from counterfix.detectors.refusal import RefusalDetector
from counterfix.detectors.truncation import TruncationDetector


def default_detectors() -> list[Detector]:
    """The live detectors ``guard()`` uses when none are passed. None makes an LLM call."""
    return [
        ProviderFilterDetector(),
        EmptyOutputDetector(),
        TruncationDetector(),
        FormatDetector(),
        RefusalDetector(),
        InstructionCheck(),
    ]


__all__ = [
    "AmbiguityCheck",
    "ComputationCheck",
    "DroppedInstructionCheck",
    "EmptyOutputDetector",
    "FalsePremiseCheck",
    "FormatDetector",
    "HallucinationCheck",
    "InstructionCheck",
    "OverRefusalCheck",
    "ProviderFilterDetector",
    "RefusalDetector",
    "TruncationDetector",
    "default_detectors",
    "expected_format",
    "extract_constraints",
    "judge_detectors",
]
