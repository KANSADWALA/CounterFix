"""Built-in detectors, and the helpers they read (retrieval, agent runs, code, confidence).

Live detectors are cheap: they make no LLM calls and run on every guarded call. Each
detector loads on first use, so importing one helper does not load them all.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from counterfix.core.detector import Detector

_LAZY: dict[str, str] = {
    "AgentTaskCheck": "counterfix.detect.tools",
    "AmbiguityCheck": "counterfix.detect.judged",
    "CanaryCheck": "counterfix.detect.injection",
    "ComputationCheck": "counterfix.detect.judged",
    "ContextUseCheck": "counterfix.detect.rag",
    "DroppedInstructionCheck": "counterfix.detect.judged",
    "EmptyOutputDetector": "counterfix.detect.basic",
    "FalsePremiseCheck": "counterfix.detect.judged",
    "FormatDetector": "counterfix.detect.fmt",
    "FragilityCheck": "counterfix.detect.fragility",
    "HallucinationCheck": "counterfix.detect.judged",
    "InjectionCheck": "counterfix.detect.injection",
    "InstructionCheck": "counterfix.detect.instructions",
    "OverRefusalCheck": "counterfix.detect.judged",
    "ProviderFilterDetector": "counterfix.detect.basic",
    "RefusalDetector": "counterfix.detect.refusal",
    "RetrievalLabelCheck": "counterfix.detect.rag",
    "ToolUseCheck": "counterfix.detect.tools",
    "TruncationDetector": "counterfix.detect.truncation",
    "expected_format": "counterfix.detect.fmt",
    "extract_constraints": "counterfix.detect.instructions",
    "judge_detectors": "counterfix.detect.judged",
}


def __getattr__(name: str) -> Any:
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'counterfix.detect' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


def default_detectors() -> list[Detector]:
    """The live detectors ``guard()`` uses when none are passed. None makes an LLM call."""
    from counterfix.detect.basic import EmptyOutputDetector, ProviderFilterDetector
    from counterfix.detect.fmt import FormatDetector
    from counterfix.detect.injection import CanaryCheck
    from counterfix.detect.instructions import InstructionCheck
    from counterfix.detect.rag import RetrievalLabelCheck
    from counterfix.detect.refusal import RefusalDetector
    from counterfix.detect.tools import ToolUseCheck
    from counterfix.detect.truncation import TruncationDetector

    return [
        ProviderFilterDetector(),
        EmptyOutputDetector(),
        TruncationDetector(),
        FormatDetector(),
        RefusalDetector(),
        InstructionCheck(),
        RetrievalLabelCheck(),
        CanaryCheck(),
        ToolUseCheck(),
    ]


__all__ = [
    "AgentTaskCheck",
    "AmbiguityCheck",
    "CanaryCheck",
    "ComputationCheck",
    "ContextUseCheck",
    "DroppedInstructionCheck",
    "EmptyOutputDetector",
    "FalsePremiseCheck",
    "FormatDetector",
    "FragilityCheck",
    "HallucinationCheck",
    "InjectionCheck",
    "InstructionCheck",
    "OverRefusalCheck",
    "ProviderFilterDetector",
    "RefusalDetector",
    "RetrievalLabelCheck",
    "ToolUseCheck",
    "TruncationDetector",
    "default_detectors",
    "expected_format",
    "extract_constraints",
    "judge_detectors",
]


if TYPE_CHECKING:
    from counterfix.detect.basic import EmptyOutputDetector, ProviderFilterDetector
    from counterfix.detect.fmt import FormatDetector, expected_format
    from counterfix.detect.fragility import FragilityCheck
    from counterfix.detect.injection import CanaryCheck, InjectionCheck
    from counterfix.detect.instructions import InstructionCheck, extract_constraints
    from counterfix.detect.judged import (
        AmbiguityCheck,
        ComputationCheck,
        DroppedInstructionCheck,
        FalsePremiseCheck,
        HallucinationCheck,
        OverRefusalCheck,
        judge_detectors,
    )
    from counterfix.detect.rag import ContextUseCheck, RetrievalLabelCheck
    from counterfix.detect.refusal import RefusalDetector
    from counterfix.detect.tools import AgentTaskCheck, ToolUseCheck
    from counterfix.detect.truncation import TruncationDetector
