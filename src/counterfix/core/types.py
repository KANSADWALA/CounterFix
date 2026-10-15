"""Failure types and small shared enums."""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """The failure taxonomy (v1: F1 to F9) plus non-model categories.

    ``INFRASTRUCTURE``, ``PROVIDER_FILTER`` and ``EMPTY_OUTPUT`` are kept apart from
    the model failure types so that API errors or provider filtering are never
    counted as the model's fault.
    """

    TRUNCATION = "F1"
    FORMAT_BREAK = "F2"
    OVER_REFUSAL = "F3"
    HALLUCINATION = "F4"
    FALSE_PREMISE = "F5"
    DROPPED_INSTRUCTION = "F6"
    PROMPT_FRAGILITY = "F7"
    COMPUTATION = "F8"
    AMBIGUITY = "F9"

    INFRASTRUCTURE = "INFRA"
    PROVIDER_FILTER = "FILTER"
    EMPTY_OUTPUT = "EMPTY"
    UNCLASSIFIED = "UNCLASSIFIED"

    @property
    def is_model_failure(self) -> bool:
        """True for F1 to F9, the failures that belong to the model's behaviour."""
        return self.value.startswith("F") and self.value[1:].isdigit()

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS: dict[FailureType, str] = {
    FailureType.TRUNCATION: "Truncation",
    FailureType.FORMAT_BREAK: "Format break (JSON, schema)",
    FailureType.OVER_REFUSAL: "Over-refusal",
    FailureType.HALLUCINATION: "Hallucination from missing knowledge",
    FailureType.FALSE_PREMISE: "False-premise acceptance",
    FailureType.DROPPED_INSTRUCTION: "Dropped instruction",
    FailureType.PROMPT_FRAGILITY: "Prompt fragility",
    FailureType.COMPUTATION: "Computation or token-level error",
    FailureType.AMBIGUITY: "Ambiguous request",
    FailureType.INFRASTRUCTURE: "Infrastructure error (not a model failure)",
    FailureType.PROVIDER_FILTER: "Provider content filter (not a model failure)",
    FailureType.EMPTY_OUTPUT: "Empty output",
    FailureType.UNCLASSIFIED: "Unclassified failure",
}


class PrescriptionKind(str, Enum):
    """What kind of change a prescription asks the user to make."""

    PARAM_CHANGE = "param_change"  # e.g. raise max_tokens, enable JSON mode
    PROMPT_CHANGE = "prompt_change"  # e.g. restructure, add "check premises"
    ADD_CONTEXT = "add_context"  # e.g. add retrieval
    USE_TOOL = "use_tool"  # e.g. route arithmetic to a code tool
    ASK_USER = "ask_user"  # e.g. ask a clarifying question (F9)
    ESCALATE = "escalate"  # e.g. larger model, abstain; no prompt-level fix
