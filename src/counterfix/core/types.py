"""Failure types and small shared enums."""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """The failure taxonomy (F1 to F9; F10 and F11 for RAG, F12 for code, F13 for prompt
    injection, F14 for tool use) and non-model categories.

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
    RETRIEVAL_MISS = "F10"
    CONTEXT_IGNORED = "F11"
    CODE_FAILURE = "F12"
    INJECTION_FOLLOWED = "F13"
    TOOL_MISUSE = "F14"

    INFRASTRUCTURE = "INFRA"
    PROVIDER_FILTER = "FILTER"
    EMPTY_OUTPUT = "EMPTY"
    UNCLASSIFIED = "UNCLASSIFIED"

    @property
    def is_model_failure(self) -> bool:
        """True for F1 to F14: failures of the app's answer that a change can fix (F10 is
        the retriever's, but it is fixed the same way: change something, re-run, check)."""
        return self.value.startswith("F") and self.value[1:].isdigit()

    @property
    def is_rag(self) -> bool:
        """True for F10 and F11, which exist only when the call was given retrieved text."""
        return self in (FailureType.RETRIEVAL_MISS, FailureType.CONTEXT_IGNORED)

    @property
    def is_code(self) -> bool:
        """True for F12, found by running the code against tests (not by a judge)."""
        return self is FailureType.CODE_FAILURE

    @property
    def is_injection(self) -> bool:
        """True for F13, which exists only when the call carried untrusted text."""
        return self is FailureType.INJECTION_FOLLOWED

    @property
    def is_tool(self) -> bool:
        """True for F14, which exists only on calls that offer tools or on agent runs."""
        return self is FailureType.TOOL_MISUSE

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
    FailureType.RETRIEVAL_MISS: "Retrieval miss (the answer was not in the retrieved text)",
    FailureType.CONTEXT_IGNORED: "Context ignored (the answer was there, the model missed it)",
    FailureType.CODE_FAILURE: "Code fails its tests",
    FailureType.INJECTION_FOLLOWED: (
        "Followed instructions planted in retrieved text or tool output"
    ),
    FailureType.TOOL_MISUSE: "Tool misuse (wrong, unknown or missing tool call, loop)",
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
