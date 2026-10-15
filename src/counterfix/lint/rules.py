"""Rule-based prompt lint checks. No LLM calls.

Each rule names the failure type it tends to cause. Rules L2, L5 and L6 need meaning
(conflicts, contradictions, missing referents) and arrive with the judge in Phase 3.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from counterfix.core.record import Message
from counterfix.core.types import FailureType

Severity = Literal["info", "warning"]


class LintIssue(BaseModel):
    """One problem found in a prompt before it is sent."""

    rule: str
    title: str
    severity: Severity
    message: str
    suggestion: str
    failure_types: list[FailureType] = Field(default_factory=list)
    snippet: str | None = None


class PromptView:
    """The prompt as the linter sees it."""

    def __init__(self, messages: Sequence[Message], params: dict[str, Any]):
        self.messages = list(messages)
        self.params = params
        self.system = "\n".join(m.text for m in messages if m.role in ("system", "developer"))
        users = [m.text for m in messages if m.role == "user"]
        self.last_user = users[-1] if users else ""
        self.all_text = "\n".join(m.text for m in messages if m.role != "assistant")


LintRule = Callable[[PromptView], list[LintIssue]]

_SENTENCES = re.compile(r"[^.!?\n]+[.!?]?")
_CONSTRAINT = re.compile(
    r"\b(must|never|always|do not|don't|only|exactly|at most|at least|no more than|"
    r"without|avoid|make sure|ensure)\b",
    re.IGNORECASE,
)


def _clip(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def l1_buried_instruction(view: PromptView) -> list[LintIssue]:
    """A constraint sits in the middle of a long prompt and is not repeated near the end."""
    text = view.all_text
    if len(text.split()) < 150:
        return []
    tail = text[int(len(text) * 0.8) :].lower()
    buried: list[str] = []
    for match in _SENTENCES.finditer(text):
        sentence = match.group(0).strip()
        position = match.start() / max(len(text), 1)
        if 0.2 <= position <= 0.8 and _CONSTRAINT.search(sentence):
            key = " ".join(sentence.lower().split()[:5])
            if key not in tail:
                buried.append(sentence)
    if not buried:
        return []
    return [
        LintIssue(
            rule="L1",
            title="Instruction buried mid-prompt",
            severity="warning",
            message=(
                f"{len(buried)} constraint(s) sit in the middle of a long prompt, where models "
                "most often drop them."
            ),
            suggestion=(
                "Move key constraints to the end, or repeat them as a short checklist there."
            ),
            failure_types=[FailureType.DROPPED_INSTRUCTION],
            snippet=_clip(buried[0]),
        )
    ]


_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|which|whose|why|how many|how much|how long|in which|in what|"
    r"did|does|do|is|was|were|are|can|has|have)\b",
    re.IGNORECASE,
)
_ABSTAIN = re.compile(
    r"don'?t know|do not know|not sure|unsure|uncertain|say so|if (?:it'?s |you are )?unknown|"
    r"can(?:no|')t find|not (?:in|found in) the (?:context|document|text|source)|"
    r"insufficient (?:information|context)|admit|no answer",
    re.IGNORECASE,
)


_COMPUTATION = re.compile(
    r"\d{5,}|\d+\s*[-+*/^x\u00d7]\s*\d+|\b(?:sum|product|multipl\w*|divid\w*|division|"
    r"power of|square root|percent\w*|backwards|revers\w*|letters?|characters?|digits?|count)\b",
    re.IGNORECASE,
)


def l3_no_abstain_option(view: PromptView) -> list[LintIssue]:
    """A factual question with no permission to say "I don't know".

    Computation and character-level questions are skipped: their fix is a tool (F8),
    not permission to abstain.
    """
    question = view.last_user.strip()
    if not question.endswith("?") or not _QUESTION_START.search(question):
        return []
    if _COMPUTATION.search(question):
        return []
    if _ABSTAIN.search(view.all_text):
        return []
    return [
        LintIssue(
            rule="L3",
            title="No 'I don't know' option",
            severity="info",
            message="Factual question with no permission to say the answer is unknown.",
            suggestion=(
                'Add a line such as: "If you are not sure, or the premise is wrong, say so '
                'instead of guessing."'
            ),
            failure_types=[FailureType.HALLUCINATION, FailureType.FALSE_PREMISE],
            snippet=_clip(question),
        )
    ]


_FORMAT_REQUEST = re.compile(
    r"\b(?:only|valid|return|output|respond|reply|answer|format(?:ted)?|as|in|convert\b[^.\n]{0,40}\bto)"
    r"\b[^.\n]{0,40}\b(json|yaml|csv|xml)\b",
    re.IGNORECASE,
)
_STRUCTURE_HINT = re.compile(r"[{\[]|^\s*\|.*\|", re.MULTILINE)


def l4_vague_format(view: PromptView) -> list[LintIssue]:
    """A structured format is asked for in words only, with no example and no JSON mode."""
    match = _FORMAT_REQUEST.search(view.all_text)
    if match is None:
        return []
    if isinstance(view.params.get("response_format"), dict):
        return []
    if _STRUCTURE_HINT.search(view.all_text):
        return []
    fmt = match.group(1).upper()
    return [
        LintIssue(
            rule="L4",
            title="Output format described only in words",
            severity="warning",
            message=f"{fmt} output is requested, but there is no example, schema or JSON mode.",
            suggestion=(
                f"Show a small {fmt} example or schema in the prompt"
                + (", and turn on JSON mode / structured output." if fmt == "JSON" else ".")
            ),
            failure_types=[FailureType.FORMAT_BREAK],
            snippet=_clip(match.group(0)),
        )
    ]


_INPUT_LABEL = re.compile(
    r"^\s*(?:document|context|text|email|review|article|input|transcript|data|source|"
    r"page|message|comment)s?\s*:\s*(.+)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_DELIMITERS = re.compile(r'```|"""|\'\'\'|<[a-z_][\w-]*>|^-{3,}\s*$|^#{2,}\s', re.MULTILINE | re.I)


def l7_undelimited_input(view: PromptView) -> list[LintIssue]:
    """Pasted outside text (a document, email, review...) with no delimiters around it."""
    match = _INPUT_LABEL.search(view.last_user)
    if match is None or len(match.group(1)) < 200:
        return []
    if _DELIMITERS.search(view.last_user):
        return []
    return [
        LintIssue(
            rule="L7",
            title="Outside text without delimiters",
            severity="warning",
            message=(
                "Long pasted text is mixed into the instructions. Instructions hidden inside it "
                "can hijack the model (prompt injection)."
            ),
            suggestion=(
                "Wrap outside text in clear delimiters, e.g. <document>...</document>, and say "
                "that text inside them is data, not instructions."
            ),
            failure_types=[],
            snippet=_clip(match.group(0)),
        )
    ]


RULES: dict[str, LintRule] = {
    "L1": l1_buried_instruction,
    "L3": l3_no_abstain_option,
    "L4": l4_vague_format,
    "L7": l7_undelimited_input,
}
