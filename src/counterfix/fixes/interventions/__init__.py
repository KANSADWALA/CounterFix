"""Every change :func:`counterfix.fix` can try, by cause, and what to suggest when none works.

``INTERVENTIONS`` is searched in this order; per failure the engine tries the ones aimed at
that cause. Profiles can reorder them per model.
"""

from __future__ import annotations

from counterfix.core.types import FailureType
from counterfix.fixes.interventions.base import Change, F, Intervention, _with_system
from counterfix.fixes.interventions.code import CODE
from counterfix.fixes.interventions.injection import INJECTION
from counterfix.fixes.interventions.prompt import PROMPT
from counterfix.fixes.interventions.rag import RAG
from counterfix.fixes.interventions.tools import TOOLS

INTERVENTIONS: list[Intervention] = [*PROMPT, *INJECTION, *TOOLS, *CODE, *RAG]

ESCALATION: dict[FailureType, list[str]] = {
    F.TOOL_MISUSE: [
        "rename tools so their names say what they do, and merge tools that overlap",
        "check tool calls in code before running them, and return errors the model can read",
        "use a model trained for tool calling",
    ],
    F.INJECTION_FOLLOWED: [
        "filter or sanitise retrieved text and tool results before they reach the model",
        "give the model fewer permissions (no tools that send, buy or reveal) on untrusted input",
        "use a model trained to resist prompt injection",
    ],
    F.CODE_FAILURE: [
        "let the model run the tests itself (a code tool) and iterate",
        "use a stronger coding model",
    ],
    F.TRUNCATION: ["ask for a shorter answer", "split the task into parts"],
    F.FORMAT_BREAK: ["use a model with structured outputs", "repair the JSON in code"],
    F.OVER_REFUSAL: ["state the purpose and audience in the prompt", "try another model"],
    F.HALLUCINATION: ["add retrieval (pass the source text)", "use a larger model", "abstain"],
    F.FALSE_PREMISE: ["remove the false premise from the question", "use a larger model"],
    F.DROPPED_INSTRUCTION: ["check the constraint in code and re-ask", "split the prompt"],
    F.COMPUTATION: ["route the calculation to a code or calculator tool"],
    F.PROMPT_FRAGILITY: ["add one or two worked examples (few-shot)", "use a larger model"],
    F.AMBIGUITY: ["ask the user a clarifying question"],
    F.RETRIEVAL_MISS: [
        "retrieve more passages (larger k)",
        "rewrite or expand the query before retrieval",
        "add keyword (hybrid) search or a reranker",
        "re-index with smaller or overlapping chunks",
    ],
    F.CONTEXT_IGNORED: ["use a larger model", "shorten or clean the passages"],
}


__all__ = ["ESCALATION", "INTERVENTIONS", "Change", "Intervention", "_with_system"]
