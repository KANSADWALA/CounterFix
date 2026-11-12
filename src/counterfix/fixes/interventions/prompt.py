"""Interventions for answers: truncation, format, refusals, invented facts, false premises,
missed instructions, fragile prompts, maths and vague requests (F1 to F9)."""

from __future__ import annotations

from counterfix.core.diagnosis import Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import PrescriptionKind
from counterfix.fixes.interventions.base import (
    Change,
    F,
    Intervention,
    _suffix,
    _system,
    _with_system,
    _with_user_suffix,
)
from counterfix.judges.tasks import CONTEXT_KEY


def _raise_limit(r: CallRecord, f: Finding) -> Change | None:
    patch = f.prescription.patch if f.prescription is not None else {}
    limit = patch.get(r.limit_key) or patch.get("max_tokens") or (r.max_tokens or 512) * 2
    return list(r.messages), {**r.params, r.limit_key: limit}


def _raise_limit_more(r: CallRecord, f: Finding) -> Change | None:
    limit = r.max_tokens or 512
    return list(r.messages), {**r.params, r.limit_key: min(max(limit * 16, 4096), 16384)}


def _shorter(r: CallRecord, f: Finding) -> Change | None:
    if not r.max_tokens:
        return None
    words = max(10, int(r.max_tokens * 0.5))  # ~0.75 words per token, with room to spare
    return _suffix(f"Keep the whole answer under {words} words so it is not cut off.")(r, f)


def _json_mode(r: CallRecord, f: Finding) -> Change | None:
    if r.provider == "anthropic" or "response_format" in r.params:
        return None  # no OpenAI-style JSON mode there, or it was already on
    messages = _with_user_suffix(r.messages, "Reply in JSON.")  # JSON mode needs the word
    if messages is None:
        return None
    return messages, {**r.params, "response_format": {"type": "json_object"}}


def _checklist(r: CallRecord, f: Finding) -> Change | None:
    items: list[str] = []
    for e in f.evidence:
        items += [b["instruction"] for b in e.data.get("broken") or [] if "instruction" in b]
        items += [str(x) for x in e.data.get("missed") or []]
    if not items:
        items = [e.detail.removeprefix("instructions not met: ") for e in f.evidence]
    listed = "\n".join(f"- {i}" for i in items)
    return _suffix(f"Before you reply, make sure the answer meets every instruction:\n{listed}")(
        r, f
    )


def _add_context(r: CallRecord, f: Finding) -> Change | None:
    context = r.metadata.get(CONTEXT_KEY)
    if not context:
        return None
    text = f"Use only this source to answer:\n<source>\n{context}\n</source>"
    return _with_system(r.messages, text), dict(r.params)


def _clarify(r: CallRecord, f: Finding) -> Change | None:
    readings = [x for e in f.evidence for x in e.data.get("readings") or []]
    if not readings:
        return None
    return _suffix(f"(To be clear, I mean: {readings[0]})")(r, f)


PROMPT: list[Intervention] = [
    Intervention(
        "raise_max_tokens",
        F.TRUNCATION,
        PrescriptionKind.PARAM_CHANGE,
        "Raise max_tokens",
        _raise_limit,
    ),
    Intervention(
        "raise_max_tokens_more",
        F.TRUNCATION,
        PrescriptionKind.PARAM_CHANGE,
        "Raise max_tokens a lot (long answers, or reasoning that uses the budget)",
        _raise_limit_more,
    ),
    Intervention(
        "ask_shorter",
        F.TRUNCATION,
        PrescriptionKind.PROMPT_CHANGE,
        "Ask for an answer short enough to fit the current max_tokens",
        _shorter,
    ),
    Intervention(
        "json_only_prompt",
        F.FORMAT_BREAK,
        PrescriptionKind.PROMPT_CHANGE,
        "Ask for bare JSON at the end of the prompt",
        _suffix("Reply with only valid JSON: no code fences and no text before or after it."),
    ),
    Intervention(
        "json_mode",
        F.FORMAT_BREAK,
        PrescriptionKind.PARAM_CHANGE,
        "Turn on JSON mode (response_format json_object)",
        _json_mode,
    ),
    Intervention(
        "state_purpose",
        F.OVER_REFUSAL,
        PrescriptionKind.PROMPT_CHANGE,
        "Tell the model the request is legitimate and to help unless it is truly unsafe",
        _system(
            "The user's requests are legitimate (education, safety, work or creative "
            "writing). Help fully; decline only what would cause real harm."
        ),
    ),
    Intervention(
        "add_source",
        F.HALLUCINATION,
        PrescriptionKind.ADD_CONTEXT,
        "Put the source text in the prompt and answer only from it",
        _add_context,
    ),
    Intervention(
        "allow_unknown",
        F.HALLUCINATION,
        PrescriptionKind.PROMPT_CHANGE,
        "Let the model say it does not know instead of guessing",
        _system(
            "Only state facts you are sure of. If you do not know or cannot verify "
            "something, say so plainly instead of guessing."
        ),
    ),
    Intervention(
        "check_premises",
        F.FALSE_PREMISE,
        PrescriptionKind.PROMPT_CHANGE,
        "Tell the model to check the question's assumptions first",
        _system(
            "Before answering, check whether the question assumes something false. If it "
            "does, say so and correct it before answering anything else."
        ),
    ),
    Intervention(
        "instruction_checklist",
        F.DROPPED_INSTRUCTION,
        PrescriptionKind.PROMPT_CHANGE,
        "Repeat the missed instructions as a checklist at the end of the prompt",
        _checklist,
    ),
    Intervention(
        "fixed_answer_format",
        F.PROMPT_FRAGILITY,
        PrescriptionKind.PROMPT_CHANGE,
        "Pin the answer format so rewordings get the same answer",
        _system(
            "Read the request for what it asks, not how it is worded. Give the answer "
            "itself first, in as few words as possible, in the same form every time."
        ),
    ),
    Intervention(
        "work_it_out",
        F.COMPUTATION,
        PrescriptionKind.PROMPT_CHANGE,
        "Ask for the calculation step by step, checked, before the final answer",
        # ponytail: a prompt nudge; routing to a code tool is the stronger fix, needs tools
        _system(
            "For any arithmetic, counting or dates, work it out step by step, check the "
            "result, then give the final answer."
        ),
    ),
    Intervention(
        "add_detail",
        F.AMBIGUITY,
        PrescriptionKind.ASK_USER,
        "Add the missing detail (here: the first reading the judge found)",
        _clarify,
    ),
    Intervention(
        "state_reading",
        F.AMBIGUITY,
        PrescriptionKind.PROMPT_CHANGE,
        "Have the model name the reading it chose, or ask, when a request is unclear",
        _system(
            "If a request could mean more than one thing, say in one line which meaning "
            "you are answering, or ask a short clarifying question."
        ),
    ),
]
