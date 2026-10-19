"""Verified fixes: change one thing, re-run the call, keep only what worked.

>>> result = counterfix.fix(response)          # doctest: +SKIP
>>> print(result.explain())                    # doctest: +SKIP
>>> result.fix.params, result.fix.messages     # the change to apply, if one was verified

For each failure found (most confident first), the engine tries the interventions for
that cause, one at a time. Each is re-run ``runs`` times (3) and counts as a verified
fix only if the failure is gone, with no new failure, in at least ``need`` runs (2), so
luck does not count. Before any intervention it re-runs the call unchanged: if a plain
retry already passes, no change is claimed as the fix. Runs stop early once the outcome
is settled, the engine stops at the first verified fix, and ``max_runs`` caps the calls
to the model being fixed (judge calls come on top).

When nothing is verified, ``escalation`` lists what is left to try outside the prompt.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.caller import ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis, Finding, Prescription
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.judges.tasks import CONTEXT_KEY

F = FailureType
Change = tuple[list[Message], dict[str, Any]]


@dataclass(frozen=True)
class Intervention:
    """One change aimed at one cause. ``apply`` returns new (messages, params), or None
    when it does not fit this call."""

    name: str
    failure_type: FailureType
    kind: PrescriptionKind
    description: str
    apply: Callable[[CallRecord, Finding], Change | None]


def _with_system(messages: list[Message], text: str) -> list[Message]:
    out = copy.deepcopy(messages)
    if out and out[0].role in ("system", "developer") and isinstance(out[0].content, str):
        out[0].content = f"{out[0].content}\n\n{text}"
    else:
        out.insert(0, Message(role="system", content=text))
    return out


def _with_user_suffix(messages: list[Message], text: str) -> list[Message] | None:
    out = copy.deepcopy(messages)
    for m in reversed(out):
        if m.role == "user":
            if isinstance(m.content, str):
                m.content = f"{m.content}\n\n{text}"
            elif isinstance(m.content, list):
                m.content = [*m.content, {"type": "text", "text": text}]
            else:
                return None
            return out
    return None


def _system(text: str) -> Callable[[CallRecord, Finding], Change | None]:
    return lambda r, f: (_with_system(r.messages, text), dict(r.params))


def _suffix(text: str) -> Callable[[CallRecord, Finding], Change | None]:
    def apply(r: CallRecord, f: Finding) -> Change | None:
        messages = _with_user_suffix(r.messages, text)
        return None if messages is None else (messages, dict(r.params))

    return apply


def _raise_limit(r: CallRecord, f: Finding) -> Change | None:
    patch = f.prescription.patch if f.prescription is not None else {}
    limit = patch.get("max_tokens") or (r.max_tokens or 512) * 2
    return list(r.messages), {**r.params, "max_tokens": limit}


def _raise_limit_more(r: CallRecord, f: Finding) -> Change | None:
    limit = r.max_tokens or 512
    return list(r.messages), {**r.params, "max_tokens": min(max(limit * 16, 4096), 16384)}


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


INTERVENTIONS: list[Intervention] = [
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

ESCALATION: dict[FailureType, list[str]] = {
    F.TRUNCATION: ["ask for a shorter answer", "split the task into parts"],
    F.FORMAT_BREAK: ["use a model with structured outputs", "repair the JSON in code"],
    F.OVER_REFUSAL: ["state the purpose and audience in the prompt", "try another model"],
    F.HALLUCINATION: ["add retrieval (pass the source text)", "use a larger model", "abstain"],
    F.FALSE_PREMISE: ["remove the false premise from the question", "use a larger model"],
    F.DROPPED_INSTRUCTION: ["check the constraint in code and re-ask", "split the prompt"],
    F.COMPUTATION: ["route the calculation to a code or calculator tool"],
    F.PROMPT_FRAGILITY: ["add one or two worked examples (few-shot)", "use a larger model"],
    F.AMBIGUITY: ["ask the user a clarifying question"],
}


class FixAttempt(BaseModel):
    """One intervention (or the plain retry) and how its re-runs went."""

    name: str
    failure_type: FailureType | None = None
    description: str
    kind: PrescriptionKind | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    """The full request parameters to use (only those that changed, see ``changed``)."""
    messages: list[Message] | None = None
    """The full new messages when the prompt changed; None when only params changed."""
    changed: dict[str, Any] = Field(default_factory=dict)
    runs: int = 0
    passed: int = 0
    verified: bool = False
    seen: list[str] = Field(default_factory=list)
    """What the failed runs still showed, e.g. ["F1 Truncation"]."""
    errors: list[str] = Field(default_factory=list)


class FixResult(BaseModel):
    """What the engine tried, what worked, and what it cost."""

    record_id: str
    diagnosis: Diagnosis
    retry: FixAttempt | None = None
    attempts: list[FixAttempt] = Field(default_factory=list)
    fix: FixAttempt | None = None
    escalation: list[str] = Field(default_factory=list)
    calls: int = 0
    """Calls made to the model being fixed (judge calls not counted)."""
    unreachable: str | None = None
    """The error, when the model could not be called at all (nothing was tested)."""
    stopped: str = ""

    def explain(self) -> str:
        lines = [f"found: {self.diagnosis.summary()}"]
        if self.retry is not None:
            lines.append(f"plain retry: passed {self.retry.passed}/{self.retry.runs}")
        for a in self.attempts:
            mark = "VERIFIED" if a.verified else "no"
            still = f" (still: {', '.join(sorted(set(a.seen)))})" if a.seen else ""
            lines.append(f"tried {a.name}: passed {a.passed}/{a.runs} -> {mark}{still}")
            if a.errors:
                lines.append(f"  {len(a.errors)} call(s) failed, e.g. {a.errors[0][:120]}")
        if self.unreachable:
            lines.append(f"could not reach the model: {self.unreachable[:200]}")
        if self.fix is not None:
            lines.append(f"fix: {self.fix.description}")
            if self.fix.changed:
                lines.append(f"  change: {self.fix.changed}")
        elif self.escalation:
            lines.append("no verified fix; next: " + "; ".join(self.escalation))
        lines.append(f"{self.calls} re-run(s); {self.stopped}")
        return "\n".join(lines)


def _unreachable(result: FixResult) -> FixResult:
    result.stopped = "stopped: the model could not be called, so no fix was tested"
    result.escalation = []  # advice would be a guess: nothing ran
    return result


def _caller_for(record: CallRecord, client: Any) -> ModelCaller:
    if not record.model:
        raise ValueError("fix() needs the record's model to re-run the call")
    if client is not None:
        if not hasattr(client, "chat") and hasattr(client, "messages"):
            from counterfix.adapters.anthropic import AnthropicCaller

            return AnthropicCaller(client, record.model)
        from counterfix.core.caller import OpenAICompatibleCaller

        return OpenAICompatibleCaller(client, record.model, provider=record.provider)
    from counterfix.presets import PRESETS, make_caller

    if record.provider in PRESETS:
        return make_caller(f"{record.provider}:{record.model}")
    raise ValueError(
        "fix() cannot re-run this call: pass caller=..., or a response from a guarded client"
    )


def fix(
    target: Any,
    *,
    judge: Any = None,
    caller: ModelCaller | None = None,
    runs: int = 3,
    need: int = 2,
    max_runs: int = 15,
    check_retry: bool = True,
    profile: Any = "auto",
) -> FixResult:
    """Find a verified fix for one failed call. ``target`` is a guarded response, its
    Diagnosis or a CallRecord; ``judge`` is chosen as in :func:`counterfix.diagnose`
    (``judge=False`` for live checks only); ``caller`` re-runs the call (default: the
    guarded client, or the provider preset named on the record). ``profile`` orders the
    interventions by what worked before on this model: "auto" loads the stored profile
    for the record's model if there is one, None keeps the default order."""
    from counterfix.detectors import default_detectors, judge_detectors
    from counterfix.diagnose import _target, diagnose
    from counterfix.judges.auto import resolve_judge

    record, client = _target(target)
    reuse = isinstance(target, Diagnosis) and (target.judge or judge is False)
    found = target if reuse else None
    diagnosis = found or diagnose(record, judge=judge)
    result = FixResult(record_id=record.id, diagnosis=diagnosis)
    failures = sorted(diagnosis.model_failures, key=lambda f: -f.confidence)
    if not failures:
        result.stopped = "nothing to fix"
        return result
    rerun = caller or _caller_for(record, client)
    judges_needed = judge is not False and any(
        f.failure_type not in (F.TRUNCATION, F.FORMAT_BREAK) for f in failures
    )  # (F7 can be checked without one: FragilityCheck falls back to a heuristic)
    chosen = (
        resolve_judge(None if judge is True else judge, target_model=record.model)[0]
        if judges_needed
        else None
    )
    before = {f.failure_type for f in diagnosis.model_failures}

    def checks_for(kind: FailureType) -> list[Detector]:
        from counterfix.detectors import FragilityCheck

        extra = [] if chosen is None else judge_detectors(chosen)
        if kind is F.PROMPT_FRAGILITY:  # re-asks rewordings: extra calls to the model
            extra.append(FragilityCheck(rerun, judge=chosen))
        return default_detectors() + [d for d in extra if kind in getattr(d, "covers", ())]

    def checkable(kind: FailureType) -> bool:
        return any(kind in getattr(d, "covers", ()) for d in checks_for(kind))

    failures = [f for f in failures if checkable(f.failure_type)]
    if not failures:
        result.stopped = "no check can confirm a fix for these failures without a judge"
        return result

    def trial(attempt: FixAttempt, kind: FailureType, messages: list[Message]) -> FixAttempt:
        checks = checks_for(kind)
        for _ in range(runs):
            if result.calls >= max_runs:
                break
            result.calls += 1
            attempt.runs += 1
            try:
                new = rerun.complete(messages, **attempt.params)
            except Exception as exc:
                attempt.errors.append(f"{type(exc).__name__}: {exc}")
                if len(attempt.errors) == attempt.runs:  # nothing got through yet
                    result.unreachable = attempt.errors[0]
                    break
                continue
            new.metadata = {**record.metadata, **new.metadata}
            after = run_detectors(new, checks, Config())
            left = {f.failure_type for f in after.model_failures}
            if kind not in left and left <= before:
                attempt.passed += 1
            else:
                attempt.seen += [f"{t.value} {t.label}" for t in sorted(left, key=str)]
            if attempt.passed >= need or attempt.runs - attempt.passed > runs - need:
                break  # settled either way
        attempt.verified = attempt.passed >= need
        return attempt

    top = failures[0].failure_type
    if check_retry:
        retry = FixAttempt(name="plain_retry", description="Re-run unchanged", params=record.params)
        result.retry = trial(retry, top, record.messages)
        if result.unreachable:
            return _unreachable(result)
        if retry.verified:
            result.stopped = "a plain retry passes too, so no change is claimed as the fix"
            return result
    if profile == "auto":
        from counterfix.profiles import load

        profile = load(record.model) if record.model else None
    for finding in failures:
        ivs = [i for i in INTERVENTIONS if i.failure_type is finding.failure_type]
        if profile is not None:
            order = profile.fix_order(finding.failure_type.value, [i.name for i in ivs])
            ivs.sort(key=lambda i: order.index(i.name))
        for iv in ivs:
            change = iv.apply(record, finding)
            if change is None:
                continue
            if result.calls >= max_runs:
                result.stopped = f"stopped at the budget of {max_runs} re-runs"
                break
            messages, params = change
            attempt = FixAttempt(
                name=iv.name,
                failure_type=iv.failure_type,
                description=iv.description,
                kind=iv.kind,
                params=params,
                messages=None if messages == record.messages else messages,
                changed={k: v for k, v in params.items() if record.params.get(k) != v},
            )
            result.attempts.append(trial(attempt, finding.failure_type, messages))
            if result.unreachable:
                return _unreachable(result)
            if attempt.verified:
                result.fix = attempt
                patch: dict[str, Any] = dict(attempt.changed)
                if attempt.messages is not None:
                    patch["messages"] = [m.model_dump(exclude_none=True) for m in messages]
                finding.prescription = Prescription(
                    kind=iv.kind, description=iv.description, patch=patch, verified=True
                )
                result.stopped = "stopped at the first verified fix"
                return result
    result.escalation = [s for f in failures for s in ESCALATION.get(f.failure_type, [])]
    result.stopped = result.stopped or "no intervention passed often enough"
    return result
