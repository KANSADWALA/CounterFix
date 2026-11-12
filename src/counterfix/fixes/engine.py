"""Verified fixes: change one thing, re-run the call, keep only what worked.

>>> result = counterfix.fix(response)          # doctest: +SKIP
>>> print(result.explain())                    # doctest: +SKIP
>>> result.fix.params, result.fix.messages     # the change to apply, if one was verified

First the engine re-runs the call unchanged (the plain retry), up to ``runs`` times (6).
Then, for each failure found (most confident first), it tries the interventions for that
cause, one at a time, each re-run up to ``runs`` times. A run passes when the failure is
gone and no new failure appears. A change counts as a verified fix only if it passes
clearly more often than the plain retry: a one-sided Fisher exact test at ``alpha``
(0.05) divided by the number of changes the search may try (see :mod:`counterfix.fixes.verify`).
If the plain retry passes so often that no change could be shown to do better, no change
is claimed. Runs stop once the outcome is settled, the engine stops at the first verified
fix, and ``max_runs`` caps the calls to the model being fixed (judge calls come on top).

When nothing is verified, ``escalation`` lists what is left to try outside the prompt.
"""

from __future__ import annotations

import warnings
from typing import Any

from counterfix.core.caller import ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis, Prescription
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType
from counterfix.fixes.interventions import (
    ESCALATION,
    INTERVENTIONS,
    Change,
    Intervention,
    _with_system,
)
from counterfix.fixes.models import FixAttempt, FixResult
from counterfix.fixes.verify import DEFAULT_ALPHA, DEFAULT_RUNS, beatable, level, p_value, settled

F = FailureType

__all__ = [
    "ESCALATION",
    "INTERVENTIONS",
    "Change",
    "FixAttempt",
    "FixResult",
    "Intervention",
    "_with_system",
    "fix",
]


def _unreachable(result: FixResult) -> FixResult:
    result.stopped = "stopped: the model could not be called, so no fix was tested"
    result.escalation = []  # advice would be a guess: nothing ran
    return result


def _caller_for(record: CallRecord, client: Any) -> ModelCaller:
    if not record.model:
        raise ValueError("fix() needs the record's model to re-run the call")
    if client is not None:
        if not hasattr(client, "chat") and hasattr(client, "messages"):
            from counterfix.core.caller import AnthropicCaller

            return AnthropicCaller(client, record.model)
        from counterfix.core.caller import OpenAICompatibleCaller

        return OpenAICompatibleCaller(client, record.model, provider=record.provider)
    from counterfix.core.presets import PRESETS, make_caller

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
    runs: int = DEFAULT_RUNS,
    need: int | None = None,
    max_runs: int = 30,
    check_retry: bool = True,
    profile: Any = "auto",
    alpha: float = DEFAULT_ALPHA,
) -> FixResult:
    """Find a verified fix for one failed call. ``target`` is a guarded response, its
    Diagnosis or a CallRecord; ``judge`` is chosen as in :func:`counterfix.diagnose`
    (``judge=False`` for live checks only); ``caller`` re-runs the call (default: the
    guarded client, or the provider preset named on the record). ``profile`` orders the
    interventions by what worked before on this model: "auto" loads the stored profile
    for the record's model if there is one, None keeps the default order.

    ``runs`` is how often the plain retry and each change are re-run; ``alpha`` the
    chance, across the search, of calling a useless change verified. ``check_retry=False``
    skips the plain retry and assumes it never passes: use it only for failures you know
    repeat every time (a fixed token limit), or "verified" means less. ``need`` is no
    longer used (it was the old "2 of 3" rule) and only warns."""
    if need is not None:
        warnings.warn(
            "fix(need=...) is ignored: a change is now verified when it beats the plain "
            "retry (see counterfix.fixes.verify)",
            DeprecationWarning,
            stacklevel=2,
        )
    from counterfix.core.attach import target_record as _target
    from counterfix.detect import default_detectors, judge_detectors
    from counterfix.diagnose import diagnose
    from counterfix.judges.auto import NoJudgeError, resolve_judge

    record, client = _target(target)
    reuse = isinstance(target, Diagnosis) and (target.judge or judge is False)
    found = target if reuse else None
    if found is not None:
        diagnosis = found
    else:
        try:  # the target, not the bare record: it carries the client a judge can fall back on
            diagnosis = diagnose(target, judge=judge)
        except NoJudgeError:
            if judge is not None:
                raise
            # No judge anywhere: cut-off answers, broken JSON and missed instructions are
            # still found and fixed by the live checks; say what was skipped.
            warnings.warn(
                "counterfix: no judge found, so fix() used the live checks only (truncation, "
                "format, instructions, ...). Set GROQ_API_KEY or GEMINI_API_KEY (free), or "
                "call counterfix.configure(judge=...), to also check facts and premises.",
                UserWarning,
                stacklevel=2,
            )
            diagnosis = diagnose(target, judge=False)
            judge = False
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
        resolve_judge(
            None if judge is True else judge,
            target_model=record.model,
            fallback_client=client,
            fallback_provider=record.provider,
        )[0]
        if judges_needed
        else None
    )
    before = {f.failure_type for f in diagnosis.model_failures}

    def checks_for(kind: FailureType) -> list[Detector]:
        from counterfix.detect import FragilityCheck

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

    tests = sum(1 for f in failures for i in INTERVENTIONS if i.failure_type is f.failure_type)
    at = level(alpha, tests)
    result.level = at
    base = [0, runs if not check_retry else 0]  # plain retry: passed, observed

    def done(attempt: FixAttempt, is_retry: bool) -> bool:
        if is_retry:  # the retry runs in full unless it already passes too often
            return not beatable(attempt.passed, attempt.observed, runs, at)
        return settled(
            base[0],
            base[1],
            attempt.passed,
            attempt.observed,
            attempt.observed + (runs - attempt.runs),
            at,
        )

    def trial(
        attempt: FixAttempt, kind: FailureType, messages: list[Message], is_retry: bool = False
    ) -> FixAttempt:
        checks = checks_for(kind)
        for _ in range(runs):
            if result.calls >= max_runs:
                break
            result.calls += 1
            attempt.runs += 1
            try:
                new = rerun.complete(messages, **attempt.params)
                if new.response_text is None and not new.tool_calls and not new.finish_reason:
                    raise ValueError("the re-run returned no readable answer")
            except Exception as exc:  # a run that cannot be read is never counted as a pass
                attempt.errors.append(f"{type(exc).__name__}: {exc}")
                if len(attempt.errors) == attempt.runs:  # nothing got through yet
                    result.unreachable = attempt.errors[0]
                    break
                continue
            new.metadata = {**record.metadata, **new.metadata, **attempt.metadata}
            after = run_detectors(new, checks, Config())
            left = {f.failure_type for f in after.model_failures}
            if kind not in left and left <= before:
                attempt.passed += 1
            else:
                attempt.seen += [f"{t.value} {t.label}" for t in sorted(left, key=str)]
            if done(attempt, is_retry):
                break  # settled either way
        if is_retry:
            attempt.verified = not beatable(attempt.passed, attempt.observed, runs, at)
        else:
            attempt.p_value = p_value(base[0], base[1], attempt.passed, attempt.observed)
            attempt.verified = attempt.passed > 0 and attempt.p_value <= at
        return attempt

    top = failures[0].failure_type
    if check_retry:
        retry = FixAttempt(name="plain_retry", description="Re-run unchanged", params=record.params)
        result.retry = trial(retry, top, record.messages, is_retry=True)
        if result.unreachable:
            return _unreachable(result)
        base[0], base[1] = retry.passed, retry.observed
        if retry.verified:
            result.stopped = (
                f"a plain retry passed {retry.passed}/{retry.observed}: too often for any "
                "change to be shown to do better, so none is claimed"
            )
            return result
    if profile == "auto":
        from counterfix.fixes._profile_store import load

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
                metadata=iv.metadata(record, finding) if iv.metadata else {},
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
    result.stopped = result.stopped or "no change beat the plain retry clearly enough"
    return result
