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

Bring your own check: ``fix(response, check=my_test)`` lets your own test (a function,
a counterfix Metric, or a wrapper around a DeepEval or Ragas metric) decide whether a
re-run passed. The diagnosis still decides what to try; your check decides what counts.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
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
    "Check",
    "CheckError",
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


Check = Callable[[CallRecord], Any]
"""A test of one answer: returns True/False, a (bool, reason) pair, or an object with
``.passed`` (a counterfix MetricResult). A counterfix Metric is accepted as is."""


class CheckError(RuntimeError):
    """Your check raised (or returned something unclear) on the original call. The
    original exception is attached as ``__cause__``."""


def _check_name(check: Any) -> str:
    name = getattr(check, "name", None) or getattr(check, "__name__", None)
    return str(name or type(check).__name__)


def _run_check(check: Any, record: CallRecord) -> tuple[bool, str]:
    """Run the developer's check on one answer: (passed, reason). Anything other than a
    clear pass or fail is an error, so a check that forgot to return is never a pass."""
    if callable(getattr(check, "score", None)) and hasattr(check, "default_threshold"):
        out: Any = check.score(record)  # a counterfix Metric (no reference answer here)
        if out.passed is None:
            raise TypeError(
                f"the metric {_check_name(check)} gave no pass/fail on this call"
                f" ({out.reason or 'it has no threshold'}). Give it a threshold, or wrap"
                " it with its reference: check=lambda r: metric.score(r, reference).passed"
            )
    else:
        out = check(record)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
        return out[0], str(out[1] or "")
    if isinstance(out, bool):
        return out, ""
    passed = getattr(out, "passed", None)
    if isinstance(passed, bool):
        return passed, str(getattr(out, "reason", "") or "")
    raise TypeError(
        f"the check {_check_name(check)} returned {type(out).__name__}; return True or "
        "False, a (bool, reason) pair, or an object with a boolean .passed (a score needs "
        "a threshold: return score >= 0.7, not the score)"
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
    check: Check | Any | None = None,
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
    longer used (it was the old "2 of 3" rule) and only warns.

    ``check`` is your own test of an answer (see :data:`Check`). With it, a re-run passes
    when your check passes and no new failure appears; counterfix's diagnosis still
    picks what to try. If your check already passes on this call, nothing is re-run. If it
    fails but counterfix finds no cause, nothing is re-run either and ``stopped`` says so:
    a targeted change needs a cause. An error raised by your check on the original call
    is raised to you as :class:`CheckError`; one raised on a re-run counts that run as a
    fail."""
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
    if check is not None:
        try:
            ok, why = _run_check(check, record)
        except Exception as exc:  # the caller's bug to see, said plainly
            raise CheckError(
                f"your check {_check_name(check)} raised {type(exc).__name__} on this call: {exc}"
            ) from exc
        if ok:
            result = FixResult(
                record_id=record.id,
                diagnosis=target
                if isinstance(target, Diagnosis)
                else diagnose(target, judge=False),
                check=_check_name(check),
            )
            result.stopped = "your check passes on this call: nothing to fix"
            return result
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
    result = FixResult(
        record_id=record.id,
        diagnosis=diagnosis,
        check=None if check is None else _check_name(check),
    )
    failures = sorted(diagnosis.model_failures, key=lambda f: -f.confidence)
    if not failures:
        if check is not None:
            result.stopped = (f"your check fails ({why})" if why else "your check fails") + (
                ", but counterfix found no cause it can target, so no change was tried"
                + (" (a judge may find one: pass judge=...)" if judge is False else "")
            )
        else:
            result.stopped = "nothing to fix"
        return result
    rerun = caller or _caller_for(record, client)
    # with your check, it decides each run: no judge is needed to confirm a fix
    judges_needed = (
        check is None
        and judge is not False
        and any(f.failure_type not in (F.TRUNCATION, F.FORMAT_BREAK) for f in failures)
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

        if check is not None:  # live checks only, to catch a change that breaks something
            return default_detectors()
        extra = [] if chosen is None else judge_detectors(chosen)
        if kind is F.PROMPT_FRAGILITY:  # re-asks rewordings: extra calls to the model
            extra.append(FragilityCheck(rerun, judge=chosen))
        return default_detectors() + [d for d in extra if kind in getattr(d, "covers", ())]

    def checkable(kind: FailureType) -> bool:
        if check is not None:
            return True
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
            if check is not None:
                try:
                    ok, _ = _run_check(check, new)
                except Exception as exc:  # the run is read, so it counts: as a fail
                    ok = False
                    attempt.seen.append(f"your check raised {type(exc).__name__}: {exc}"[:200])
                else:
                    if not ok:
                        attempt.seen.append("your check failed")
                passed = ok and left <= before
                if ok and not left <= before:
                    attempt.seen += [f"{t.value} {t.label}" for t in sorted(left - before, key=str)]
            else:
                passed = kind not in left and left <= before
                if not passed:
                    attempt.seen += [f"{t.value} {t.label}" for t in sorted(left, key=str)]
            if passed:
                attempt.passed += 1
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
