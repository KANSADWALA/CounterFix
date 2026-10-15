"""Lint rules that need meaning, so they ask a judge: L2 conflicting constraints,
L5 system/user contradiction, L6 underspecified request. Each costs one judge call."""

from __future__ import annotations

from collections.abc import Callable

from counterfix.core.types import FailureType
from counterfix.judges.base import Judge, JudgeTask
from counterfix.lint.rules import LintIssue, PromptView, _clip

JudgeRule = Callable[[PromptView, Judge], list[LintIssue]]


def _listed(value: object) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value[:3])
    return str(value or "")


def l2_conflicting_constraints(view: PromptView, judge: Judge) -> list[LintIssue]:
    if not view.all_text.strip():
        return []
    verdict = judge.ask(
        JudgeTask(
            name="L2_conflict",
            instructions=(
                "Do any two instructions in this prompt conflict, so that both cannot be "
                "followed at once (e.g. 'be brief' and 'explain in full detail', 'exactly 3 "
                "bullets' and 'cover all 5 points')? Ordinary, compatible instructions do not "
                "count."
            ),
            material={"prompt": view.all_text},
            labels=("no_conflict", "conflict"),
            extra_fields={"conflicts": "list of the conflicting instruction pairs, quoted"},
        )
    )
    if verdict.label != "conflict":
        return []
    conflicts = _listed(verdict.extra.get("conflicts")) or verdict.reason
    return [
        LintIssue(
            rule="L2",
            title="Conflicting instructions",
            severity="warning",
            message=f"Some instructions cannot all be followed: {conflicts}",
            suggestion="Decide which instruction wins and remove or reword the other.",
            failure_types=[FailureType.DROPPED_INSTRUCTION, FailureType.PROMPT_FRAGILITY],
        )
    ]


def l5_system_user_contradiction(view: PromptView, judge: Judge) -> list[LintIssue]:
    if not view.system.strip() or not view.last_user.strip():
        return []
    verdict = judge.ask(
        JudgeTask(
            name="L5_contradiction",
            instructions=(
                "Does the user message ask for something the system instructions forbid or "
                "contradict (format, language, length, scope, persona)?"
            ),
            material={"system_instructions": view.system, "user_message": view.last_user},
            labels=("consistent", "contradiction"),
            extra_fields={"contradiction": "what contradicts what, quoted"},
        )
    )
    if verdict.label != "contradiction":
        return []
    return [
        LintIssue(
            rule="L5",
            title="System and user message disagree",
            severity="warning",
            message=f"{_listed(verdict.extra.get('contradiction')) or verdict.reason}",
            suggestion="Make the system prompt and the user message agree, or say which one wins.",
            failure_types=[FailureType.DROPPED_INSTRUCTION, FailureType.OVER_REFUSAL],
        )
    ]


def l6_underspecified(view: PromptView, judge: Judge) -> list[LintIssue]:
    if not view.last_user.strip():
        return []
    verdict = judge.ask(
        JudgeTask(
            name="L6_underspecified",
            instructions=(
                "Is this request missing a detail without which a correct answer is a guess "
                "(which place, product, version, time period, unit or person is meant)? "
                "Requests that are open-ended on purpose (e.g. 'write a poem') are fine."
            ),
            material={"request": view.last_user},
            labels=("clear", "underspecified"),
            extra_fields={"missing": "the missing detail(s)"},
        )
    )
    if verdict.label != "underspecified":
        return []
    return [
        LintIssue(
            rule="L6",
            title="Request is missing a key detail",
            severity="info",
            message=f"Missing: {_listed(verdict.extra.get('missing')) or verdict.reason}",
            suggestion="Add the missing detail, or let the model ask a clarifying question.",
            failure_types=[FailureType.AMBIGUITY],
            snippet=_clip(view.last_user),
        )
    ]


JUDGE_RULES: dict[str, JudgeRule] = {
    "L2": l2_conflicting_constraints,
    "L5": l5_system_user_contradiction,
    "L6": l6_underspecified,
}
