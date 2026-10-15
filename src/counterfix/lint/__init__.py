"""The prompt linter: catch failure-prone prompts before any call is made."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from counterfix.core.record import Message
from counterfix.judges.base import Judge
from counterfix.lint.judged import JUDGE_RULES
from counterfix.lint.rules import RULES, LintIssue, LintRule, PromptView

PromptLike = str | Sequence[Message | dict[str, Any]]


def _messages(prompt: PromptLike) -> list[Message]:
    if isinstance(prompt, str):
        return [Message(role="user", content=prompt)]
    return [m if isinstance(m, Message) else Message.model_validate(m) for m in prompt]


def lint(
    prompt: PromptLike,
    *,
    params: dict[str, Any] | None = None,
    rules: Iterable[str] | None = None,
    judge: Judge | None = None,
) -> list[LintIssue]:
    """Check a prompt (a string or a chat message list) and return the issues found.

    Rule-based rules (L1, L3, L4, L7) always run and cost nothing. Pass a ``judge`` to
    also run L2, L5 and L6, which cost one judge call each.

    >>> [i.rule for i in lint("What year did India host the 2016 Olympics?")]
    ['L3']
    """
    view = PromptView(_messages(prompt), params or {})
    if rules is not None:
        selected = list(rules)
    else:
        selected = list(RULES) + (list(JUDGE_RULES) if judge is not None else [])
    issues: list[LintIssue] = []
    for rule_id in selected:
        if rule_id in RULES:
            issues.extend(RULES[rule_id](view))
        elif rule_id in JUDGE_RULES:
            if judge is None:
                raise ValueError(f"rule {rule_id} needs a judge: lint(..., judge=...)")
            issues.extend(JUDGE_RULES[rule_id](view, judge))
        else:
            raise KeyError(rule_id)
    return sorted(issues, key=lambda i: i.rule)


__all__ = ["JUDGE_RULES", "RULES", "LintIssue", "LintRule", "lint"]
