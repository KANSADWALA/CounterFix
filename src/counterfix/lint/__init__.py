"""The prompt linter: catch failure-prone prompts before any call is made."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from counterfix.core.record import Message
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
) -> list[LintIssue]:
    """Check a prompt (a string or a chat message list) and return the issues found.

    >>> [i.rule for i in lint("What year did India host the 2016 Olympics?")]
    ['L3']
    """
    view = PromptView(_messages(prompt), params or {})
    selected = list(rules) if rules is not None else list(RULES)
    issues: list[LintIssue] = []
    for rule_id in selected:
        issues.extend(RULES[rule_id](view))
    return issues


__all__ = ["RULES", "LintIssue", "LintRule", "lint"]
