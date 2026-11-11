"""Fix rules: (failure type, model) -> a change to the call's parameters.

Rules are what live mode applies when it retries. They start hand-written; later phases
add rules learned from verified fixes. Stored as JSON so they can be reviewed and shared.
"""

from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.types import FailureType

DEFAULTS_FOR_MISSING: dict[str, float] = {"max_tokens": 1024}
_LIMIT_ALIASES = ("max_completion_tokens", "max_output_tokens")


class FixRule(BaseModel):
    """A parameter change that fixes one failure type for models matching ``model``.

    ``model`` is a glob pattern (``*`` matches anything, e.g. ``openai/gpt-oss-*``).
    ``set`` replaces parameters; ``multiply`` scales numeric ones (at least ``floor``,
    at most ``cap``).
    """

    failure_type: FailureType
    model: str = "*"
    description: str
    set: dict[str, Any] = Field(default_factory=dict)
    multiply: dict[str, float] = Field(default_factory=dict)
    cap: dict[str, float] = Field(default_factory=dict)
    floor: dict[str, float] = Field(default_factory=dict)
    source: str = "builtin"

    def matches(self, failure_type: FailureType, model: str | None) -> bool:
        return failure_type is self.failure_type and fnmatchcase(model or "", self.model)

    @property
    def specificity(self) -> int:
        """More literal characters in the pattern means a more specific rule."""
        return len(self.model.replace("*", "").replace("?", ""))

    def apply(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``params`` with this rule's changes applied."""
        new = dict(params)
        new.update(self.set)
        for key, factor in self.multiply.items():
            if key == "max_tokens" and key not in new:  # the limit under its newer name
                key = next((k for k in _LIMIT_ALIASES if k in new), key)
            base = new.get(
                key, DEFAULTS_FOR_MISSING.get("max_tokens" if key in _LIMIT_ALIASES else key)
            )
            if isinstance(base, int | float) and not isinstance(base, bool):
                value = base * factor
                cap = self.cap.get(
                    key, self.cap.get("max_tokens") if key in _LIMIT_ALIASES else None
                )
                floor = self.floor.get(
                    key, self.floor.get("max_tokens") if key in _LIMIT_ALIASES else None
                )
                if floor is not None:
                    value = max(value, floor)
                if cap is not None:
                    value = min(value, cap)
                new[key] = int(value) if isinstance(base, int) else value
        return new

    def changes(self, params: dict[str, Any]) -> dict[str, Any]:
        """Only the parameters this rule would change, with their new values."""
        new = self.apply(params)
        return {k: v for k, v in new.items() if params.get(k) != v}


BUILTIN_RULES: tuple[FixRule, ...] = (
    FixRule(
        failure_type=FailureType.TRUNCATION,
        # a small first raise (5 -> 20) rarely helps, and reasoning models spend the
        # budget before answering; a limit is a ceiling, so a short answer costs no more
        description="Raise max_tokens 4x, to at least 1024 (at most 32768)",
        multiply={"max_tokens": 4},
        floor={"max_tokens": 1024},
        cap={"max_tokens": 32768},
    ),
    FixRule(
        failure_type=FailureType.FORMAT_BREAK,
        description="Turn on JSON mode",
        set={"response_format": {"type": "json_object"}},
    ),
)


class RuleStore:
    """An ordered set of fix rules. The most specific matching rule wins; on a tie, the
    rule added last wins, so user rules override built-in ones."""

    def __init__(self, rules: list[FixRule] | None = None, *, builtins: bool = True):
        self.rules: list[FixRule] = list(BUILTIN_RULES) if builtins else []
        self.rules.extend(rules or [])

    def add(self, rule: FixRule) -> None:
        self.rules.append(rule)

    def find(self, failure_type: FailureType, model: str | None) -> FixRule | None:
        best: FixRule | None = None
        for rule in self.rules:
            if rule.matches(failure_type, model) and (
                best is None or rule.specificity >= best.specificity
            ):
                best = rule
        return best

    def save(self, path: str | Path) -> None:
        """Write the non-built-in rules to a JSON file."""
        custom = [r.model_dump(mode="json") for r in self.rules if r.source != "builtin"]
        Path(path).write_text(json.dumps(custom, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, builtins: bool = True) -> RuleStore:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([FixRule.model_validate(item) for item in data], builtins=builtins)

    def __len__(self) -> int:
        return len(self.rules)
