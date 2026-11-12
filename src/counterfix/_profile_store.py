"""Stored model profiles: the Profile model and where profiles are saved and loaded.

Split from :mod:`counterfix.profiles` so the fix engine can read a profile without
importing the code that measures one. Import from ``counterfix.profiles``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROBES = Path(__file__).with_name("data") / "probes.jsonl"
SHIPPED = Path(__file__).with_name("data") / "profiles"  # measured profiles, see bench/

TASKS = {
    "F1": "long answers under a token limit",
    "F2": "JSON and structured output",
    "F3": "sensitive but harmless requests",
    "F4": "facts it may not know",
    "F5": "questions with a false premise",
    "F6": "exact formatting instructions",
    "F7": "reworded prompts",
    "F8": "maths and counting",
    "F9": "vague requests",
}


class Profile(BaseModel):
    """One model's measured behaviour on the probe suite, plus what fixed it."""

    model: str
    """The model id as the provider reports it (what ``fix()`` looks up)."""
    spec: str | None = None
    """``provider:model`` it was measured through, e.g. ``groq:openai/gpt-oss-120b``."""
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    judge: str | None = None
    cost_per_mtok: float | None = None
    """Blended price per million tokens, for ``route`` (set it yourself; optional)."""
    categories: dict[str, list[int]] = Field(default_factory=dict)
    """Probe category -> [failed, probes]."""
    fixes: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    """Failure type -> intervention -> [verified, tried]."""
    errors: int = 0
    """Probes that got a provider error (not counted in ``categories``)."""
    judge_skips: int = 0
    """Judge checks that could not run (judge errors, limits); failure rates may be low."""

    def failure_rate(self, category: str) -> float | None:
        failed, n = self.categories.get(category, [0, 0])
        return failed / n if n else None

    def fix_order(self, failure_type: str, names: list[str]) -> list[str]:
        """``names`` reordered: best verified rate first; never-verified after 2 tries last;
        untried ones keep their default place in between."""
        stats = self.fixes.get(failure_type, {})

        def key(item: tuple[int, str]) -> tuple[int, float, int]:
            i, name = item
            verified, tried = stats.get(name, [0, 0])
            if not tried:
                return (1, 0.0, i)
            if verified == 0 and tried >= 2:
                return (2, 0.0, i)
            return (0, -verified / tried, i)

        return [name for _, name in sorted(enumerate(names), key=key)]

    def learn(self, result: Any) -> None:
        """Add one ``FixResult``'s attempts to the fix statistics."""
        for attempt in result.attempts:
            if attempt.failure_type is None:
                continue
            row = self.fixes.setdefault(attempt.failure_type.value, {})
            verified, tried = row.get(attempt.name, [0, 0])
            row[attempt.name] = [verified + attempt.verified, tried + 1]


def home() -> Path:
    return Path(os.environ.get("COUNTERFIX_HOME") or Path.home() / ".counterfix")


def profile_path(model: str) -> Path:
    return home() / "profiles" / (re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".json")


def save(profile: Profile, path: str | Path | None = None) -> Path:
    target = Path(path) if path else profile_path(profile.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return target


def load(model_or_path: str | Path) -> Profile | None:
    """A profile from a file path, or for a model: yours (``~/.counterfix/profiles``) first,
    then the ones shipped with counterfix. ``provider:model`` and bare model ids both work.
    None if there is none."""
    path = Path(model_or_path)
    if path.suffix == ".json" and path.is_file():
        return Profile.model_validate_json(path.read_text(encoding="utf-8"))
    name = str(model_or_path)
    names = [name, name.split(":", 1)[1]] if ":" in name else [name]
    for model in names:
        for candidate in (profile_path(model), SHIPPED / profile_path(model).name):
            if candidate.is_file():
                return Profile.model_validate_json(candidate.read_text(encoding="utf-8"))
    return None
