"""Schema and loader for the seed failure set (bench/seed_failures.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from counterfix import FailureType, Message

SEED_PATH = Path(__file__).with_name("seed_failures.jsonl")


class SeedItem(BaseModel):
    """One prompt that is expected to trigger a given failure type."""

    model_config = ConfigDict(extra="forbid")

    id: str
    failure_type: FailureType
    messages: list[Message] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    why: str
    source: str

    gold: str | None = None  # expected answer, when one exists
    gold_context: str | None = None  # F4: the context that should fix the failure
    control: bool = False  # True for items that should NOT be flagged
    expected: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")  # F2
    constraints: list[str] | None = None  # F6
    variants: list[str] | None = None  # F7
    readings: list[str] | None = None  # F9


def load_seed(path: Path = SEED_PATH) -> list[SeedItem]:
    """Load and validate every row of the seed set."""
    items: list[SeedItem] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                items.append(SeedItem.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path.name} line {line_no}: {exc}") from exc
    return items
