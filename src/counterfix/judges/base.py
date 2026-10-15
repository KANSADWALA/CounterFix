"""Judges answer one structured question about a call, e.g. "does this answer accept a
false premise?". A judge can be one LLM, a panel of LLMs, or a fast decision model."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class JudgeTask(BaseModel):
    """One question for a judge.

    ``labels`` are the only allowed answers. ``instructions`` say what to decide;
    ``material`` is the data to judge (question, answer, context...), passed as text.
    ``extra_fields`` names extra JSON fields the judge should return (e.g. "claims").
    """

    name: str
    instructions: str
    material: dict[str, str]
    labels: tuple[str, ...]
    extra_fields: dict[str, str] = Field(default_factory=dict)


class Vote(BaseModel):
    """One judge's answer inside a panel."""

    judge: str
    label: str
    confidence: float
    reason: str = ""


class Verdict(BaseModel):
    """A judge's decision. ``label`` is one of the task's labels, or "error" if the judge
    could not answer. ``confidence`` is in [0, 1]."""

    task: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    judge: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
    votes: list[Vote] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.label == "error"


@runtime_checkable
class Judge(Protocol):
    """Anything that can answer a JudgeTask."""

    @property
    def name(self) -> str: ...

    def ask(self, task: JudgeTask) -> Verdict: ...
