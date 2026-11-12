"""What :func:`counterfix.fix` returns: each attempt and the overall result."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import Message
from counterfix.core.types import FailureType, PrescriptionKind


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
    metadata: dict[str, Any] = Field(default_factory=dict)
    """What the re-runs' records carry instead of the original's (e.g. new passages)."""
    runs: int = 0
    passed: int = 0
    verified: bool = False
    """For a change: it passed clearly more often than the plain retry (see
    :mod:`counterfix.fixes.verify`). For the plain retry: it passed so often that no change
    could be shown to do better."""
    p_value: float | None = None
    """For a change: the one-sided Fisher exact p-value against the plain retry."""
    seen: list[str] = Field(default_factory=list)
    """What the failed runs still showed, e.g. ["F1 Truncation"]."""
    errors: list[str] = Field(default_factory=list)

    @property
    def observed(self) -> int:
        """Runs that returned a readable answer (calls that errored are not evidence)."""
        return self.runs - len(self.errors)


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
    level: float | None = None
    """The p-value a change had to reach (``alpha`` over the number of changes)."""
    check: str | None = None
    """The name of your own check, when ``fix(check=...)`` decided which runs passed."""

    def explain(self) -> str:
        lines = [f"found: {self.diagnosis.summary()}"]
        if self.check is not None:
            lines.append(f"pass test: your check {self.check} (and no new failure)")
        if self.retry is not None:
            lines.append(f"plain retry: passed {self.retry.passed}/{self.retry.observed}")
        for a in self.attempts:
            mark = "VERIFIED" if a.verified else "no"
            still = ""
            if a.seen:
                found = {
                    f"{f.failure_type.value} {f.failure_type.label}"
                    for f in self.diagnosis.findings
                }
                yours = sorted({t for t in a.seen if t.startswith("your check")})
                kept = sorted(t for t in set(a.seen) if t in found)
                new = sorted(t for t in set(a.seen) if t not in found and t not in yours)
                parts = (
                    yours
                    + ([f"still: {', '.join(kept)}"] if kept else [])
                    + ([f"new: {', '.join(new)}"] if new else [])
                )
                still = f" ({'; '.join(parts)})"
            vs = ""
            if self.retry is not None and a.p_value is not None:
                base = f"{self.retry.passed}/{self.retry.observed}"
                vs = f" vs retry {base}, p={a.p_value:.3f}"
            elif a.p_value is not None:
                vs = f", p={a.p_value:.3f}"
            lines.append(f"tried {a.name}: passed {a.passed}/{a.observed}{vs} -> {mark}{still}")
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
