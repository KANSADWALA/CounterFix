"""What counterfix returns: findings, their evidence, and prescriptions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.types import FailureType, PrescriptionKind


class Evidence(BaseModel):
    """Why a detector believes a failure happened."""

    detector: str
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)


class Prescription(BaseModel):
    """A suggested change. ``verified`` is True only after a counterfactual re-run passed."""

    kind: PrescriptionKind
    description: str
    patch: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False


class Finding(BaseModel):
    """One suspected failure in a call."""

    failure_type: FailureType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    prescription: Prescription | None = None
    inferred: bool = False  # True when the signal was indirect (e.g. no stop reason)


class Diagnosis(BaseModel):
    """The result of checking one call.

    ``checked`` lists the detectors that ran, ``skipped`` those that could not run
    (with the reason), and ``errors`` any internal errors. Internal errors never
    reach the user's application; they are recorded here instead.
    """

    record_id: str
    findings: list[Finding] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no failure was found."""
        return not self.findings

    @property
    def top(self) -> Finding | None:
        """The most confident finding, or None."""
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: f.confidence)

    def summary(self) -> str:
        """A one-line human summary."""
        if self.ok:
            note = f" ({len(self.errors)} internal errors)" if self.errors else ""
            return f"no failure found by {len(self.checked)} checks{note}"
        parts = [
            f"{f.failure_type.value} {f.failure_type.label} ({f.confidence:.0%})"
            for f in sorted(self.findings, key=lambda f: -f.confidence)
        ]
        return "; ".join(parts)
