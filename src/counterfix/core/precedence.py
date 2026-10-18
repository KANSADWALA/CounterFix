"""One mistake, one cause. Several checks can fire on the same underlying problem: a wrong
digit sum is a computation failure (F8) that a hallucination judge also calls a false claim
(F4); an answer cut off by the token limit (F1) also breaks a sentence count (F6). Reporting
both sends the user to two fixes for one problem, so the root cause wins and the other
finding is kept in ``Diagnosis.suppressed`` (nothing is hidden)."""

from __future__ import annotations

from counterfix.core.diagnosis import Evidence, Finding
from counterfix.core.types import FailureType as F

SUPERSEDES: dict[F, tuple[F, ...]] = {
    # a cut-off answer naturally breaks its format and its instructions
    F.TRUNCATION: (F.FORMAT_BREAK, F.DROPPED_INSTRUCTION),
    # a refusal has no claims to check and follows no instructions
    F.OVER_REFUSAL: (F.FORMAT_BREAK, F.DROPPED_INSTRUCTION, F.HALLUCINATION),
    # a wrong calculation, verified in code, is not also a hallucination
    F.COMPUTATION: (F.HALLUCINATION,),
    # going along with a false premise produces false claims: the premise is the cause
    F.FALSE_PREMISE: (F.HALLUCINATION,),
}
MIN_CONFIDENCE = 0.5
"""A finding must be at least this confident to suppress another one."""


def _merge(findings: list[Finding]) -> list[Finding]:
    """Several findings of the same type become one: the most confident, with all evidence."""
    by_type: dict[F, list[Finding]] = {}
    for finding in findings:
        by_type.setdefault(finding.failure_type, []).append(finding)
    merged = []
    for group in by_type.values():
        top = max(group, key=lambda f: f.confidence)
        if len(group) == 1:
            merged.append(top)
            continue
        best = top.model_copy(deep=True)
        for other in group:
            if other is not top:
                best.evidence.extend(e.model_copy() for e in other.evidence)
        best.inferred = all(f.inferred for f in group)
        merged.append(best)
    return merged


def resolve(findings: list[Finding]) -> tuple[list[Finding], list[Finding]]:
    """Return (findings to report, findings superseded by a root cause)."""
    merged = _merge(findings)
    strong = {f.failure_type for f in merged if f.confidence >= MIN_CONFIDENCE}
    kept, suppressed = [], []
    for finding in merged:
        causes = [c for c in strong if finding.failure_type in SUPERSEDES.get(c, ())]
        if causes:
            note = finding.model_copy(deep=True)
            names = ", ".join(f"{c.value} {c.label}" for c in sorted(causes, key=lambda c: c.value))
            note.evidence.append(
                Evidence(
                    detector="precedence",
                    detail=f"reported as {names} instead (the root cause)",
                    data={"superseded_by": [c.value for c in causes]},
                )
            )
            suppressed.append(note)
        else:
            kept.append(finding)
    return kept, suppressed
