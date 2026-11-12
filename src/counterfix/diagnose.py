"""Diagnose mode: run the free live checks and the judge checks on one call.

>>> client = counterfix.guard(OpenAI())                       # doctest: +SKIP
>>> response = client.chat.completions.create(...)            # doctest: +SKIP
>>> report = counterfix.diagnose(response)                    # doctest: +SKIP
>>> print(report.summary())                                   # doctest: +SKIP

Judge checks cost extra model calls, so they never run in live mode; call ``diagnose``
when you want the deeper look (on a sample of traffic, on failures, or in tests). The
judge is chosen as described in :mod:`counterfix.judges.auto` (bring your own key).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from counterfix.core.attach import target_record as _target
from counterfix.core.config import Config
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord
from counterfix.judges.auto import JudgeChoice, resolve_judge


def diagnose(
    target: Any,
    *,
    judge: JudgeChoice | bool | None = None,
    detectors: Sequence[Detector] | None = None,
    config: Config | None = None,
) -> Diagnosis:
    """Check one call with the live checks plus the judge checks.

    ``target`` is a response from a guarded client, its ``Diagnosis``, or a
    :class:`CallRecord`. ``judge`` overrides the automatic choice (a spec such as
    ``"groq:qwen/qwen3.8-27b"``, a list of specs for a panel, or a judge object);
    ``judge=False`` runs only the free live checks (no extra calls, no key needed).
    ``detectors`` replaces the default set entirely.
    """
    from counterfix.detect import default_detectors

    record, client = _target(target)
    if detectors is not None or judge is False:
        checks = list(detectors) if detectors is not None else default_detectors()
        result = run_detectors(record, checks, config or Config())
        result._client = client
        return result
    return _judged(record, client, None if judge is True else judge, config or Config())


def _judged(record: CallRecord, client: Any, judge: Any, config: Config) -> Diagnosis:
    """Live checks plus judge checks with the judge chosen for this record."""
    from counterfix.detect import default_detectors, judge_detectors

    chosen, source = resolve_judge(
        judge,
        target_model=record.model,
        fallback_client=client,
        fallback_provider=record.provider,
    )
    result = run_detectors(record, default_detectors() + judge_detectors(chosen), config)
    _add_confidence(record, result)
    result.judge = f"{chosen.name} ({source})"
    result._client = client
    return result


def _add_confidence(record: CallRecord, result: Diagnosis) -> None:
    """Token probabilities, when the provider gave them, as extra evidence on F4."""
    from counterfix.core.diagnosis import Evidence
    from counterfix.core.types import FailureType
    from counterfix.detect.confidence import describe, of

    text = describe(record)
    if text is None:
        return
    s = of(record) or {}
    for finding in result.findings:
        if finding.failure_type is FailureType.HALLUCINATION:
            finding.evidence.append(
                Evidence(detector="logprobs", detail=text, data={"logprobs": s})
            )
