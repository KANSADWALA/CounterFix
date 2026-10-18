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

from counterfix.adapters import get_diagnosis
from counterfix.core.config import Config
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord
from counterfix.judges.auto import JudgeChoice, resolve_judge


def _target(target: Any) -> tuple[CallRecord, Any]:
    """(record, raw client or None) from a record, a diagnosis or a guarded response."""
    if isinstance(target, CallRecord):
        return target, None
    diagnosis = target if isinstance(target, Diagnosis) else get_diagnosis(target)
    if diagnosis is None or diagnosis.record is None:
        raise TypeError(
            "diagnose() needs a CallRecord, or a response from a client wrapped with "
            "counterfix.guard()"
        )
    return diagnosis.record, diagnosis._client


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
    from counterfix.detectors import default_detectors

    record, client = _target(target)
    if detectors is not None or judge is False:
        checks = list(detectors) if detectors is not None else default_detectors()
        result = run_detectors(record, checks, config or Config())
        result._client = client
        return result
    return _judged(record, client, None if judge is True else judge, config or Config())


def _judged(record: CallRecord, client: Any, judge: Any, config: Config) -> Diagnosis:
    """Live checks plus judge checks with the judge chosen for this record."""
    from counterfix.detectors import default_detectors, judge_detectors

    chosen, source = resolve_judge(
        judge,
        target_model=record.model,
        fallback_client=client,
        fallback_provider=record.provider,
    )
    result = run_detectors(record, default_detectors() + judge_detectors(chosen), config)
    result.judge = f"{chosen.name} ({source})"
    result._client = client
    return result
