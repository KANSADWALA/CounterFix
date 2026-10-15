"""The detector interface and the runner that applies detectors to a call."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from counterfix.core.config import Config
from counterfix.core.diagnosis import Diagnosis, Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType

logger = logging.getLogger("counterfix")


class SkipCheck(Exception):
    """Raised by a detector that cannot run on this call (e.g. missing capability)."""


@runtime_checkable
class Detector(Protocol):
    """Anything with a name, the failure types it covers, and a ``check`` method.

    ``check`` returns the findings for one call (an empty list when the call looks
    fine) or raises :class:`SkipCheck` when it cannot judge this call.
    """

    @property
    def name(self) -> str: ...

    @property
    def covers(self) -> tuple[FailureType, ...]: ...

    def check(self, record: CallRecord, config: Config) -> list[Finding]: ...


def run_detectors(
    record: CallRecord,
    detectors: Iterable[Detector],
    config: Config,
) -> Diagnosis:
    """Run each detector on the record and collect a :class:`Diagnosis`.

    A detector that raises is recorded under ``errors`` and the others still run,
    so one broken check never hides the rest or breaks the host application.
    """
    diagnosis = Diagnosis(record_id=record.id)
    for detector in detectors:
        name = getattr(detector, "name", type(detector).__name__)
        try:
            findings: Sequence[Finding] = detector.check(record, config)
        except SkipCheck as skip:
            diagnosis.skipped[name] = str(skip) or "not applicable"
            continue
        except Exception as exc:
            if not config.fail_open:
                raise
            logger.warning("counterfix detector %s failed: %s", name, exc)
            diagnosis.errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        diagnosis.checked.append(name)
        diagnosis.findings.extend(findings)
    return diagnosis
