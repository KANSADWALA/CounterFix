"""Speed as metrics, so an eval can gate on them: latency and TTFT under a limit (ms)."""

from __future__ import annotations

from typing import Any

from counterfix.core.record import CallRecord
from counterfix.metrics.base import Metric


class Latency(Metric):
    """End-to-end latency in ms; passes when at or under the threshold."""

    name = "latency"
    higher_is_better = False
    needs_reference = False

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        if record.latency_ms is None:
            return None, "the call was not timed"
        return record.latency_ms, f"{record.latency_ms:.0f} ms"


class Ttft(Metric):
    """Time to first answer chunk in ms (streamed calls); passes at or under the threshold."""

    name = "ttft"
    higher_is_better = False
    needs_reference = False

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        t = record.timing
        if t is None or t.ttft_ms is None:
            return None, "no TTFT: the call was not streamed"
        return t.ttft_ms, f"first chunk after {t.ttft_ms:.0f} ms"
