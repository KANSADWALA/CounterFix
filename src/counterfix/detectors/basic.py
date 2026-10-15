"""Checks for outcomes that are not model failures, or that come before any other check."""

from __future__ import annotations

from counterfix.core.config import Config
from counterfix.core.diagnosis import Evidence, Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType

FILTER_REASONS = frozenset(
    {"content_filter", "safety", "recitation", "blocklist", "prohibited_content"}
)
LENGTH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


class ProviderFilterDetector:
    """The provider's content filter stopped the answer. Kept apart from F3 over-refusal:
    the model did not refuse; the provider blocked the output."""

    name = "provider_filter"
    covers = (FailureType.PROVIDER_FILTER,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        reason = (record.finish_reason or "").lower()
        if reason not in FILTER_REASONS:
            return []
        return [
            Finding(
                failure_type=FailureType.PROVIDER_FILTER,
                confidence=1.0,
                evidence=[
                    Evidence(
                        detector=self.name,
                        detail=f"the provider stopped the answer (finish_reason={reason!r})",
                    )
                ],
            )
        ]


class EmptyOutputDetector:
    """No visible answer and no tool call, for a reason other than length or filtering."""

    name = "empty_output"
    covers = (FailureType.EMPTY_OUTPUT,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        reason = (record.finish_reason or "").lower()
        if reason in LENGTH_REASONS or reason in FILTER_REASONS:
            return []  # explained by truncation or the provider filter
        if record.tool_calls or record.answer_text:
            return []
        detail = "the model returned no visible answer"
        if record.has_reasoning:
            detail += " (it produced reasoning only)"
        return [
            Finding(
                failure_type=FailureType.EMPTY_OUTPUT,
                confidence=0.9,
                evidence=[Evidence(detector=self.name, detail=detail)],
            )
        ]
