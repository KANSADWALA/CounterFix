"""OpenTelemetry: send diagnoses to your tracing backend, or check GenAI spans you already have.

Export (needs ``opentelemetry-api``; ``pip install "counterfix[otel]"``):

>>> client = guard(OpenAI(), config=Config(on_diagnosis=counterfix.otel.export()))  # doctest: +SKIP

Each diagnosis becomes a short ``counterfix.diagnosis`` span (a child of the current span,
so it sits under your request) with ``gen_ai.*`` and ``counterfix.*`` attributes: the
failure codes, the top finding, its confidence and the suggested fix, plus token counts,
latency and TTFT.

Speed metrics under the GenAI metric names (duration, time to first chunk, token usage):

>>> client = guard(OpenAI(), config=Config(on_diagnosis=counterfix.otel.metrics()))
...                                                            # doctest: +SKIP

Read (apps already instrumented with OpenTelemetry GenAI conventions, e.g. OpenLLMetry):

>>> provider.add_span_processor(counterfix.otel.SpanChecker())  # doctest: +SKIP

runs the live checks on every finished chat span and sends the result to
``Config.on_diagnosis``. It understands ``gen_ai.input.messages`` /
``gen_ai.output.messages`` (JSON) and the older ``gen_ai.prompt.N.*`` /
``gen_ai.completion.N.*`` attributes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from counterfix.core.config import Config, DiagnosisHook
from counterfix.core.convert import record_from_span as record_from_span
from counterfix.core.detector import Detector
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord


def attributes(record: CallRecord, diagnosis: Diagnosis) -> dict[str, Any]:
    """Span attributes for one diagnosis (OpenTelemetry accepts str, int, float, bool, lists)."""
    attrs: dict[str, Any] = {
        "counterfix.ok": diagnosis.ok,
        "counterfix.failures": [f.failure_type.value for f in diagnosis.findings],
        "counterfix.summary": diagnosis.summary(),
    }
    if record.model:
        attrs["gen_ai.request.model"] = record.model
    if record.provider:
        attrs["gen_ai.system"] = record.provider
    top = diagnosis.top
    if top is not None:
        attrs["counterfix.top"] = top.failure_type.value
        attrs["counterfix.top.confidence"] = top.confidence
        attrs["counterfix.top.detail"] = "; ".join(e.detail for e in top.evidence)[:500]
        if top.prescription is not None:
            attrs["counterfix.top.fix"] = top.prescription.description
            attrs["counterfix.top.fix_verified"] = top.prescription.verified
    if diagnosis.judge:
        attrs["counterfix.judge"] = diagnosis.judge
    usage = record.usage
    if usage is not None and usage.prompt_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = usage.prompt_tokens
    if usage is not None and usage.completion_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = usage.completion_tokens
    if record.latency_ms is not None:
        attrs["counterfix.latency_ms"] = round(record.latency_ms, 1)
    if record.timing is not None and record.timing.ttft_ms is not None:
        attrs["counterfix.ttft_ms"] = round(record.timing.ttft_ms, 1)
    return attrs


def export(tracer: Any = None) -> DiagnosisHook:
    """A ``Config.on_diagnosis`` / ``on_judged`` hook that writes each diagnosis as a span."""
    if tracer is None:
        from opentelemetry import trace

        tracer = trace.get_tracer("counterfix")

    def hook(record: CallRecord, diagnosis: Diagnosis) -> None:
        with tracer.start_as_current_span("counterfix.diagnosis") as span:
            span.set_attributes(attributes(record, diagnosis))

    return hook


def metrics(meter: Any = None) -> DiagnosisHook:
    """A ``Config.on_diagnosis`` hook that records speed and token metrics under the
    OpenTelemetry GenAI names: ``gen_ai.client.operation.duration``,
    ``gen_ai.client.operation.time_to_first_chunk`` (streamed calls) and
    ``gen_ai.client.token.usage``. Durations are in seconds, as the conventions ask."""
    if meter is None:
        from opentelemetry import metrics as otel_metrics

        meter = otel_metrics.get_meter("counterfix")
    duration = meter.create_histogram(
        "gen_ai.client.operation.duration", unit="s", description="GenAI operation duration"
    )
    first_chunk = meter.create_histogram(
        "gen_ai.client.operation.time_to_first_chunk",
        unit="s",
        description="Time to receive the first chunk of a streamed response",
    )
    tokens = meter.create_histogram(
        "gen_ai.client.token.usage", unit="{token}", description="Input and output tokens used"
    )

    def hook(record: CallRecord, diagnosis: Diagnosis) -> None:
        base: dict[str, Any] = {"gen_ai.operation.name": "chat"}
        if record.model:
            base["gen_ai.request.model"] = record.model
        if record.provider:
            base["gen_ai.provider.name"] = record.provider
        if record.error:
            base["error.type"] = record.error.split(":", 1)[0][:64]
        if record.latency_ms is not None:
            duration.record(record.latency_ms / 1000, base)
        if record.timing is not None and record.timing.ttft_ms is not None:
            first_chunk.record(record.timing.ttft_ms / 1000, base)
        usage = record.usage
        if usage is not None:
            if usage.prompt_tokens is not None:
                tokens.record(usage.prompt_tokens, {**base, "gen_ai.token.type": "input"})
            if usage.completion_tokens is not None:
                tokens.record(usage.completion_tokens, {**base, "gen_ai.token.type": "output"})

    return hook


class SpanChecker:
    """An OpenTelemetry span processor that runs the live checks on finished GenAI spans.

    Only spans with model output (``gen_ai.output.messages`` or ``gen_ai.completion.*``)
    are checked; others pass by untouched. The span itself is never changed.
    """

    def __init__(
        self, *, config: Config | None = None, detectors: Sequence[Detector] | None = None
    ):
        from counterfix.integrations.openai_compat import _Guard

        if detectors is None:
            from counterfix.detect import default_detectors

            detectors = default_detectors()
        self._state = _Guard(config or Config(), detectors, provider=None)
        self.last_diagnosis: Diagnosis | None = None

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def _on_ending(self, span: Any) -> None:
        pass  # called by newer SDKs (1.4x) just before on_end; nothing to do here

    def on_end(self, span: Any) -> None:
        attrs = dict(getattr(span, "attributes", None) or {})
        if "gen_ai.output.messages" not in attrs and "gen_ai.completion.0.content" not in attrs:
            return
        try:
            start, end = getattr(span, "start_time", None), getattr(span, "end_time", None)
            ms = (end - start) / 1e6 if start and end else None
            record = record_from_span(attrs, latency_ms=ms)
        except Exception:
            if not self._state.config.fail_open:
                raise
            return
        self.last_diagnosis = self._state.check(record, span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
