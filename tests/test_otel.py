from __future__ import annotations

import json
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import Config, FailureType
from counterfix.otel import SpanChecker, record_from_span


def test_record_from_new_and_old_genai_attributes() -> None:
    new = {
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.request.max_tokens": 8,
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "Write an essay."}]}]
        ),
        "gen_ai.output.messages": json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": "Essays"}],
                    "finish_reason": "length",
                }
            ]
        ),
        "gen_ai.usage.output_tokens": 8,
    }
    record = record_from_span(new)
    assert record.prompt_text == "Write an essay." and record.answer_text == "Essays"
    assert record.finish_reason == "length" and record.max_tokens == 8
    old = {
        "gen_ai.system": "openai",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": "Hi",
        "gen_ai.completion.0.content": "Hello!",
        "gen_ai.response.finish_reasons": ("stop",),
    }
    record = record_from_span(old)
    assert record.provider == "openai" and record.answer_text == "Hello!"
    assert record.finish_reason == "stop"


def test_export_and_span_checker_with_the_sdk() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    checker = SpanChecker()
    provider.add_span_processor(checker)
    tracer = provider.get_tracer("test")

    client = counterfix.guard(
        FakeClient(FakeCompletions(make_response("Essays", finish_reason="length"))),
        config=Config(on_diagnosis=counterfix.otel.export(tracer)),
    )
    with tracer.start_as_current_span("request"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "?"}])
    spans = {s.name: s for s in exporter.get_finished_spans()}
    diag: Any = spans["counterfix.diagnosis"]
    assert diag.parent.span_id == spans["request"].context.span_id
    assert diag.attributes["counterfix.top"] == "F1"
    assert list(diag.attributes["counterfix.failures"]) == ["F1"]

    with tracer.start_as_current_span("chat") as span:
        span.set_attribute("gen_ai.prompt.0.content", "Answer in exactly one word: sky colour?")
        span.set_attribute("gen_ai.completion.0.content", "The sky is blue on a clear day.")
    assert checker.last_diagnosis is not None
    assert checker.last_diagnosis.top.failure_type is FailureType.DROPPED_INSTRUCTION
