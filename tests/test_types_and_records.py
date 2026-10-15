from __future__ import annotations

from types import SimpleNamespace

from conftest import make_response
from counterfix import CallRecord, Diagnosis, FailureType, Finding, Message, record_from_openai


def test_model_failure_types_are_f1_to_f9() -> None:
    model = [t for t in FailureType if t.is_model_failure]
    assert [t.value for t in model] == [f"F{i}" for i in range(1, 10)]
    assert not FailureType.INFRASTRUCTURE.is_model_failure
    assert not FailureType.PROVIDER_FILTER.is_model_failure
    assert all(t.label for t in FailureType)


def test_message_text_handles_parts() -> None:
    m = Message(role="user", content=[{"type": "text", "text": "a"}, {"type": "image_url"}, "b"])
    assert m.text == "a\nb"
    assert Message(role="user", content=None).text == ""


def test_record_properties(messages: list[dict[str, str]]) -> None:
    record = CallRecord(messages=[Message(**m) for m in messages], params={"max_tokens": 50})
    assert record.prompt_text == "Say hello."
    assert record.system_text == "You are terse."
    assert record.max_tokens == 50
    assert CallRecord(params={"max_completion_tokens": 9}).max_tokens == 9
    assert CallRecord().max_tokens is None


def test_record_from_object_response(messages: list[dict[str, str]]) -> None:
    request = {"model": "m", "messages": messages, "temperature": 0, "max_tokens": 20}
    record = record_from_openai(
        request, make_response("Hi", "length"), latency_ms=5.0, provider="groq"
    )
    assert record.response_text == "Hi"
    assert record.finish_reason == "length"
    assert record.model == "fake-model"
    assert record.params == {"temperature": 0, "max_tokens": 20}
    assert record.usage is not None and record.usage.total_tokens == 15
    assert record.provider == "groq" and record.latency_ms == 5.0


def test_record_from_dict_response_with_tool_calls() -> None:
    response = {
        "model": "m",
        "choices": [
            {
                "message": {"content": None, "tool_calls": [{"id": "t1", "type": "function"}]},
                "finish_reason": "tool_calls",
            }
        ],
    }
    record = record_from_openai({"messages": [SimpleNamespace(role="user", content="x")]}, response)
    assert record.tool_calls == [{"id": "t1", "type": "function"}]
    assert record.response_text is None
    assert record.prompt_text == "x"


def test_record_from_empty_response() -> None:
    record = record_from_openai({"model": "m", "messages": []}, {"choices": []})
    assert record.response_text is None and record.model == "m"


def test_diagnosis_summary() -> None:
    d = Diagnosis(record_id="r")
    assert d.ok and d.top is None and "no failure" in d.summary()
    d.findings = [
        Finding(failure_type=FailureType.FALSE_PREMISE, confidence=0.4),
        Finding(failure_type=FailureType.TRUNCATION, confidence=0.9),
    ]
    assert not d.ok
    assert d.top is not None and d.top.failure_type is FailureType.TRUNCATION
    assert d.summary().startswith("F1 Truncation (90%)")


def test_reasoning_field_and_tokens_are_captured() -> None:
    response = make_response(content="", finish_reason="length")
    response.choices[0].message.reasoning = "Let me think about greetings..."
    response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=50)
    record = record_from_openai({"model": "m", "messages": [], "max_tokens": 50}, response)
    assert record.reasoning_text == "Let me think about greetings..."
    assert record.usage is not None and record.usage.reasoning_tokens == 50
    assert record.has_reasoning and record.answer_text == ""


def test_reasoning_content_field_is_captured() -> None:
    response = {"choices": [{"message": {"content": "Hi", "reasoning_content": "hmm"}}]}
    assert record_from_openai({"messages": []}, response).reasoning_text == "hmm"


def test_inline_think_tags_are_separated_from_the_answer() -> None:
    record = CallRecord(response_text="<think>plan the reply</think>\nHello there!")
    assert record.answer_text == "Hello there!" and record.has_reasoning
    cut = CallRecord(response_text="<think>still thinking when the limit hit")
    assert cut.answer_text == "" and cut.has_reasoning
    plain = CallRecord(response_text="Hello")
    assert plain.answer_text == "Hello" and not plain.has_reasoning


def test_gemma_thought_tags_and_thinking_tags() -> None:
    gemma = CallRecord(response_text="<thought>Task: say hello.</thought>Hello there, my friend!")
    assert gemma.answer_text == "Hello there, my friend!" and gemma.has_reasoning
    cut = CallRecord(response_text="<thought>Monsoons. Detailed essay. Explain")
    assert cut.answer_text == "" and cut.has_reasoning
    other = CallRecord(response_text="<THINKING>x</THINKING> ok")
    assert other.answer_text == "ok"
    mismatched = CallRecord(response_text="<think>a</thought> b")  # tags must match
    assert mismatched.answer_text == ""
