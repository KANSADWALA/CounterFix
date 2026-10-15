from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from conftest import (
    Broken,
    FakeAsyncCompletions,
    FakeClient,
    FakeCompletions,
    FlagAll,
    make_response,
)
from counterfix import CallRecord, Config, Diagnosis, FailureType, get_diagnosis, guard


def test_same_api_and_diagnosis_attached(messages: list[dict[str, str]]) -> None:
    detector = FlagAll()
    raw = FakeClient()
    client = guard(raw, detectors=[detector], provider="fake")

    response = client.chat.completions.create(model="m", messages=messages, max_tokens=5)

    assert response is raw.chat.completions.response  # the user's object, unchanged type
    assert response.choices[0].message.content == "Hello!"
    assert isinstance(response.diagnosis, Diagnosis)
    assert response.diagnosis.top.failure_type is FailureType.TRUNCATION
    assert get_diagnosis(response) is response.diagnosis
    record = detector.seen[0]
    assert record.prompt_text == "Say hello." and record.max_tokens == 5
    assert record.provider == "fake" and record.latency_ms is not None


def test_other_attributes_pass_through() -> None:
    raw = FakeClient()
    client = guard(raw, detectors=[])
    assert client.api_key == "test-key"
    assert client.models.list() == ["fake-model"]
    client.base_url = "http://other/v1"
    assert raw.base_url == "http://other/v1"
    assert client.unwrapped is raw
    assert guard(client) is client  # guarding twice is a no-op
    assert "counterfix.guard" in repr(client)


def test_provider_errors_reach_the_user_unchanged() -> None:
    error = ValueError("rate limited")
    client = guard(FakeClient(FakeCompletions(error=error)), detectors=[FlagAll()])
    with pytest.raises(ValueError) as caught:
        client.chat.completions.create(model="m", messages=[])
    assert caught.value is error


def test_broken_detector_never_breaks_the_call() -> None:
    client = guard(FakeClient(), detectors=[Broken()])
    response = client.chat.completions.create(model="m", messages=[])
    assert response.choices[0].message.content == "Hello!"
    assert "broken" in response.diagnosis.errors


def test_broken_hook_is_fail_open_by_default() -> None:
    def hook(record: CallRecord, diagnosis: Diagnosis) -> None:
        raise RuntimeError("hook bug")

    client = guard(FakeClient(), detectors=[], config=Config(on_diagnosis=hook))
    response = client.chat.completions.create(model="m", messages=[])
    assert response.choices[0].message.content == "Hello!"


def test_hook_errors_raise_when_fail_open_is_off() -> None:
    def hook(record: CallRecord, diagnosis: Diagnosis) -> None:
        raise RuntimeError("hook bug")

    client = guard(FakeClient(), detectors=[], config=Config(on_diagnosis=hook, fail_open=False))
    with pytest.raises(RuntimeError):
        client.chat.completions.create(model="m", messages=[])


def test_hook_receives_record_and_diagnosis() -> None:
    seen: list[tuple[CallRecord, Diagnosis]] = []
    client = guard(
        FakeClient(),
        detectors=[FlagAll()],
        config=Config(on_diagnosis=lambda r, d: seen.append((r, d))),
    )
    client.chat.completions.create(model="m", messages=[])
    assert len(seen) == 1 and not seen[0][1].ok


def test_disabled_config_skips_everything() -> None:
    detector = FlagAll()
    client = guard(FakeClient(), detectors=[detector], config=Config(enabled=False))
    response = client.chat.completions.create(model="m", messages=[])
    assert detector.seen == [] and get_diagnosis(response) is None


def test_streaming_is_passed_through() -> None:
    detector = FlagAll()
    stream = iter(["chunk"])
    client = guard(FakeClient(FakeCompletions(response=stream)), detectors=[detector])
    assert client.chat.completions.create(model="m", messages=[], stream=True) is stream
    assert detector.seen == []


def test_async_client() -> None:
    raw = FakeClient(FakeAsyncCompletions())
    client = guard(raw, detectors=[FlagAll()])
    response = asyncio.run(client.chat.completions.create(model="m", messages=[]))
    assert response.diagnosis.top.failure_type is FailureType.TRUNCATION


class Slotted:
    __slots__ = ("choices", "model", "usage")

    def __init__(self) -> None:
        base = make_response()
        self.choices, self.model, self.usage = base.choices, base.model, base.usage


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str
    choices: list[Any]


@pytest.mark.parametrize("response_factory", [Slotted, lambda: Frozen(model="m", choices=[])])
def test_responses_that_refuse_new_attributes(response_factory: Any) -> None:
    response = response_factory()
    client = guard(FakeClient(FakeCompletions(response=response)), detectors=[FlagAll()])
    returned = client.chat.completions.create(model="m", messages=[])
    assert returned is response
    assert isinstance(get_diagnosis(returned), Diagnosis)


def test_default_detectors_run_and_find_nothing_on_a_clean_call() -> None:
    response = guard(FakeClient()).chat.completions.create(model="m", messages=[])
    assert response.diagnosis.ok
    assert set(response.diagnosis.checked) == {
        "provider_filter",
        "empty_output",
        "truncation",
        "format",
        "refusal",
    }


def test_default_detectors_flag_truncation() -> None:
    raw = FakeClient(FakeCompletions(response=make_response("A long essay about", "length")))
    response = guard(raw).chat.completions.create(model="m", messages=[], max_tokens=40)
    assert response.diagnosis.top.failure_type is FailureType.TRUNCATION
    assert response.diagnosis.top.prescription.patch == {"max_tokens": 160}
