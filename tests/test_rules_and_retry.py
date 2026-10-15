from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conftest import FakeClient, make_response
from counterfix import Config, FailureType, FixRule, RuleStore, guard

JSON_PROMPT = [{"role": "user", "content": "Return ONLY a JSON object with key a."}]


class Scripted:
    """Completions whose answer depends on the request, to simulate fixes working."""

    def __init__(self, respond: Callable[[dict[str, Any]], Any]):
        self.respond = respond
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.respond(kwargs)


def needs_tokens(threshold: int) -> Callable[[dict[str, Any]], Any]:
    def respond(kwargs: dict[str, Any]) -> Any:
        if kwargs.get("max_tokens", 0) < threshold:
            return make_response("A long answer that got", "length")
        return make_response("A long answer that got finished.", "stop")

    return respond


# --- rules store ---------------------------------------------------------------------


def test_builtin_rules_and_apply() -> None:
    store = RuleStore()
    rule = store.find(FailureType.TRUNCATION, "any-model")
    assert rule is not None
    assert rule.apply({"max_tokens": 40}) == {"max_tokens": 160}
    assert rule.apply({}) == {"max_tokens": 4096}  # default for a missing limit
    assert rule.apply({"max_tokens": 20000}) == {"max_tokens": 32768}  # capped
    assert rule.changes({"max_tokens": 40, "temperature": 0}) == {"max_tokens": 160}
    assert store.find(FailureType.FALSE_PREMISE, "m") is None


def test_more_specific_and_later_rules_win() -> None:
    store = RuleStore()
    store.add(
        FixRule(
            failure_type=FailureType.TRUNCATION,
            model="openai/gpt-oss-*",
            description="reasoning models need room",
            multiply={"max_tokens": 8},
            source="user",
        )
    )
    assert store.find(FailureType.TRUNCATION, "openai/gpt-oss-120b").description.startswith(
        "reasoning"
    )
    assert store.find(FailureType.TRUNCATION, "qwen/qwen3").source == "builtin"
    store.add(
        FixRule(
            failure_type=FailureType.TRUNCATION,
            model="*",
            description="later",
            set={"max_tokens": 999},
            source="user",
        )
    )
    assert store.find(FailureType.TRUNCATION, "qwen/qwen3").description == "later"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = RuleStore(
        [
            FixRule(
                failure_type=FailureType.FORMAT_BREAK,
                model="llama*",
                description="strict json",
                set={"temperature": 0},
                source="user",
            )
        ]
    )
    path = tmp_path / "rules.json"
    store.save(path)
    loaded = RuleStore.load(path)
    assert len(loaded) == len(store)
    assert loaded.find(FailureType.FORMAT_BREAK, "llama3").description == "strict json"


def test_retry_budget() -> None:
    assert Config().retry_budget == 0
    assert Config(max_extra_calls=3).retry_budget == 0  # flag mode never retries
    assert Config(mode="retry").retry_budget == 1
    assert Config(mode="retry", max_extra_calls=2).retry_budget == 2


# --- retry mode ----------------------------------------------------------------------


def test_flag_mode_never_retries() -> None:
    fake = Scripted(needs_tokens(100))
    client = guard(FakeClient(fake))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=[], max_tokens=40)
    assert len(fake.calls) == 1 and not response.diagnosis.ok and response.diagnosis.retries == []


def test_retry_fixes_truncation_and_returns_better_response() -> None:
    fake = Scripted(needs_tokens(100))
    client = guard(FakeClient(fake), config=Config(mode="retry"))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=[], max_tokens=40)
    assert [c["max_tokens"] for c in fake.calls] == [40, 160]
    assert response.choices[0].message.content.endswith("finished.")
    assert response.diagnosis.ok
    [attempt] = response.diagnosis.retries
    assert attempt.outcome == "fixed" and attempt.changes == {"max_tokens": 160}
    assert attempt.source == "prescription"
    assert [f.failure_type for f in response.diagnosis.retried_from] == [FailureType.TRUNCATION]


def test_retry_that_does_not_help_keeps_original() -> None:
    fake = Scripted(needs_tokens(10_000))
    client = guard(FakeClient(fake), config=Config(mode="retry"))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=[], max_tokens=40)
    assert len(fake.calls) == 2
    assert response.choices[0].finish_reason == "length"  # the original response
    [attempt] = response.diagnosis.retries
    assert attempt.outcome == "no_better"


def test_retry_respects_budget_and_can_chain() -> None:
    fake = Scripted(needs_tokens(2000))
    client = guard(
        FakeClient(fake),
        config=Config(mode="retry", max_extra_calls=3),  # type: ignore[arg-type]
    )
    response = client.chat.completions.create(model="m", messages=[], max_tokens=40)
    # 40 -> 160 (still cut: no_better, stop). Chaining continues only after an improvement.
    assert [c["max_tokens"] for c in fake.calls] == [40, 160]
    assert response.diagnosis.retries[0].outcome == "no_better"


def test_user_rule_overrides_prescription() -> None:
    fake = Scripted(needs_tokens(1000))
    store = RuleStore(
        [
            FixRule(
                failure_type=FailureType.TRUNCATION,
                model="m",
                description="big budget",
                set={"max_tokens": 4000},
                source="user",
            )
        ]
    )
    client = guard(FakeClient(fake), config=Config(mode="retry", rules=store))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=[], max_tokens=40)
    assert fake.calls[1]["max_tokens"] == 4000
    assert response.diagnosis.retries[0].source == "rule:big budget"


def test_retry_fixes_broken_json_with_json_mode() -> None:
    def respond(kwargs: dict[str, Any]) -> Any:
        if kwargs.get("response_format") == {"type": "json_object"}:
            return make_response('{"a": 1}')
        return make_response('Sure! {"a": 1}')

    fake = Scripted(respond)
    client = guard(FakeClient(fake), config=Config(mode="retry"))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=JSON_PROMPT)
    assert response.choices[0].message.content == '{"a": 1}'
    assert response.diagnosis.retries[0].changes == {"response_format": {"type": "json_object"}}


def test_retry_error_is_recorded_not_raised() -> None:
    calls = {"n": 0}

    def respond(kwargs: dict[str, Any]) -> Any:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("400: response_format not supported")
        return make_response('Sure! {"a": 1}')

    client = guard(FakeClient(Scripted(respond)), config=Config(mode="retry"))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=JSON_PROMPT)
    assert response.choices[0].message.content == 'Sure! {"a": 1}'
    [attempt] = response.diagnosis.retries
    assert attempt.outcome == "error" and "not supported" in (attempt.error or "")


def test_unfixable_findings_are_not_retried() -> None:
    fake = Scripted(lambda kwargs: make_response("I'm sorry, but I can't help with that."))
    client = guard(FakeClient(fake), config=Config(mode="retry"))  # type: ignore[arg-type]
    response = client.chat.completions.create(model="m", messages=[])
    assert response.diagnosis.top.failure_type is FailureType.OVER_REFUSAL
    assert len(fake.calls) == 1 and response.diagnosis.retries == []


def test_async_retry() -> None:
    sync = Scripted(needs_tokens(100))

    class AsyncScripted:
        async def create(self, **kwargs: Any) -> Any:
            return sync.create(**kwargs)

    client = guard(FakeClient(AsyncScripted()), config=Config(mode="retry"))  # type: ignore[arg-type]
    response = asyncio.run(client.chat.completions.create(model="m", messages=[], max_tokens=40))
    assert response.diagnosis.ok and response.diagnosis.retries[0].outcome == "fixed"
