"""guard() and watch() keyword shortcuts: common setups without a Config object."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, FlagAll, make_response
from counterfix import Config
from counterfix.adapters.openai_compat import settings


def _state(client: Any) -> Any:
    return object.__getattribute__(client, "_cf_guard")


def test_no_shortcuts_keeps_the_config_object() -> None:
    config = Config(sample_rate=0.1)
    assert settings(config) is config
    assert settings().mode == "flag"


def test_mode_shortcut_turns_on_retry() -> None:
    client = counterfix.guard(FakeClient(), mode="retry")  # type: ignore[arg-type]
    assert _state(client).config.mode == "retry"
    assert _state(client).config.retry_budget == 1


def test_shortcuts_override_a_passed_config_without_changing_it() -> None:
    config = Config(mode="flag", max_extra_calls=2)
    client = counterfix.guard(FakeClient(), config=config, mode="retry")  # type: ignore[arg-type]
    assert _state(client).config.mode == "retry"
    assert _state(client).config.max_extra_calls == 2  # kept from the config
    assert config.mode == "flag"  # the caller's object is not mutated


def test_judge_and_sample_rate_shortcuts() -> None:
    config = settings(judge="ollama:qwen2.5:3b", sample_rate=0.05)
    assert config.judge == "ollama:qwen2.5:3b"
    assert config.sample_rate == 0.05
    with pytest.raises(ValueError):
        settings(sample_rate=2.0)  # still validated


def test_log_shortcut_writes_every_call_and_keeps_an_existing_hook(
    tmp_path: Path, messages: list[dict[str, str]]
) -> None:
    seen: list[str] = []
    log = tmp_path / "calls.jsonl"
    client = counterfix.guard(
        FakeClient(FakeCompletions(make_response("The answer is"))),  # type: ignore[arg-type]
        config=Config(on_diagnosis=lambda record, diagnosis: seen.append(record.id)),
        detectors=[FlagAll()],
        log=log,
    )
    client.chat.completions.create(model="fake-model", messages=messages)
    client.chat.completions.create(model="fake-model", messages=messages)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and len(seen) == 2
    assert all(json.loads(line) for line in lines)


def test_guarding_twice_with_a_shortcut_rewraps() -> None:
    once = counterfix.guard(FakeClient())  # type: ignore[arg-type]
    assert counterfix.guard(once) is once
    twice = counterfix.guard(once, mode="retry")
    assert twice is not once
    assert _state(twice).config.mode == "retry"


def test_watch_log_shortcut(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"

    @counterfix.watch(model="m", log=log)
    def ask(prompt: str) -> str:
        return "Blue."

    ask("Sky colour?")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_top_level_is_small_and_old_names_still_work() -> None:
    assert len(counterfix.__all__) == 13  # 12 names plus __version__
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for the old names in 0.2.x
        assert counterfix.RuleStore is counterfix.rules.RuleStore
        from counterfix import evaluate, metrics

        assert callable(evaluate) and metrics.__name__ == "counterfix.metrics"
    assert "evaluate" in dir(counterfix)
    with pytest.raises(AttributeError):
        counterfix.not_a_name  # noqa: B018
