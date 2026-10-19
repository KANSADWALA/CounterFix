"""Shared fakes. Tests never call a real LLM API."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from counterfix import Config, FailureType, Finding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes `bench` importable


def make_response(
    content: str | None = "Hello!",
    finish_reason: str | None = "stop",
    model: str = "fake-model",
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    """A response shaped like the OpenAI SDK's ChatCompletion object."""
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
    return SimpleNamespace(id="resp-1", model=model, choices=[choice], usage=usage)


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response if response is not None else make_response()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return FakeCompletions.create(self, **kwargs)


class FakeClient:
    """Mimics `client.chat.completions.create` plus some unrelated attributes."""

    def __init__(self, completions: FakeCompletions | None = None):
        self.chat = SimpleNamespace(completions=completions or FakeCompletions())
        self.api_key = "test-key"
        self.models = SimpleNamespace(list=lambda: ["fake-model"])
        self.base_url = "http://localhost/v1"


class FlagAll:
    """A detector that flags every call as F1."""

    name = "flag_all"
    covers = (FailureType.TRUNCATION,)

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def check(self, record: Any, config: Config) -> list[Finding]:
        self.seen.append(record)
        return [Finding(failure_type=FailureType.TRUNCATION, confidence=0.9)]


class Broken:
    name = "broken"
    covers = (FailureType.FORMAT_BREAK,)

    def check(self, record: Any, config: Config) -> list[Finding]:
        raise RuntimeError("detector bug")


@pytest.fixture
def messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hello."},
    ]


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profiles and other stored state go to a temp folder, never the real home."""
    monkeypatch.setenv("COUNTERFIX_HOME", str(tmp_path_factory.mktemp("cfhome")))
