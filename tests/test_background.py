from __future__ import annotations

import threading
from typing import Any

import pytest

import counterfix
from conftest import FakeClient, FakeCompletions, make_response
from counterfix import Config, background
from counterfix.judges.base import JudgeTask, Verdict


class SlowStub:
    name = "stub"

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate

    def ask(self, task: JudgeTask) -> Verdict:
        if self.gate is not None:
            self.gate.wait(5)
        return Verdict(task=task.name, label="no_claims", confidence=0.9, judge=self.name)


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(background, "_judged_at", background.deque())
    monkeypatch.setattr(background, "stats", dict.fromkeys(background.stats, 0))
    background.flush(5)


def _client(config: Config, finish: str = "stop") -> Any:
    response = make_response("Paris.", model="m", finish_reason=finish)
    return counterfix.guard(FakeClient(FakeCompletions(response)), config=config)


def _call(client: Any) -> Any:
    return client.chat.completions.create(model="m", messages=[{"role": "user", "content": "?"}])


def test_off_by_default_and_flagged_calls_always_go() -> None:
    _call(_client(Config(judge=SlowStub())))
    assert background.stats["queued"] == 0
    seen: list[str | None] = []
    config = Config(
        judge=SlowStub(), sample_rate=0.000001, on_judged=lambda r, d: seen.append(d.judge)
    )
    _call(_client(config))  # passes live checks: almost never sampled
    _call(_client(config, finish="length"))  # flagged by live checks: always judged
    assert background.flush(5)
    assert background.stats["judged"] == 1 and seen and "stub" in (seen[0] or "")


def test_call_returns_before_the_judge_and_cap_skips_extra_calls() -> None:
    gate = threading.Event()
    config = Config(judge=SlowStub(gate), sample_rate=1.0, max_judged_per_hour=2)
    client = _client(config)
    for _ in range(4):
        assert _call(client).choices[0].message.content == "Paris."  # not blocked by the judge
    assert background.stats["queued"] == 2 and background.stats["dropped_cap"] == 2
    gate.set()
    assert background.flush(5) and background.stats["judged"] == 2


def test_full_queue_drops_instead_of_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(background, "_queue", background.queue.Queue(1))
    monkeypatch.setattr(background, "_ensure_worker", lambda: None)  # nothing drains it
    config = Config(judge=SlowStub(), sample_rate=1.0)
    client = _client(config)
    _call(client)
    _call(client)
    assert background.stats["queued"] == 1 and background.stats["dropped_full"] == 1
