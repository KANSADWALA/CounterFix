from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from counterfix.judges import JudgeTask

TASK = JudgeTask(
    name="F5",
    instructions="Was the premise accepted?",
    material={"user_request": "q", "model_answer": "a"},
    labels=("no_false_premise", "premise_corrected", "premise_accepted"),
)


class FakeClient:
    def __init__(self, choice: str = "premise_accepted", fail: bool = False):
        self.choice, self.fail, self.requests = choice, fail, []
        self.closed = False

    def system_one(self, state: dict[str, str], questions: dict[str, Any]) -> Any:
        self.requests.append((state, questions))
        if self.fail:
            raise RuntimeError("429 rate limited")
        probs = {"no_false_premise": 0.05, "premise_corrected": 0.1, self.choice: 0.85}
        return SimpleNamespace(
            choices={
                "verdict": SimpleNamespace(choice=self.choice, probabilities=probs, confidence=0.8)
            }
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("typesafe_sdk")
    module.Choice = lambda instructions, criteria: {
        "instructions": instructions,
        "criteria": criteria,
    }  # type: ignore[attr-defined]
    module.TypeSafeClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)


def test_jev_judge_maps_task_to_choice(fake_sdk: None) -> None:
    from counterfix.judges.jev import JevJudge

    client = FakeClient()
    judge = JevJudge(client)
    verdict = judge.ask(TASK)
    assert (verdict.label, verdict.confidence, verdict.judge) == (
        "premise_accepted",
        0.85,
        "typesafe:jev",
    )
    state, questions = client.requests[0]
    assert state == TASK.material
    assert questions["verdict"]["criteria"] == {label: None for label in TASK.labels}
    judge.close()
    assert client.closed


def test_jev_errors_and_bad_labels_become_error_verdicts(fake_sdk: None) -> None:
    from counterfix.judges.jev import JevJudge

    assert JevJudge(FakeClient(fail=True)).ask(TASK).failed
    assert JevJudge(FakeClient(choice="maybe")).ask(TASK).failed


def test_jev_default_client_and_missing_sdk(
    fake_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from counterfix.judges.jev import JevJudge

    assert isinstance(JevJudge().client, FakeClient)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(ImportError):
        JevJudge()
