from __future__ import annotations

from collections.abc import Callable
from typing import Any

import counterfix
from counterfix import CallRecord, FailureType, Message
from counterfix.judges.base import JudgeTask, Verdict


class Scripted:
    """A fake model: ``answer(messages, params)`` decides each reply."""

    name = "fake:m"

    def __init__(self, answer: Callable[[list[Message], dict[str, Any]], tuple[str, str]]):
        self.answer = answer
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        msgs = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
        self.calls.append((msgs, params))
        text, finish = self.answer(msgs, params)
        return CallRecord(
            model="m", messages=msgs, params=params, response_text=text, finish_reason=finish
        )


def cut_off() -> CallRecord:
    return CallRecord(
        model="m",
        messages=[Message(role="user", content="List the planets, one per line.")],
        params={"max_tokens": 20},
        response_text="Mercury\nVenus",
        finish_reason="length",
    )


def test_raising_max_tokens_is_verified_and_written_back() -> None:
    model = Scripted(
        lambda m, p: (
            ("Mercury ... Neptune.", "stop") if p["max_tokens"] > 20 else ("Mercury", "length")
        )
    )
    result = counterfix.fix(cut_off(), judge=False, caller=model)
    assert result.retry is not None and result.retry.passed == 0
    assert result.fix is not None and result.fix.name == "raise_max_tokens"
    assert result.fix.changed == {"max_tokens": 80} and result.fix.passed == 2
    assert result.calls == 2 + 2  # retry settled after 2 fails, fix after 2 passes
    finding = result.diagnosis.findings[0]
    assert finding.prescription is not None and finding.prescription.verified
    assert "VERIFIED" in result.explain()


def test_a_plain_retry_that_passes_claims_no_fix() -> None:
    model = Scripted(lambda m, p: ("Mercury ... Neptune.", "stop"))
    result = counterfix.fix(cut_off(), judge=False, caller=model)
    assert result.fix is None and result.retry is not None and result.retry.verified
    assert "plain retry passes" in result.stopped


def test_one_lucky_pass_is_not_a_fix_and_escalation_is_listed() -> None:
    replies = iter(
        [("x", "length")] * 2  # retry: two fails, settled
        + [("Mercury ... Neptune.", "stop"), ("x", "length"), ("x", "length")]
    )
    model = Scripted(lambda m, p: next(replies, ("x", "length")))
    result = counterfix.fix(cut_off(), judge=False, caller=model)
    assert result.fix is None and result.attempts[0].passed == 1
    assert "ask for a shorter answer" in result.escalation


def test_budget_and_judge_only_failures() -> None:
    model = Scripted(lambda m, p: ("x", "length"))
    result = counterfix.fix(cut_off(), judge=False, caller=model, max_runs=1)
    assert result.calls == 1 and result.fix is None

    premise = CallRecord(
        model="m",
        messages=[Message(role="user", content="Why did Einstein win three Nobel Prizes?")],
        response_text="Because he was brilliant.",
    )

    class PremiseJudge:
        name = "stub"

        def ask(self, task: JudgeTask) -> Verdict:
            text = str(task.material.get("model_answer", ""))
            if task.name.startswith("F5"):
                label = "premise_rejected" if "only one" in text else "premise_accepted"
                return Verdict(task=task.name, label=label, confidence=0.9, judge="stub")
            return Verdict(task=task.name, label=task.labels[-1], confidence=0.9, judge="stub")

    def reply(messages: list[Message], params: dict[str, Any]) -> tuple[str, str]:
        checked = "assumes something false" in messages[0].text
        return ("He won only one Nobel Prize." if checked else "Because he was brilliant."), "stop"

    no_judge = counterfix.fix(premise, judge=False, caller=Scripted(reply))
    assert no_judge.calls == 0  # nothing could confirm a fix, so nothing was re-run

    result = counterfix.fix(premise, judge=PremiseJudge(), caller=Scripted(reply))
    assert result.diagnosis.top is not None
    assert result.diagnosis.top.failure_type is FailureType.FALSE_PREMISE
    assert result.fix is not None and result.fix.name == "check_premises"
    assert result.fix.messages is not None and result.fix.messages[0].role == "system"


def test_fragility_check_and_its_fix() -> None:
    from counterfix.detectors import FragilityCheck

    record = CallRecord(
        model="m",
        messages=[Message(role="user", content="Is 'not bad at all' positive or negative?")],
        response_text="Positive.",
        metadata={
            "variants": ["Sentiment of 'not bad at all'?", "Positive or negative: 'not bad at all'"]
        },
    )

    def fragile(messages: list[Message], params: dict[str, Any]) -> tuple[str, str]:
        pinned = any("same form every time" in m.text for m in messages)
        reworded = "Sentiment" in messages[-1].text
        return ("Negative." if reworded and not pinned else "Positive."), "stop"

    report = counterfix.diagnose(record, detectors=[FragilityCheck(Scripted(fragile))])
    assert report.top is not None and report.top.failure_type is FailureType.PROMPT_FRAGILITY
    assert "1 of 2 rewordings" in report.top.evidence[0].detail

    result = counterfix.fix(report, judge=False, caller=Scripted(fragile), check_retry=False)
    assert result.fix is not None and result.fix.name == "fixed_answer_format"


def test_unreachable_model_stops_at_once_without_advice() -> None:
    class Down:
        name = "down"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Any, **params: Any) -> CallRecord:
            self.calls += 1
            raise ConnectionError("Connection refused")

    down = Down()
    result = counterfix.fix(cut_off(), judge=False, caller=down)
    assert down.calls == 1 and result.calls == 1
    assert result.fix is None and result.escalation == []
    assert result.unreachable and "Connection refused" in result.unreachable
    assert "could not reach the model" in result.explain()

    flaky = iter([ConnectionError("blip"), ("Mercury ... Neptune.", "stop"), ("x", "length")])

    def once_down(m: list[Message], p: dict[str, Any]) -> tuple[str, str]:
        item = next(flaky, ("Mercury ... Neptune.", "stop"))
        if isinstance(item, Exception):
            raise item
        return item

    later = counterfix.fix(cut_off(), judge=False, caller=Scripted(once_down), check_retry=False)
    assert later.unreachable is not None  # the very first call failed: treated as down
