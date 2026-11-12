"""Sampling checks: ask again and compare the answers (F4 when they disagree)."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from counterfix import CallRecord, Config, FailureType, Message, SkipCheck
from counterfix.detect.sampling import SamplingCheck, same_meaning_heuristic
from test_judges import FixedJudge


class CyclingCaller:
    """Returns answers from a cycle, recording each call's params."""

    def __init__(self, answers: Sequence[str]):
        self.answers: Iterator[str] = itertools.cycle(answers)
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "fake:target"

    def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
        self.calls.append(params)
        return CallRecord(response_text=next(self.answers), finish_reason="stop")


def rec(answer: str, **params: Any) -> CallRecord:
    return CallRecord(
        messages=[Message(role="user", content="When was the bank founded?")],
        response_text=answer,
        params=params,
    )


def test_heuristic_equivalence() -> None:
    assert same_meaning_heuristic("It was founded in 1921.", "Founded 1921")
    assert not same_meaning_heuristic("It was founded in 1921.", "In 1934, I believe.")
    assert same_meaning_heuristic("Tbilisi is the capital", "the capital is Tbilisi")
    assert not same_meaning_heuristic("Tbilisi", "Atlanta")


def test_consistent_answers_pass_and_params_are_reused() -> None:
    caller = CyclingCaller(["Founded in 1921."])
    check = SamplingCheck(caller, n=4)
    assert check.check(rec("It was founded in 1921.", max_tokens=50, stream=False), Config()) == []
    assert len(caller.calls) == 4
    assert caller.calls[0] == {"temperature": 1.0, "max_tokens": 50}


def test_disagreeing_answers_flag_f4() -> None:
    caller = CyclingCaller(["In 1934.", "In 1902.", "In 1955.", "1921 I think"])
    [f] = SamplingCheck(caller, n=4).check(rec("It was founded in 1921."), Config())
    assert f.failure_type is FailureType.HALLUCINATION and f.inferred
    assert f.evidence[0].data["agreement"] == 0.4 and f.evidence[0].data["n_samples"] == 5
    assert 0 < f.evidence[0].data["entropy"] <= 1


def test_sampling_skips_long_or_empty_answers_and_errors() -> None:
    caller = CyclingCaller(["x"])
    with pytest.raises(SkipCheck):
        SamplingCheck(caller).check(rec("word " * 80), Config())
    with pytest.raises(SkipCheck):
        SamplingCheck(caller).check(rec(""), Config())

    class Broken(CyclingCaller):
        def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
            raise RuntimeError("429")

    with pytest.raises(SkipCheck):
        SamplingCheck(Broken(["x"])).check(rec("In 1921."), Config())


def test_sampling_can_use_a_judge_for_meaning() -> None:
    caller = CyclingCaller(["The bank opened its doors in nineteen twenty-one."])
    judge = FixedJudge("same")
    assert SamplingCheck(caller, n=3, judge=judge).check(rec("In 1921."), Config()) == []
    assert judge.asked and judge.asked[0].name == "same_meaning"
