"""Token confidence from logprobs, and perplexity."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from counterfix.core.diagnosis import Diagnosis, Evidence, Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.detect.confidence import describe, mean_probability
from counterfix.evals.metrics import get_metric
from counterfix.evals.perplexity import windows_for
from counterfix.integrations.openai_compat import record_from_openai


def _choice_with_logprobs(tokens: list[tuple[str, float]]) -> Any:
    content = [SimpleNamespace(token=t, logprob=lp) for t, lp in tokens]
    message = SimpleNamespace(
        role="assistant", content="".join(t for t, _ in tokens), tool_calls=None
    )
    return SimpleNamespace(index=0, message=message, finish_reason="stop",
                           logprobs=SimpleNamespace(content=content))  # fmt: skip


def test_logprobs_summarised_on_the_record() -> None:
    choice = _choice_with_logprobs([("Paris", -0.01), (" is", -0.2), (" 1889", -3.5)])
    rec = record_from_openai(
        {"messages": [], "logprobs": True}, SimpleNamespace(model="m", choices=[choice], usage=None)
    )
    s = rec.metadata["logprobs"]
    assert s["n"] == 3 and s["min"] == -3.5 and s["lowest"][0][0] == " 1889"
    assert mean_probability(rec) == pytest.approx(math.exp((-0.01 - 0.2 - 3.5) / 3))
    assert "least likely" in (describe(rec) or "")
    value = get_metric("confidence").score(rec)
    assert value.value == pytest.approx(mean_probability(rec)) and value.passed is None
    low = get_metric("confidence:0.5").score(rec)
    assert low.passed is False
    assert get_metric("confidence").score(CallRecord(response_text="x")).value is None


def test_stream_keeps_logprobs() -> None:
    from counterfix.integrations.streams import _OpenAIParts

    parts = _OpenAIParts()
    for tok, lp in (("Hi", -0.1), ("!", -0.5)):
        delta = SimpleNamespace(
            content=tok, reasoning=None, reasoning_content=None, tool_calls=None
        )
        logprobs = SimpleNamespace(content=[SimpleNamespace(token=tok, logprob=lp)])
        choice = SimpleNamespace(delta=delta, finish_reason=None, logprobs=logprobs)
        parts.feed(SimpleNamespace(model="m", usage=None, choices=[choice]))
    rec = record_from_openai({"messages": []}, parts.response())
    assert rec.metadata["logprobs"]["n"] == 2


def test_caller_drops_logprobs_the_provider_refuses() -> None:
    from counterfix.core.caller import OpenAICompatibleCaller

    class BadRequestError(Exception):
        status_code = 400

    sent: list[dict[str, Any]] = []

    def create(**req: Any) -> Any:
        sent.append(req)
        if "logprobs" in req:
            raise BadRequestError("`logprobs` is not supported with this model")
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            model="m", choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    rec = OpenAICompatibleCaller(client, "m").complete(
        [{"role": "user", "content": "x"}], logprobs=True, top_logprobs=1
    )
    assert (
        rec.response_text == "ok" and "logprobs" not in sent[-1] and "top_logprobs" not in sent[-1]
    )


def test_confidence_added_to_a_judged_f4() -> None:
    from counterfix.diagnose import _add_confidence

    rec = CallRecord(
        response_text="x",
        metadata={"logprobs": {"n": 2, "mean": -1.0, "min": -2.0, "lowest": [["x", -2.0]]}},
    )
    f4 = Finding(
        failure_type=FailureType.HALLUCINATION,
        confidence=0.8,
        evidence=[Evidence(detector="j", detail="d")],
    )
    d = Diagnosis(record_id=rec.id, findings=[f4])
    _add_confidence(rec, d)
    assert d.findings[0].evidence[-1].detector == "logprobs"


def test_perplexity_windows_score_every_token_once() -> None:
    for seq_len, window, stride in [(10, 4, 2), (3000, 1024, 512), (1024, 1024, 1024), (5, 8, 3)]:
        plan = windows_for(seq_len, window, stride)
        scored: list[int] = []
        for begin, end, target in plan:
            assert end - begin <= window and target <= end - begin
            scored += list(range(end - target, end))
        assert scored == list(range(seq_len))  # each position once (position 0 has no loss)
