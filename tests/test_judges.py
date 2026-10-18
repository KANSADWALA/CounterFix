from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeClient, FakeCompletions, make_response
from counterfix import CallRecord, Config, FailureType, Message
from counterfix.core.caller import ModelCaller, OpenAICompatibleCaller
from counterfix.detectors import (
    AmbiguityCheck,
    ComputationCheck,
    DroppedInstructionCheck,
    FalsePremiseCheck,
    HallucinationCheck,
    OverRefusalCheck,
    judge_detectors,
)
from counterfix.detectors.judged import answer_contains
from counterfix.judges import (
    CachedJudge,
    JudgePanel,
    JudgeTask,
    LLMJudge,
    Verdict,
    parse_reply,
    render,
)
from counterfix.safe_eval import UnsafeExpression, safe_eval

TASK = JudgeTask(
    name="t",
    instructions="Decide.",
    material={"q": "Q?", "a": "A."},
    labels=("yes", "no"),
    extra_fields={"why": "text"},
)


class ScriptCaller:
    """A ModelCaller that replies with a scripted text (or raises)."""

    def __init__(self, reply: str | Callable[[dict[str, Any]], str], name: str = "fake:judge"):
        self.reply, self._name = reply, name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
        self.calls.append(params)
        text = self.reply(params) if callable(self.reply) else self.reply
        return CallRecord(response_text=text, finish_reason="stop")


def reply(label: str, confidence: float = 0.9, **extra: Any) -> str:
    return json.dumps({"label": label, "confidence": confidence, "reason": "because", **extra})


class FixedJudge:
    def __init__(self, label: str, confidence: float = 0.9, name: str = "fixed", **extra: Any):
        self.verdict = dict(label=label, confidence=confidence, extra=extra)
        self._name = name
        self.asked: list[JudgeTask] = []

    @property
    def name(self) -> str:
        return self._name

    def ask(self, task: JudgeTask) -> Verdict:
        self.asked.append(task)
        return Verdict(task=task.name, judge=self._name, reason="r", **self.verdict)


# --- callers -------------------------------------------------------------------------


def test_openai_caller_builds_request_and_unwraps_guard() -> None:
    fake = FakeCompletions(response=make_response("hi"))
    from counterfix import guard

    caller = OpenAICompatibleCaller(
        guard(FakeClient(fake)), "m", provider="groq", defaults={"temperature": 0}
    )
    assert isinstance(caller, ModelCaller) and caller.name == "groq:m"
    record = caller.complete([Message(role="user", content="x")], max_tokens=5)
    assert record.response_text == "hi" and record.provider == "groq"
    assert fake.calls[0] == {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0,
        "max_tokens": 5,
    }
    assert not hasattr(fake.response, "diagnosis")  # raw client used, no live checks


# --- LLM judge -----------------------------------------------------------------------


def test_render_lists_material_labels_and_fields() -> None:
    text = render(TASK)
    assert "<q>\nQ?\n</q>" in text and '"yes", "no"' in text and '"why": <text>' in text


@pytest.mark.parametrize(
    "raw",
    [
        '{"label": "yes", "confidence": 0.8}',
        '```json\n{"label": "yes", "confidence": 0.8}\n```',
        'Sure! {"label": "yes", "confidence": 0.8,}',
    ],
)
def test_parse_reply_tolerates_noise(raw: str) -> None:
    assert parse_reply(raw) == {"label": "yes", "confidence": 0.8}


def test_llm_judge_valid_reply_and_think_tags() -> None:
    judge = LLMJudge(ScriptCaller("<think>hmm</think>" + reply("no", 0.7, why="x")))
    v = judge.ask(TASK)
    assert (v.label, v.confidence, v.extra, v.judge) == ("no", 0.7, {"why": "x"}, "fake:judge")


def test_llm_judge_bad_label_and_errors_become_error_verdicts() -> None:
    assert LLMJudge(ScriptCaller(reply("maybe"))).ask(TASK).failed
    assert LLMJudge(ScriptCaller("not json at all")).ask(TASK).failed

    def boom(params: dict[str, Any]) -> str:
        raise RuntimeError("rate limited")

    v = LLMJudge(ScriptCaller(boom)).ask(TASK)
    assert v.failed and "rate limited" in v.reason


def test_llm_judge_falls_back_when_json_mode_is_refused() -> None:
    def respond(params: dict[str, Any]) -> str:
        if "response_format" in params:
            raise RuntimeError("400: response_format is not supported")
        return reply("yes")

    caller = ScriptCaller(respond)
    judge = LLMJudge(caller)
    assert judge.ask(TASK).label == "yes" and judge.json_mode is False
    judge.ask(TASK)
    assert "response_format" not in caller.calls[-1]


def test_confidence_is_clamped() -> None:
    assert LLMJudge(ScriptCaller(reply("yes", 7))).ask(TASK).confidence == 1.0


# --- panel and cache -----------------------------------------------------------------


def test_panel_majority_and_confidence() -> None:
    panel = JudgePanel(
        [FixedJudge("yes", 0.9, "a"), FixedJudge("yes", 0.7, "b"), FixedJudge("no", 0.9, "c")]
    )
    v = panel.ask(TASK)
    assert v.label == "yes" and v.confidence == pytest.approx(2 / 3 * 0.8, abs=1e-4)
    assert [vote.judge for vote in v.votes] == ["a", "b", "c"]


def test_panel_ties_and_failures() -> None:
    tie = JudgePanel([FixedJudge("yes", 0.6, "a"), FixedJudge("no", 0.9, "b")], parallel=False)
    assert tie.ask(TASK).label == "no"
    partly = JudgePanel([FixedJudge("error", 0, "a"), FixedJudge("no", 0.8, "b")])
    assert partly.ask(TASK).label == "no" and partly.ask(TASK).confidence == 0.8
    assert JudgePanel([FixedJudge("error", 0, "a")]).ask(TASK).failed
    with pytest.raises(ValueError):
        JudgePanel([])


def test_cache_hits_skip_errors_and_persist(tmp_path: Path) -> None:
    inner = FixedJudge("yes")
    path = tmp_path / "cache.jsonl"
    cached = CachedJudge(inner, path)
    cached.ask(TASK)
    cached.ask(TASK)
    assert len(inner.asked) == 1 and cached.hits == 1
    assert CachedJudge(FixedJudge("no"), path).ask(TASK).label == "yes"  # same judge name
    failing = CachedJudge(FixedJudge("error", 0, name="bad"), tmp_path / "c2.jsonl")
    failing.ask(TASK)
    assert len(failing) == 0


# --- safe evaluator ------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,value",
    [
        ("48763 * 9284", 452715692),
        ("'strawberry'.count('r')", 3),
        ("'encyclopedia'[::-1]", "aidepolcycne"),
        ("sum([int(c) for c in '987'])" if False else "sum([9, 8, 7])", 24),
        ("17 ** 5", 1419857),
        ("round(3847 * 0.15, 2)", 577.05),
        ("len('nevertheless') - 'nevertheless'.count('e')", 8),
    ],
)
def test_safe_eval_allows_whitelisted_expressions(expr: str, value: Any) -> None:
    assert safe_eval(expr) == value


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "(1).__class__",
        "{1: 2}",
        "2 ** 100000",
        "'a' * 10**9",
        "lambda: 1",
        "1 +",
    ],
)
def test_safe_eval_rejects_everything_else(expr: str) -> None:
    with pytest.raises(UnsafeExpression):
        safe_eval(expr)


def test_answer_contains() -> None:
    assert answer_contains("The result is 452,715,692.", 452715692)
    assert answer_contains("There are three r's.", 3)
    assert not answer_contains("There are two r's in it.", 3)
    assert answer_contains("Backwards: AIDEPOLCYCNE", "aidepolcycne")
    assert answer_contains("about 577.05 rupees", 577.05)
    # seen live: gpt-oss-120b writes LaTeX digit groups
    assert answer_contains("\\(17^5 = 1{,}419{,}857\\).", 1419857)
    assert answer_contains("1\u202f419\u202f857", 1419857)
    assert answer_contains("1\\,419\\,857", 1419857)
    assert not answer_contains("14198576", 1419857)  # qwen2.5:3b's wrong answer


# --- judge-based checks --------------------------------------------------------------

CFG = Config()


def rec(prompt: str, answer: str, **meta: Any) -> CallRecord:
    return CallRecord(
        messages=[Message(role="user", content=prompt)],
        response_text=answer,
        finish_reason="stop",
        metadata=meta,
    )


def test_hallucination_check() -> None:
    record = rec("What is the Nimbus return window?", "Their standard 30-day window.")
    [f] = HallucinationCheck(
        FixedJudge("contains_unsupported", 0.8, claims=["30-day window"])
    ).check(record, CFG)
    assert f.failure_type is FailureType.HALLUCINATION and f.confidence == pytest.approx(0.68)
    assert "30-day window" in f.evidence[0].detail
    assert f.prescription is not None and f.prescription.kind.value == "add_context"
    grounded = rec("q", "a", context="the policy text")
    [g] = HallucinationCheck(FixedJudge("contains_false", 0.9)).check(grounded, CFG)
    assert g.prescription is not None and g.prescription.kind.value == "prompt_change"
    assert HallucinationCheck(FixedJudge("supported")).check(record, CFG) == []


def test_context_reaches_the_judge() -> None:
    judge = FixedJudge("supported")
    HallucinationCheck(judge).check(rec("q", "a", context="CTX", reference_answer="REF"), CFG)
    assert judge.asked[0].material["context"] == "CTX"
    assert judge.asked[0].material["reference_answer"] == "REF"


def test_false_premise_check() -> None:
    record = rec("Why did Marie Curie win three Nobel Prizes?", "She won three because...")
    [f] = FalsePremiseCheck(FixedJudge("premise_accepted", premise="she won three")).check(
        record, CFG
    )
    assert f.failure_type is FailureType.FALSE_PREMISE and "she won three" in f.evidence[0].detail
    assert FalsePremiseCheck(FixedJudge("premise_corrected")).check(record, CFG) == []


def test_dropped_instruction_check_and_skip() -> None:
    record = rec("Write exactly 3 sentences, all lowercase. Mention the price.", "One. Two.")
    unmet = [
        "exactly 3 sentences",  # countable: code checks it, so the judge's word is dropped
        {"instruction": "Mention the price", "evidence": "missing"},
    ]
    [f] = DroppedInstructionCheck(FixedJudge("some_unmet", unmet=unmet)).check(record, CFG)
    assert "Mention the price (answer: missing)" in f.evidence[0].detail
    assert "sentences" not in f.evidence[0].detail
    # a complaint only about countable instructions, or with nothing quoted, is no finding
    only_counts = FixedJudge("some_unmet", unmet=["all lowercase"])
    assert DroppedInstructionCheck(only_counts).check(record, CFG) == []
    assert DroppedInstructionCheck(FixedJudge("some_unmet", unmet=[])).check(record, CFG) == []
    from counterfix import SkipCheck

    with pytest.raises(SkipCheck):
        DroppedInstructionCheck(FixedJudge("all_met")).check(rec("Tell me a joke", "Ha"), CFG)


def test_ambiguity_check() -> None:
    record = rec("What is the capital of Georgia?", "Atlanta.")
    [f] = AmbiguityCheck(FixedJudge("ambiguous_silent", readings=["country", "US state"])).check(
        record, CFG
    )
    assert f.prescription is not None and f.prescription.kind.value == "ask_user"
    assert AmbiguityCheck(FixedJudge("ambiguous_acknowledged")).check(record, CFG) == []


def test_over_refusal_check_only_asks_on_refusals() -> None:
    from counterfix import SkipCheck

    judge = FixedJudge("over_refusal")
    with pytest.raises(SkipCheck):
        OverRefusalCheck(judge).check(rec("How do I kill a process?", "Use kill -9."), CFG)
    assert judge.asked == []
    [f] = OverRefusalCheck(judge).check(
        rec("Write a villain monologue", "I can't help with that."), CFG
    )
    assert f.failure_type is FailureType.OVER_REFUSAL
    corrected = FixedJudge("premise_correction")
    assert (
        OverRefusalCheck(corrected).check(rec("q", "I cannot provide that because..."), CFG) == []
    )


def test_computation_check() -> None:
    wrong = rec("How many r's in strawberry?", "There are two r's.")
    judge = FixedJudge("computable", expression="'strawberry'.count('r')")
    [f] = ComputationCheck(judge).check(wrong, CFG)
    assert f.failure_type is FailureType.COMPUTATION and "3" in f.evidence[0].detail
    right = rec("How many r's in strawberry?", "There are 3 r's.")
    assert ComputationCheck(judge).check(right, CFG) == []
    assert ComputationCheck(FixedJudge("not_computable")).check(wrong, CFG) == []
    from counterfix import SkipCheck

    with pytest.raises(SkipCheck):
        ComputationCheck(FixedJudge("computable", expression="open('x')")).check(wrong, CFG)


def test_judge_failure_becomes_skip() -> None:
    from counterfix import run_detectors

    d = run_detectors(rec("q?", "a"), judge_detectors(FixedJudge("error", 0)), CFG)
    assert d.findings == [] and "judge_hallucination" in d.skipped


def test_computation_check_skips_requests_without_numbers() -> None:
    from counterfix import SkipCheck

    judge = FixedJudge("computable", expression="1+1")
    with pytest.raises(SkipCheck):
        ComputationCheck(judge).check(rec("Write a poem about rain.", "Drops fall."), CFG)
    assert judge.asked == []


def test_cache_offline_mode_never_calls_the_judge(tmp_path: Path) -> None:
    inner = FixedJudge("yes")
    cached = CachedJudge(inner, tmp_path / "c.jsonl")
    cached.ask(TASK)
    view = cached.cache_only()
    other = JudgeTask(name="t2", instructions="x", material={}, labels=("yes", "no"))
    assert view.ask(TASK).label == "yes"
    assert view.ask(other).failed and len(inner.asked) == 1
    assert not cached.offline  # the original still calls the judge


def test_safe_eval_comprehensions() -> None:
    from counterfix.safe_eval import UnsafeExpression, safe_eval

    assert safe_eval("sum(int(c) for c in str(987654321987))") == 69
    assert safe_eval("len([c for c in 'strawberry' if c == 'r'])") == 3
    assert safe_eval("''.join(reversed('encyclopedia'))") == "aidepolcycne"
    assert safe_eval("sum(x * x for x in range(1, 4))") == 14
    assert safe_eval("1 < 2 <= 2 and 'a' in 'cat'") is True
    for bad in (
        "[x for x in range(10**9)]",
        "[(x, y) for x in 'ab' for y in 'cd']",
        "__import__('os')",
        "(lambda: 1)()",
        "x",
    ):
        with pytest.raises(UnsafeExpression):
            safe_eval(bad)


def test_llm_judge_merges_system_prompt_when_refused_and_passes_params() -> None:
    seen: list[list[str]] = []

    class NoSystem(ScriptCaller):
        def complete(self, messages: Sequence[Any], **params: Any) -> CallRecord:
            roles = [m["role"] for m in messages]
            seen.append(roles)
            if "system" in roles:
                raise RuntimeError("400: Developer instruction is not enabled for this model")
            return super().complete(messages, **params)

    caller = NoSystem(reply("yes"))
    judge = LLMJudge(caller, params={"reasoning_effort": "low"})
    assert judge.ask(TASK).label == "yes" and judge.system_role is False
    assert seen[-1] == ["user"] and caller.calls[-1]["reasoning_effort"] == "low"


def test_retry_after_reads_headers_and_groq_messages() -> None:
    from counterfix.core.caller import retry_after

    class Limited(Exception):
        def __init__(self, msg: str, headers: dict[str, str] | None = None):
            super().__init__(msg)
            self.response = type("R", (), {"headers": headers or {}})()

    assert retry_after(Limited("429", {"retry-after": "12"})) == 12.0
    assert retry_after(Limited("Rate limit reached. Please try again in 7.5s.")) == 7.5
    assert retry_after(Limited("Please try again in 1m3.2s")) == 63.2
    assert retry_after(Limited("Please try again in 450ms")) == 0.45
    assert retry_after(Limited("429 too many requests")) is None


def test_llm_judge_drops_json_mode_on_a_bare_server_error() -> None:
    def respond(params: dict[str, Any]) -> str:
        if "response_format" in params:
            raise RuntimeError("Error code: 500 - Internal error encountered.")
        return reply("yes")

    judge = LLMJudge(ScriptCaller(respond))
    assert judge.ask(TASK).label == "yes" and judge.json_mode is False


def test_computation_yes_no_results() -> None:
    def check(answer: str, expression: str) -> list[Any]:
        record = rec("Answer yes or no: is 0.1 + 0.2 equal to 0.3?", answer)
        judge = FixedJudge("computable", expression=expression)
        return ComputationCheck(judge).check(record, CFG)

    assert check("No.", "0.1 + 0.2 == 0.3") == []
    assert len(check("Yes, exactly.", "0.1 + 0.2 == 0.3")) == 1
    from counterfix import SkipCheck

    with pytest.raises(SkipCheck):
        check("9.9 is larger than 9.11.", "9.11 > 9.9")


def test_rate_limited_caller_retries_a_passing_server_error() -> None:
    from counterfix.core.caller import RateLimitedCaller

    class InternalServerError(Exception):
        pass

    calls = {"n": 0}

    def respond(params: dict[str, Any]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise InternalServerError("Error code: 500")
        return reply("yes")

    limited = RateLimitedCaller(ScriptCaller(respond), 0)
    import counterfix.core.caller as caller_module

    real_sleep = caller_module.time.sleep
    caller_module.time.sleep = lambda s: None  # type: ignore[assignment]
    try:
        assert limited.complete([]).response_text
    finally:
        caller_module.time.sleep = real_sleep  # type: ignore[assignment]
    assert calls["n"] == 2
