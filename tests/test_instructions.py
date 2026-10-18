"""Exact instruction checks (F6 in code) and the one-mistake-one-cause rule."""

from __future__ import annotations

import pytest

from counterfix import CallRecord, Config, FailureType, Finding, Message, SkipCheck, run_detectors
from counterfix.core.precedence import resolve
from counterfix.detectors.instructions import (
    InstructionCheck,
    extract_constraints,
    list_items,
    mentions,
    sentences,
    words,
)


def rec(prompt: str, answer: str, finish: str | None = "stop") -> CallRecord:
    return CallRecord(
        model="m",
        messages=[Message(role="user", content=prompt)],
        response_text=answer,
        finish_reason=finish,
    )


def broken(prompt: str, answer: str) -> list[str]:
    """Descriptions of the instructions the answer breaks."""
    return [c.description for c in extract_constraints(prompt) if c.violation(answer)]


def test_counting_helpers() -> None:
    text = "# Title\n\nFirst one. Second one! Third?\n- item a\n- item b\n  - nested"
    assert sentences(text) == [
        "First one.",
        "Second one!",
        "Third?",
        "item a",
        "item b",
        "nested",
    ]
    assert list_items(text) == ["item a", "item b"]
    assert len(words("Hello, **world** - it's fine")) == 4
    assert len(words("API एक जादू की दरवाज़ा है।")) == 6  # Devanagari with vowel signs
    assert sentences("e.g. this is one sentence. And 3.5 stays.") == [
        "e.g. this is one sentence.",
        "And 3.5 stays.",
    ]


@pytest.mark.parametrize(
    ("prompt", "good", "bad"),
    [
        ("Write exactly 3 sentences.", "A. B. C.", "A. B."),
        ("Write two sentences about logos.", "One here. Two here.", "Sure! One. Two."),
        ("Explain in 2-3 sentences.", "A. B.", "A."),
        ("Answer in under 5 words.", "Yes it is.", "Yes it is a fine idea."),
        ("Summarise in 4 words or fewer.", "Short and sweet.", "This is far too long here."),
        ("Give 3 bullet points.", "- a\n- b\n- c", "- a\n- b"),
        ("List 2 tips. Number them.", "1. a\n2. b", "- a\n- b"),
        ("Do not use the word 'rain'.", "clouds burst", "the rains came"),
        ("Do not start any tip with 'Be'.", "1. Arrive early", "1. Be early"),
        ("End with a question.", "Ready? Shall we?", "Ready? Let us go."),
        ("Use only lowercase letters.", "all small here", "All small here"),
        ("Reply in Hindi.", "नमस्ते, आप कैसे हैं?", "Hello, how are you?"),
        ("Use British spelling.", "a bright colour", "a bright color"),
        ("Use American spelling.", "we organize it", "we organise it"),
        ("Add a title in square brackets.", "[rain song]\nline", "rain song\nline"),
        ("What is 2*3? Give only the number.", "**6**", "The answer is 6."),
        ("Is a tomato a fruit? One word.", "Fruit.", "It is a fruit."),
        ("Is it spam? Yes or no.", "**No.** It is a real notice.", "It is not spam."),
        ("Write it in exactly two paragraphs.", "One.\n\nTwo.", "One.\n\nTwo.\n\nThree."),
        ("Each tip must be under 6 words.", "- go early\n- smile", "- go early to the venue ok"),
    ],
)
def test_constraints_pass_and_fail(prompt: str, good: str, bad: str) -> None:
    assert extract_constraints(prompt), prompt
    assert broken(prompt, good) == [], (prompt, good)
    assert broken(prompt, bad), (prompt, bad)


@pytest.mark.parametrize(
    "prompt",
    [
        "Write a detailed 800-word essay on the French Revolution.",
        "Summarise each of the seven books in one paragraph per book.",
        'Answer with a JSON object {"answer": <yes or no>, "reason": <one sentence>}.',
        "Never mention competitor brands.",  # content: left to the judge
        "What does 'bank' mean in Hindi?",
        "Give me 3 examples.",  # prose answers are fine without a list
    ],
)
def test_no_false_constraints(prompt: str) -> None:
    constraints = extract_constraints(prompt)
    if prompt.startswith("Give me 3"):
        assert broken(prompt, "First, x. Second, y. Third, z.") == []
    else:
        assert constraints == [], [c.description for c in constraints]


def test_instruction_check_finding_and_skips() -> None:
    prompt = "Reply in Hindi in under 10 words and end with a question."
    [f] = InstructionCheck().check(
        rec(prompt, "This is an English reply without a question."), Config()
    )
    assert f.failure_type is FailureType.DROPPED_INSTRUCTION and f.confidence == 0.9
    assert "written in Hindi" in f.evidence[0].detail
    assert "ends with a question" in f.evidence[0].detail
    assert InstructionCheck().check(rec(prompt, "क्या तुम तैयार हो?"), Config()) == []
    with pytest.raises(SkipCheck):
        InstructionCheck().check(rec("Tell me a joke", "Ha"), Config())
    with pytest.raises(SkipCheck):  # a cut-off answer is F1, not F6
        InstructionCheck().check(rec("Write 3 sentences.", "One. Two", finish="length"), Config())


def test_mentions_maps_judge_complaints_to_code_checks() -> None:
    [count, banned] = extract_constraints("Write exactly 3 sentences. Do not use the word 'rain'.")
    assert mentions(count, "The answer has 4 sentences, not 3")
    assert mentions(banned, "uses rain")
    assert not mentions(count, "does not mention the price")


def finding(kind: FailureType, confidence: float = 0.9, detail: str = "x") -> Finding:
    from counterfix import Evidence

    return Finding(
        failure_type=kind, confidence=confidence, evidence=[Evidence(detector="t", detail=detail)]
    )


def test_root_cause_wins_and_duplicates_merge() -> None:
    F = FailureType
    kept, suppressed = resolve(
        [
            finding(F.COMPUTATION),
            finding(F.HALLUCINATION),
            finding(F.DROPPED_INSTRUCTION, 0.9, "code"),
            finding(F.DROPPED_INSTRUCTION, 0.6, "judge"),
        ]
    )
    assert [f.failure_type for f in kept] == [F.COMPUTATION, F.DROPPED_INSTRUCTION]
    assert [e.detail for e in kept[1].evidence] == ["code", "judge"]
    assert suppressed[0].failure_type is F.HALLUCINATION
    assert "F8" in suppressed[0].evidence[-1].detail
    # a weak signal (e.g. a premise correction scored 0.3 as a refusal) suppresses nothing
    kept, suppressed = resolve([finding(F.OVER_REFUSAL, 0.3), finding(F.HALLUCINATION)])
    assert len(kept) == 2 and suppressed == []


def test_runner_applies_precedence_and_keeps_record() -> None:
    class Two:
        name = "two"
        covers = (FailureType.TRUNCATION, FailureType.DROPPED_INSTRUCTION)

        def check(self, record: CallRecord, config: Config) -> list[Finding]:
            return [finding(FailureType.TRUNCATION), finding(FailureType.DROPPED_INSTRUCTION)]

    record = rec("hi", "hello")
    d = run_detectors(record, [Two()], Config())
    assert [f.failure_type for f in d.findings] == [FailureType.TRUNCATION]
    assert [f.failure_type for f in d.suppressed] == [FailureType.DROPPED_INSTRUCTION]
    assert d.record is record and "record" not in d.model_dump()
