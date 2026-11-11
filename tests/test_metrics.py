"""Reference metrics (E2): each matches its library or a hand-computed value."""

from __future__ import annotations

import pytest

from counterfix import CallRecord, FailureType, Timing
from counterfix.metrics import (
    METRICS,
    extract_choice,
    extract_code,
    extract_number,
    get_metric,
    normalize,
    rouge_l,
    token_f1,
)


def answer(text: str, **kw: object) -> CallRecord:
    return CallRecord(response_text=text, finish_reason="stop", **kw)  # type: ignore[arg-type]


def test_normalize_follows_squad() -> None:
    assert normalize("The  Eiffel-Tower!") == "eiffeltower"
    assert normalize("an Apple, a day") == "apple day"


@pytest.mark.parametrize(
    ("pred", "ref", "expected"),
    [
        ("the cat sat", "a cat sat down", 2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3)),
        ("Paris", "paris", 1.0),
        ("London", "Paris", 0.0),
        ("", "", 1.0),
        ("", "x", 0.0),
    ],
)
def test_token_f1_hand_computed(pred: str, ref: str, expected: float) -> None:
    assert token_f1(pred, ref) == pytest.approx(expected)


def test_exact_match_with_several_references_and_think_tags() -> None:
    m = get_metric("exact_match")
    r = m.score(answer("<think>hmm</think> The Eiffel Tower."), ["Eiffel Tower", "x"])
    assert r.value == 1.0 and r.passed is True
    miss = m.score(answer("Big Ben"), "Eiffel Tower")
    assert miss.value == 0.0 and miss.passed is False


def test_contains_needs_whole_words() -> None:
    m = get_metric("contains")
    assert m.score(answer("It is in Paris, France."), "paris").passed is True
    assert m.score(answer("Parisian food"), "paris").passed is False


def test_graded_metrics_report_a_number_without_failing_unless_given_a_threshold() -> None:
    r = get_metric("f1").score(answer("the cat sat"), "a cat sat down")
    assert r.passed is None and r.value == pytest.approx(0.8)
    r = get_metric("f1:0.9").score(answer("the cat sat"), "a cat sat down")
    assert r.passed is False and r.threshold == 0.9


def test_missing_reference_and_failed_calls() -> None:
    m = get_metric("exact_match")
    none = m.score(answer("x"))
    assert none.value is None and "no reference" in none.reason
    failed = m.score(CallRecord(error="APIConnectionError: down"), "x")
    assert failed.passed is False and "failed" in failed.reason


def test_rouge_l_matches_rouge_score_package() -> None:
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    pairs = [
        ("The quick brown fox jumps over the lazy dog", "A quick brown dog jumps on the lazy fox"),
        ("police killed the gunman", "the gunman was shot down by police"),
        ("Hello, World! 123", "hello world 1234"),
        ("nothing in common", "zzz yyy"),
    ]
    for pred, ref in pairs:
        expected = scorer.score(ref, pred)["rougeL"].fmeasure
        assert rouge_l(pred, ref) == pytest.approx(expected, abs=1e-6)


def test_bleu_and_chrf_match_sacrebleu() -> None:
    sacrebleu = pytest.importorskip("sacrebleu")
    pred, refs = "The cat is on the mat.", ["There is a cat on the mat.", "A cat is on the mat."]
    rec = answer(pred)
    assert get_metric("bleu").score(rec, refs).value == pytest.approx(
        sacrebleu.sentence_bleu(pred, refs).score / 100, abs=1e-6
    )
    assert get_metric("chrf").score(rec, refs).value == pytest.approx(
        sacrebleu.sentence_chrf(pred, refs).score / 100, abs=1e-6
    )


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("The answer is (B).", "B"),
        ("C", "C"),
        ("**D**", "D"),
        ("I first thought A. Final answer: C", "C"),
        ("(A) apples\n(B) pears\nAnswer: B", "B"),
        ("The answer is a cat", None),
        ("\\boxed{A}", "A"),
        ("Nothing chosen.", None),
    ],
)
def test_extract_choice(text: str, letter: str | None) -> None:
    assert extract_choice(text) == letter


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("She earns 9 * 2 = $18.", 18.0),
        ("reasoning... #### 1,234", 1234.0),
        ("The answer is 42 apples, not 7.", 42.0),
        ("so \\boxed{3.5}", 3.5),
        ("It drops to -7 degrees", -7.0),
        ("no digits", None),
    ],
)
def test_extract_number(text: str, number: float | None) -> None:
    assert extract_number(text) == number


def test_choice_and_number_metrics() -> None:
    assert (
        get_metric("choice").score(answer("Looking at the options, (B) is right."), "B").passed
        is True
    )
    assert get_metric("choice").score(answer("I am not sure"), "A").value == 0.0
    gsm_ref = "Natalia sold 48/2 = 24 clips in May.\n#### 72"
    assert get_metric("number").score(answer("In total 72 clips."), gsm_ref).passed is True
    assert get_metric("number").score(answer("In total 70 clips."), gsm_ref).passed is False


def test_extract_code_prefers_the_last_python_block() -> None:
    text = "Here:\n```bash\npip install x\n```\nand\n```python\ndef f():\n    return 1\n```\n"
    assert extract_code(text) == "def f():\n    return 1"
    assert extract_code("def g(): pass") == "def g(): pass"


def test_json_metrics_point_to_f2() -> None:
    valid = get_metric("json_valid")
    assert valid.score(answer('```json\n{"a": 1}\n```')).passed is True
    bad = valid.score(answer('{"a": 1'))
    assert bad.passed is False and bad.failure_type is FailureType.FORMAT_BREAK
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    assert get_metric("json_valid", schema=schema).score(answer('{"age": 3}')).passed is False
    match = get_metric("json_match").score(
        answer('{"name": "Ann", "address": {"city": "Pune", "pin": 1}}'),
        {"name": "Ann", "address": {"city": "Pune", "pin": 2}},
    )
    assert match.value == pytest.approx(2 / 3) and "address.pin" in match.reason


def test_speed_metrics_are_lower_is_better() -> None:
    r = answer("x", latency_ms=1200, timing=Timing(ttft_ms=300, chunks=1))
    assert get_metric("latency:1000").score(r).passed is False
    assert get_metric("ttft:500").score(r).passed is True
    not_streamed = get_metric("ttft:500").score(answer("x"))
    assert not_streamed.passed is None and "not streamed" in not_streamed.reason  # not scored


def test_registry_names_and_errors() -> None:
    assert set(METRICS) >= {"exact_match", "f1", "rouge_l", "choice", "number", "latency"}
    with pytest.raises(ValueError, match="unknown metric"):
        get_metric("nope")


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("A. 4\nB. 5\nC. 6\nD. 7\n\nThe sum is 5, so the answer is B.", "B"),
        ("Options:\nA) Paris\nB) Rome\nC) Madrid\nD) Berlin\nParis is the capital.", None),
        ("Answer: A\n\nWait, re-checking: the answer is C.", "C"),
        ("It is a common mistake to pick A, but the right answer is D.", "D"),
        ("I'd go with B.", "B"),
        ("Both A and B seem plausible.", None),
    ],
)
def test_choice_extraction_does_not_read_a_listed_option_as_the_answer(
    text: str, letter: str | None
) -> None:
    assert extract_choice(text) == letter


def test_mmlu_integer_references_are_option_indexes() -> None:
    m = get_metric("choice")
    assert m.score(answer("Answer: C"), 2).passed is True
    assert m.score(answer("Answer: C"), 0).passed is False


def test_numbers_only_group_thousands_in_threes() -> None:
    assert extract_number("The total is 1,234,567 dollars.") == 1234567
    assert extract_number("It costs $1,200.50 in total.") == 1200.5
    assert extract_number("sides 3,4 and 5") == 5


def test_json_valid_never_mistakes_a_reference_for_a_schema() -> None:
    r = get_metric("json_valid").score(answer('{"type": "apple"}'), {"type": "fruit"})
    assert r.passed is True
