"""Benchmarks: loaders (on stand-in data), seeded subsets, scoring, and the official
scorers when their libraries are installed (IFEval via lm-eval, EvalPlus)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from counterfix.core.record import CallRecord
from counterfix.evals import benchmarks
from counterfix.evals.benchmarks import get_benchmark, load_benchmark, run_benchmark, sample

# Exercises a feature deprecated in 0.3 (removed in 0.4): its FutureWarning is expected.
DEPRECATED_IN_0_3 = pytest.mark.filterwarnings(
    "ignore:.*will be removed in counterfix:FutureWarning"
)

FAKE = {
    ("cais/mmlu", "all"): [
        {"question": f"Q{i}?", "subject": s, "choices": ["w", "x", "y", "z"], "answer": i % 4}
        for i, s in enumerate(["anatomy"] * 6 + ["astronomy"] * 6)
    ],
    ("openai/gsm8k", "main"): [
        {"question": "2 + 3?", "answer": "2 + 3 = 5\n#### 5"},
        {"question": "1,000 + 1?", "answer": "#### 1,001"},
    ],
    ("allenai/ai2_arc", "ARC-Challenge"): [
        {
            "id": "a1",
            "question": "Which?",
            "answerKey": "3",
            "choices": {"label": ["1", "2", "3", "4"], "text": ["p", "q", "r", "s"]},
        },
    ],
    ("truthfulqa/truthful_qa", "multiple_choice"): [
        {
            "question": "True?",
            "mc1_targets": {"choices": ["right", "w1", "w2"], "labels": [1, 0, 0]},
        },
    ],
}


@pytest.fixture(autouse=True)
def _fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("datasets")
    module.load_dataset = lambda name, config=None, split=None: FAKE[(name, config)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


@DEPRECATED_IN_0_3
def test_loaders_build_prompts_and_references() -> None:
    mmlu = load_benchmark("mmlu")
    assert len(mmlu) == 12 and mmlu[1]["reference"] == "B" and "anatomy" in mmlu[0]["prompt"]
    assert "B. x" in mmlu[0]["prompt"] and "Answer: X" in mmlu[0]["prompt"]
    assert [r["reference"] for r in load_benchmark("gsm8k")] == ["5", "1001"]
    assert load_benchmark("arc")[0]["reference"] == "C"  # label "3" is the third option
    tqa = load_benchmark("truthfulqa", seed=1)[0]
    letter = tqa["reference"]
    assert f"{letter}. right" in tqa["prompt"]  # shuffled, still pointing at the right one


@DEPRECATED_IN_0_3
def test_subsets_are_stratified_and_repeatable() -> None:
    rows = load_benchmark("mmlu")
    two = sample(rows, per_group=2, seed=3)
    assert sorted(r["group"] for r in two) == ["anatomy", "anatomy", "astronomy", "astronomy"]
    assert two == sample(rows, per_group=2, seed=3)
    assert len(sample(rows, limit=5, seed=0)) == 5
    assert get_benchmark("HumanEval").name == "humaneval"
    assert get_benchmark("humaneval+").runs_code
    with pytest.raises(ValueError):
        get_benchmark("nope")


class _Picker:
    name = "fake:picker"

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        text = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].text
        if "2 + 3" in text:
            return CallRecord(response_text="2 + 3 = 5\nAnswer: 5", params=params)
        if "Q1?" in text or "Q5?" in text:
            return CallRecord(response_text="Answer: B", params=params)
        return CallRecord(response_text="Answer: A", params=params)


@DEPRECATED_IN_0_3
def test_run_benchmark_scores_and_reports() -> None:
    result = run_benchmark("mmlu", model=_Picker(), per_group=6, seed=0)
    assert result.info["template"] == "v1" and result.info["licence"] == "MIT"
    assert set(result.info["by_group"]) == {"anatomy", "astronomy"}
    assert result.metric("choice").passed == 5  # A right for Q0, Q4, Q8; B for Q1, Q5
    assert result.info["by_group"] == {"anatomy": "4/6", "astronomy": "1/6"}
    text = result.text()
    assert "template v1" in text and "weakest groups" in text
    gsm = run_benchmark("gsm8k", model=_Picker())
    assert [i.passed for i in gsm.items] == [True, False]
    assert gsm.items[0].record.params.get("temperature") == 0


def test_benchmark_list_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from counterfix.cli import main

    assert main(["benchmark", "--list"]) == 0
    assert "humaneval+" in capsys.readouterr().out


def test_ifeval_official_checks() -> None:
    pytest.importorskip("lm_eval.tasks.ifeval.utils")
    from counterfix.evals.benchmarks.ifeval import score_instructions

    ref = {"key": 1, "prompt": "Answer in all lowercase letters.",
           "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}  # fmt: skip
    assert score_instructions("all lowercase here.", ref).passed is True
    miss = score_instructions("Not Lowercase.", ref)
    assert miss.passed is False and "change_case:english_lowercase" in miss.reason


def test_evalplus_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32":
        pytest.skip("EvalPlus's scorer is Unix-only (signals, resource limits)")
    pytest.importorskip("evalplus")
    monkeypatch.delitem(sys.modules, "datasets")  # EvalPlus has its own loader
    monkeypatch.setenv("COUNTERFIX_ALLOW_CODE", "1")
    monkeypatch.setenv("COUNTERFIX_SANDBOX", "subprocess")
    from counterfix.evals.benchmarks.evalplus import EvalPlusTests, problems

    p = problems("humaneval")["HumanEval/0"]
    m = EvalPlusTests()
    ref = {"task_id": "HumanEval/0", "dataset": "humaneval"}
    good = CallRecord(response_text=f"```python\n{p['prompt']}{p['canonical_solution']}\n```")
    assert m.score(good, ref).passed is True
    bad = CallRecord(response_text="    return False\n")
    r = m.score(bad, ref)
    assert r.passed is False and r.reason.startswith("[wrong_output]")


def test_benchmarks_module_lists_licences() -> None:
    for b in benchmarks.BENCHMARKS.values():
        assert b.licence and b.source and b.template
    assert SimpleNamespace  # keep import used
