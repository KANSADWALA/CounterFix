"""Standard benchmarks on any model counterfix can call, with the misses explained.

>>> from counterfix.evals.benchmarks import run_benchmark
>>> r = run_benchmark("gsm8k", model="ollama:qwen2.5:3b", limit=100)   # doctest: +SKIP
>>> print(r.text())                                                    # doctest: +SKIP

Each benchmark is a chat-style (generative) run: the model writes an answer and counterfix
reads the letter, number or code from it, scores it, groups the misses by cause and, with
``fix=N``, verifies fixes. Subsets are stratified and seeded (``per_group=20`` takes 20
questions per MMLU subject), so a run fits a free tier and is repeatable; the report's
confidence interval says how far a subset can be from the full set.

Datasets are downloaded when you run them (Hugging Face ``datasets``, or EvalPlus for
code), never shipped with counterfix. Licences: see :data:`BENCHMARKS`. Prompts are
versioned (``template``): a changed prompt gets a new version, so results stay comparable.

Log-likelihood scoring (the classic MMLU setup) needs token probabilities that chat APIs
do not give; for local models use lm-evaluation-harness, which counterfix does not wrap.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from counterfix._internal.suggest import did_you_mean
from counterfix.evals.metrics.base import Metric
from counterfix.evals.metrics.extract import ChoiceMatch, NumberMatch

_LETTERS = "ABCDEFGHIJKLMNOP"


@dataclass(frozen=True)
class Benchmark:
    """One benchmark: where its data comes from, how it is asked and scored."""

    name: str
    title: str
    source: str
    licence: str
    template: str
    """Prompt version: changes whenever the prompt or the answer reading changes."""
    load: Callable[[int], list[dict[str, Any]]]
    """seed -> rows (id, prompt, reference, group), the full test set."""
    metrics: Callable[[], list[Metric]]
    params: dict[str, Any] = field(default_factory=dict)
    extra: str = "benchmarks"
    runs_code: bool = False


def _datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            'benchmarks need Hugging Face datasets: pip install "counterfix[benchmarks]"'
        ) from exc
    return datasets


def _options(choices: Sequence[str], letters: str = _LETTERS) -> str:
    return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))


_MC_TAIL = "Answer with the letter of the correct option on the last line, as 'Answer: X'."


# --- loaders ------------------------------------------------------------------------------


def _mmlu(seed: int) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for n, r in enumerate(ds):
        subject = r["subject"].replace("_", " ")
        prompt = (
            f"The following is a multiple choice question about {subject}.\n\n"
            f"{r['question'].strip()}\n{_options(r['choices'])}\n\n{_MC_TAIL}"
        )
        rows.append({"id": f"mmlu-{n}", "prompt": prompt, "reference": _LETTERS[r["answer"]],
                     "group": r["subject"]})  # fmt: skip
    return rows


def _gsm8k(seed: int) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for n, r in enumerate(ds):
        answer = r["answer"].split("####")[-1].strip().replace(",", "")
        prompt = (
            f"{r['question'].strip()}\n\nSolve it step by step, then give the final answer on "
            "the last line as 'Answer: <number>'."
        )
        rows.append({"id": f"gsm8k-{n}", "prompt": prompt, "reference": answer, "group": "all"})
    return rows


def _arc(seed: int) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for r in ds:
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        key = str(r["answerKey"]).strip()
        if key not in labels:
            continue
        answer = _LETTERS[labels.index(key)]  # "1".."4" labels become A..D
        prompt = f"{r['question'].strip()}\n{_options(texts)}\n\n{_MC_TAIL}"
        rows.append({"id": f"arc-{r['id']}", "prompt": prompt, "reference": answer,
                     "group": "all", "n_options": len(texts)})  # fmt: skip
    return rows


def _truthfulqa(seed: int) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for n, r in enumerate(ds):
        choices = list(r["mc1_targets"]["choices"])
        right = choices[list(r["mc1_targets"]["labels"]).index(1)]
        order = list(range(len(choices)))
        random.Random(f"{seed}-{n}").shuffle(order)  # the data lists the right answer first
        shuffled = [choices[i] for i in order]
        prompt = f"{r['question'].strip()}\n{_options(shuffled)}\n\n{_MC_TAIL}"
        rows.append({"id": f"truthfulqa-{n}", "prompt": prompt,
                     "reference": _LETTERS[shuffled.index(right)], "group": "all",
                     "n_options": len(choices)})  # fmt: skip
    return rows


def _ifeval(seed: int) -> list[dict[str, Any]]:
    ds = _datasets().load_dataset("google/IFEval", split="train")
    return [
        {
            "id": f"ifeval-{r['key']}",
            "prompt": r["prompt"],
            "reference": {
                "key": r["key"],
                "prompt": r["prompt"],
                "instruction_id_list": list(r["instruction_id_list"]),
                "kwargs": [dict(k) for k in r["kwargs"]],
            },
            "group": "all",
        }
        for r in ds
    ]


def _code_rows(dataset: str, plus: bool = True) -> list[dict[str, Any]]:
    """EvalPlus's problems; ``plus=False`` points the reference at the original tests,
    scored by counterfix's own checker (works on Windows, no EvalPlus scorer)."""
    from counterfix.evals.benchmarks.evalplus import base_tests, problems

    rows = []
    for task_id, p in problems(dataset).items():
        if dataset == "humaneval":
            prompt = (
                "Complete the following Python function. Reply with the whole function, "
                f"with its imports, in one ```python code block.\n\n```python\n{p['prompt']}```"
            )
        else:
            text = p["prompt"].strip().strip('"').strip()
            prompt = (
                f"{text}\n\nWrite the Python function `{p['entry_point']}`. Reply with the "
                "code, with its imports, in one ```python code block."
            )
        reference = {"task_id": task_id, "dataset": dataset} if plus else base_tests(p)
        rows.append(
            {
                "id": task_id.replace("/", "-"),
                "prompt": prompt,
                "reference": reference,
                "group": "all",
            }
        )
    return rows


def _evalplus_metric() -> list[Metric]:
    from counterfix.evals.benchmarks.evalplus import EvalPlusTests

    return [EvalPlusTests()]


def _tests_metric() -> list[Metric]:
    from counterfix.evals.metrics.code import TestsPass

    return [TestsPass()]


def _injection_metric() -> list[Metric]:
    from counterfix.evals.benchmarks.injection import InjectionResisted

    return [InjectionResisted()]


def _injection_rows(seed: int) -> list[dict[str, Any]]:
    from counterfix.evals.benchmarks.injection import probes

    return probes(seed)


def _tool_metric() -> list[Metric]:
    from counterfix.evals.metrics.tools import ToolCallMatch

    return [ToolCallMatch()]


def _tool_rows(seed: int) -> list[dict[str, Any]]:
    from counterfix.evals.benchmarks.tool_use import probes

    return probes(seed)


def _ifeval_metric() -> list[Metric]:
    from counterfix.evals.benchmarks.ifeval import IfEvalStrict

    return [IfEvalStrict()]


BENCHMARKS: dict[str, Benchmark] = {
    b.name: b
    for b in (
        Benchmark(
            "mmlu",
            "MMLU (57 subjects, multiple choice, generative)",
            "cais/mmlu (all, test)",
            "MIT",
            "v1",
            _mmlu,
            lambda: [ChoiceMatch()],
            {"temperature": 0, "max_tokens": 4096},
        ),
        Benchmark(
            "gsm8k",
            "GSM8K (grade-school maths)",
            "openai/gsm8k (main, test)",
            "MIT",
            "v1",
            _gsm8k,
            lambda: [NumberMatch()],
            {"temperature": 0, "max_tokens": 4096},
        ),
        Benchmark(
            "arc",
            "ARC-Challenge (science questions)",
            "allenai/ai2_arc (ARC-Challenge, test)",
            "CC-BY-SA-4.0",
            "v1",
            _arc,
            lambda: [ChoiceMatch(letters="ABCDE")],
            {"temperature": 0, "max_tokens": 4096},
        ),
        Benchmark(
            "truthfulqa",
            "TruthfulQA MC1 (one true answer, by generation)",
            "truthfulqa/truthful_qa (multiple_choice, validation)",
            "Apache-2.0",
            "v1",
            _truthfulqa,
            lambda: [ChoiceMatch(letters=_LETTERS)],
            {"temperature": 0, "max_tokens": 4096},
        ),
        Benchmark(
            "ifeval",
            "IFEval (verifiable instructions, strict prompt-level)",
            "google/IFEval (train)",
            "Apache-2.0",
            "v1",
            _ifeval,
            _ifeval_metric,
            {"temperature": 0, "max_tokens": 4096},
        ),
        Benchmark(
            "humaneval+",
            "HumanEval+ (EvalPlus tests)",
            "EvalPlus HumanEval+",
            "Apache-2.0 (EvalPlus); HumanEval: MIT",
            "v1",
            lambda seed: _code_rows("humaneval"),
            _evalplus_metric,
            {"temperature": 0, "max_tokens": 4096},
            extra="code",
            runs_code=True,
        ),
        Benchmark(
            "injection",
            "Prompt injection (20 probes: planted instructions in passages and tool results)",
            "written for counterfix",
            "MIT (counterfix)",
            "v1",
            _injection_rows,
            _injection_metric,
            {"temperature": 0, "max_tokens": 4096},
            extra="core",
        ),
        Benchmark(
            "tools",
            "Tool use (20 probes: right tool, right arguments, no tool when none is needed)",
            "written for counterfix (BFCL-style simple and irrelevance cases)",
            "MIT (counterfix)",
            "v1",
            _tool_rows,
            _tool_metric,
            {"temperature": 0, "max_tokens": 4096},
            extra="core",
        ),
        Benchmark(
            "humaneval",
            "HumanEval (original tests, counterfix's checker; runs on Windows)",
            "EvalPlus HumanEval+ problems, original HumanEval tests",
            "MIT (HumanEval); EvalPlus data: Apache-2.0",
            "v1",
            lambda seed: _code_rows("humaneval", plus=False),
            _tests_metric,
            {"temperature": 0, "max_tokens": 4096},
            extra="code",
            runs_code=True,
        ),
        Benchmark(
            "mbpp",
            "MBPP (original asserts, counterfix's checker; runs on Windows)",
            "EvalPlus MBPP+ problems, original MBPP asserts",
            "CC-BY-4.0 (MBPP); EvalPlus data: Apache-2.0",
            "v1",
            lambda seed: _code_rows("mbpp", plus=False),
            _tests_metric,
            {"temperature": 0, "max_tokens": 4096},
            extra="code",
            runs_code=True,
        ),
        Benchmark(
            "mbpp+",
            "MBPP+ (EvalPlus tests)",
            "EvalPlus MBPP+",
            "Apache-2.0 (EvalPlus); MBPP: CC-BY-4.0",
            "v1",
            lambda seed: _code_rows("mbpp"),
            _evalplus_metric,
            {"temperature": 0, "max_tokens": 4096},
            extra="code",
            runs_code=True,
        ),
    )
}


def sample(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
    per_group: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """A repeatable subset: ``per_group`` questions from each group (e.g. MMLU subject)
    and/or ``limit`` in all, drawn with ``seed``; the rows keep the dataset's order."""
    rng = random.Random(seed)
    index = list(range(len(rows)))
    if per_group is not None:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in index:
            groups[str(rows[i].get("group", "all"))].append(i)
        index = []
        for members in groups.values():
            index += rng.sample(members, min(per_group, len(members)))
    if limit is not None and limit < len(index):
        index = rng.sample(index, limit)
    return [rows[i] for i in sorted(index)]


def load_benchmark(
    name: str, *, limit: int | None = None, per_group: int | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    """The benchmark's rows (``id``, ``prompt``, ``reference``, ``group``), subset if asked."""
    bench = get_benchmark(name)
    return sample(bench.load(seed), limit=limit, per_group=per_group, seed=seed)


def get_benchmark(name: str) -> Benchmark:
    key = name.lower().replace("_", "-")
    key = {"arc-challenge": "arc"}.get(key, key)
    if key not in BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {name!r}{did_you_mean(name, BENCHMARKS)}; "
            f"choose from {', '.join(BENCHMARKS)}"
        )
    return BENCHMARKS[key]


def run_benchmark(
    name: str,
    *,
    model: Any = None,
    app: Any = None,
    limit: int | None = None,
    per_group: int | None = None,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    **evaluate_options: Any,
) -> Any:
    """Run a benchmark through :func:`counterfix.evaluate` and return its ``EvalResult``,
    with ``info`` holding the benchmark, template version, seed, subset and licence, and
    per-group accuracy (e.g. per MMLU subject). Other options (``judge``, ``fix``,
    ``concurrency``, ``samples``, ``out``, ``progress``) go to ``evaluate``."""
    from counterfix.evals import evaluate

    bench = get_benchmark(name)
    rows = load_benchmark(bench.name, limit=limit, per_group=per_group, seed=seed)
    result = evaluate(
        rows,
        model=model,
        app=app,
        metrics=bench.metrics(),
        input="prompt",
        reference="reference",
        params={**bench.params, **(params or {})},
        name=bench.name,
        **evaluate_options,
    )
    group_of = {str(r["id"]): str(r.get("group", "all")) for r in rows}
    totals: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for item in result.items:
        group = group_of.get(item.id.rsplit("#", 1)[0], "all")
        totals[group] += 1
        passed[group] += int(item.passed)
    result.info = {
        "benchmark": bench.title,
        "source": bench.source,
        "licence": bench.licence,
        "template": bench.template,
        "seed": seed,
        "subset": (f"{per_group} per group" if per_group else "")
        + (f"{', ' if per_group else ''}{limit} in all" if limit else "")
        or "full",
        "by_group": {g: f"{passed[g]}/{totals[g]}" for g in sorted(totals)}
        if len(totals) > 1
        else {},
    }
    return result


__all__ = ["BENCHMARKS", "Benchmark", "get_benchmark", "load_benchmark", "run_benchmark", "sample"]
