"""counterfix's probe sets, run on any model it can call, with the misses explained.

>>> from counterfix.evals.benchmarks import run_benchmark
>>> r = run_benchmark("injection", model="ollama:qwen2.5:3b")          # doctest: +SKIP
>>> print(r.text())                                                    # doctest: +SKIP

``injection`` plants instructions in passages and tool results (F13); ``tools`` checks
tool choice and arguments (F14). Both are small, written for counterfix, and test failure
types it diagnoses and fixes; ``fix=N`` verifies fixes on the misses. For standard
benchmarks (MMLU, GSM8K, HumanEval ...), use lm-evaluation-harness or EvalPlus.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from counterfix._internal.suggest import did_you_mean
from counterfix.evals.metrics.base import Metric


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


BENCHMARKS: dict[str, Benchmark] = {
    b.name: b
    for b in (
        Benchmark(
            "injection",
            "Prompt injection (20 probes: planted instructions in passages and tool results)",
            "written for counterfix",
            "MIT (counterfix)",
            "v1",
            _injection_rows,
            _injection_metric,
            {"temperature": 0, "max_tokens": 4096},
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
    """A repeatable subset: ``per_group`` questions from each group
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
    per-group accuracy. Other options (``judge``, ``fix``,
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
