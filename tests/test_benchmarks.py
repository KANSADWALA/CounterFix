"""counterfix's probe sets (injection, tools): the registry, seeded subsets and the CLI.
Running them on a model is covered in test_injection.py and test_agents.py."""

from __future__ import annotations

import pytest

from counterfix.evals import benchmarks


def test_only_the_probe_sets_are_left() -> None:
    assert set(benchmarks.BENCHMARKS) == {"injection", "tools"}
    for b in benchmarks.BENCHMARKS.values():
        assert b.licence and b.source and b.template
        assert b.load(0) and b.metrics()


def test_standard_benchmarks_point_elsewhere() -> None:
    with pytest.raises(ValueError, match="unknown benchmark 'mmlu'"):
        benchmarks.get_benchmark("mmlu")


def test_subsets_are_stratified_and_repeatable() -> None:
    rows = [{"id": i, "group": "ab"[i % 2]} for i in range(20)]
    one = benchmarks.sample(rows, per_group=3, seed=1)
    assert [r["group"] for r in one].count("a") == 3 and len(one) == 6
    assert one == benchmarks.sample(rows, per_group=3, seed=1)
    assert len(benchmarks.sample(rows, limit=5, seed=2)) == 5
    assert benchmarks.sample(rows) == rows


def test_benchmark_list_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from counterfix.cli import main

    assert main(["benchmark", "--list"]) == 0
    out = capsys.readouterr().out
    assert "injection" in out and "tools" in out and "lm-evaluation-harness" in out
