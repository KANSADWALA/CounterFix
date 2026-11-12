"""Commands that run a model over data: ``eval`` and ``benchmark``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from counterfix.cli.common import (
    _cli_judge,
    _key_values,
    _no_answers,
)


def cmd_eval(args: argparse.Namespace) -> int:
    """Run a model over a dataset, score it, explain the misses, verify fixes."""
    from counterfix.evals import evaluate
    from counterfix.evals.data import is_suite, load_suite

    try:
        params = _key_values(args.param, "--param") or None
        gates = _key_values(args.fail_under, "--fail-under")
        if is_suite(args.data):
            gates = {**load_suite(args.data).get("fail_under", {}), **gates}
    except (ValueError, OSError) as exc:
        print(exc)
        return 2
    judge: Any = False
    if args.judge:
        judge = _cli_judge(args.judge)
        if judge is None:
            return 2
    if args.allow_code or args.sandbox:
        import os

        from counterfix.detect.code.sandbox import ALLOW_ENV, SANDBOX_ENV

        if args.allow_code:
            os.environ[ALLOW_ENV] = "1"
        if args.sandbox:
            os.environ[SANDBOX_ENV] = args.sandbox
    app = None
    if args.app:
        try:
            from counterfix.evals.data import load_callable

            app = load_callable(args.app)
        except (ValueError, ImportError, AttributeError) as exc:
            print(f"--app {args.app}: {exc}")
            return 2

    def progress(n: int, total: int, label: str) -> None:
        # a fix-search label is long; an item id is short. Pad to wipe the previous line.
        text = (
            f"{n}/{total} {label}"
            if label.startswith("fix search")
            else f"item {n}/{total} {label[:30]}"
        )
        print(f"\r{text[:110]:<110}", end="", file=sys.stderr, flush=True)

    try:
        result = evaluate(
            args.data,
            model=args.model,
            metrics=args.metric,
            judge=judge,
            fix=args.fix,
            max_runs=args.max_runs,
            input=args.input,
            reference=args.reference,
            template=args.template,
            system=args.system,
            params=params,
            limit=args.limit,
            out=args.out,
            progress=progress,
            app=app,
            concurrency=args.concurrency,
            samples=args.samples,
        )
    except (ValueError, OSError, ImportError, RuntimeError) as exc:  # incl. NoJudgeError
        print(file=sys.stderr)
        print(exc)
        return 2
    print(file=sys.stderr)
    print(result.text())
    if args.json:
        result.save(args.json)
        print(f"\nResult written to {args.json}")
    if args.html:
        Path(args.html).write_text(result.html(), encoding="utf-8")
        print(f"\nHTML report written to {args.html}")
    broken = _no_answers(result)
    if broken:
        print(broken)
        return 2
    failed_gates = []
    for name, floor in gates.items():
        try:
            summary = result.metric(name)
        except KeyError as exc:
            print(exc)
            return 2
        value = summary.pass_rate if summary.pass_rate is not None else summary.mean
        if value is None or value < float(floor):
            failed_gates.append(f"{name} {value} < {floor}")
    strict = args.fail_on_flagged or (
        is_suite(args.data) and load_suite(args.data).get("fail_on_flagged")
    )
    if strict and result.flagged:
        causes = ", ".join(f"{c} {k}" for c, k in result.flagged_by_cause.items())
        failed_gates.append(f"{len(result.flagged)} item(s) flagged by live checks ({causes})")
    if failed_gates:
        print("\nFAILED: " + "; ".join(failed_gates))
        return 1
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run a standard benchmark on a model, with the misses explained."""
    from counterfix.evals.benchmarks import BENCHMARKS, run_benchmark

    if args.list or not args.name:
        print("probe sets (for MMLU, GSM8K, HumanEval ... use lm-evaluation-harness or EvalPlus):")
        for b in BENCHMARKS.values():
            print(f"  {b.name:<12} {b.title}\n  {'':<12} {b.source} · {b.licence}")
        return 0
    if not args.model and not args.app:
        print("give --model SPEC (or --app MODULE:FN)")
        return 2
    judge: Any = False
    if args.judge:
        judge = _cli_judge(args.judge)
        if judge is None:
            return 2
    app = None
    try:
        if args.app:
            from counterfix.evals.data import load_callable

            app = load_callable(args.app)
        params = _key_values(args.param, "--param") or None
    except (ValueError, ImportError, AttributeError) as exc:
        print(exc)
        return 2

    def progress(n: int, total: int, label: str) -> None:
        text = f"{n}/{total} {label}" if label.startswith("fix search") else f"item {n}/{total}"
        print(f"\r{text[:110]:<110}", end="", file=sys.stderr, flush=True)

    try:
        result = run_benchmark(
            args.name, model=args.model, app=app, limit=args.limit, per_group=args.per_group,
            seed=args.seed, params=params, judge=judge, fix=args.fix, max_runs=args.max_runs,
            out=args.out, concurrency=args.concurrency, samples=args.samples,
            progress=progress,
        )  # fmt: skip
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(file=sys.stderr)
        print(exc)
        return 2
    print(file=sys.stderr)
    print(result.text())
    if args.json:
        result.save(args.json)
        print(f"\nResult written to {args.json}")
    if args.html:
        Path(args.html).write_text(result.html(), encoding="utf-8")
        print(f"\nHTML report written to {args.html}")
    broken = _no_answers(result)
    if broken:
        print(broken)
        return 2
    return 0
