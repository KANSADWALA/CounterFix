"""Commands that measure and compare models: ``profile``, ``compare``, ``drift``, ``route``."""

from __future__ import annotations

import argparse
import sys

from counterfix.cli.common import (
    _judge_arg,
    _rerun_caller,
    _short_prompt,
)


def cmd_profile(args: argparse.Namespace) -> int:
    """Run the probe suite on one model and save its profile."""
    from counterfix.fixes import profiles

    judge = _judge_arg(args.judge)
    if judge is None:
        return 2
    caller = _rerun_caller(args.spec)

    def progress(n: int, total: int, probe_id: str) -> None:
        print(f"\rprobe {n}/{total} {probe_id}   ", end="", file=sys.stderr, flush=True)

    profile = profiles.build_profile(
        caller, judge=judge, probes=args.probes, fix=args.fix, progress=progress
    )
    print(file=sys.stderr)
    profile.spec = args.spec
    profile.cost_per_mtok = args.cost
    path = profiles.save(profile, args.out)
    print(profiles.table(profile))
    print(f"\nSaved {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Two profiles side by side, or a log of calls re-run on another model."""
    from counterfix.fixes import profiles

    if args.model is None:
        if len(args.items) != 2:
            print("Give two profiles (files or model names), or a log with --model SPEC")
            return 2
        old, new = (profiles.load(x) for x in args.items)
        if old is None or new is None:
            print(
                "Profile not found: "
                + ", ".join(x for x, p in zip(args.items, (old, new), strict=True) if p is None)
            )
            return 2
        print(profiles.compare(old, new))
        return 0
    from counterfix.diagnose import diagnose
    from counterfix.runtime.report import load_log

    judge = _judge_arg(args.judge)
    if judge is None:
        return 2
    caller = _rerun_caller(args.model)
    counts = {"fixed": 0, "still failing": 0, "new failure": 0, "ok": 0}
    for path in args.items:
        for record in load_log(path):
            before = {f.failure_type.value for f in diagnose(record, judge=judge).model_failures}
            new_record = caller.complete(record.messages, **record.params)
            new_record.metadata = {**record.metadata, **new_record.metadata}
            after = {f.failure_type.value for f in diagnose(new_record, judge=judge).model_failures}
            kind = (
                "new failure"
                if after - before
                else "still failing"
                if after
                else "fixed"
                if before
                else "ok"
            )
            counts[kind] += 1
            if kind != "ok":
                was = ",".join(sorted(before)) or "-"
                now = ",".join(sorted(after)) or "-"
                print(f"{kind:<14} {was:>8} -> {now:<8} {_short_prompt(record)}")
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()) + f" on {args.model}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    """Re-run the probes; exit 1 when a task's failure rate moved since the stored profile."""
    from counterfix.fixes import profiles

    old = profiles.load(args.spec)
    if old is None:
        print(f"No stored profile for {args.spec}: run `counterfix profile {args.spec}` first")
        return 2
    judge = _judge_arg(args.judge)
    if judge is None:
        return 2
    new = profiles.build_profile(_rerun_caller(args.spec), judge=judge, probes=args.probes)
    new.spec, new.cost_per_mtok, new.fixes = args.spec, old.cost_per_mtok, old.fixes
    moved = profiles.drift(old, new, args.threshold)
    print(profiles.compare(old, new))
    if not args.no_save:
        previous = profiles.profile_path(new.model).with_suffix(".prev.json")
        profiles.save(old, previous)
        profiles.save(new)
    if moved:
        print("\nDrift: " + "; ".join(moved))
        return 1
    print("\nNo drift beyond the threshold.")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """The cheapest model per task, from profiles (cheapest first, or by cost_per_mtok)."""
    from counterfix.fixes import profiles

    loaded = [profiles.load(x) for x in args.profiles]
    missing = [x for x, p in zip(args.profiles, loaded, strict=True) if p is None]
    if missing:
        print("Profile not found: " + ", ".join(missing))
        return 2
    found = [p for p in loaded if p is not None]
    for task, model in profiles.route(found, args.max_failure).items():
        print(f"{task} {profiles.TASKS.get(task, ''):<36} {model or 'none passes'}")
    return 0
