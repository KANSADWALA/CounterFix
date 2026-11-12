"""``counterfix diagnose FILE --from TOOL`` and ``counterfix fix FILE --from TOOL``: the
failures DeepEval, Ragas or Promptfoo found, explained and fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from counterfix.cli.common import _cli_judge, _rerun_caller
from counterfix.core.record import Message
from counterfix.integrations.eval_results import ResultCase, read_results


def _thresholds(items: list[str] | None) -> dict[str, str]:
    """--threshold faithfulness=0.8 / noise_sensitivity<=0.3 -> {name: ">=0.8" | "<=0.3"}."""
    out = {}
    for item in items or []:
        if "<=" in item:
            name, value = item.split("<=", 1)
            out[name.strip()] = f"<={value.strip()}"
        elif "=" in item:
            name, value = item.split("=", 1)
            out[name.strip().rstrip(">")] = value.strip()
        else:
            raise ValueError(f"--threshold {item!r}: use NAME=VALUE or NAME<=VALUE")
    return out


def _cases(args: argparse.Namespace) -> list[ResultCase] | None:
    try:
        return read_results(args.log, args.source, thresholds=_thresholds(args.threshold))
    except (OSError, ValueError, KeyError) as exc:
        print(exc)
        return None


def cmd_diagnose_results(args: argparse.Namespace) -> int:
    """Explain each failure in an evaluation tool's results file."""
    from counterfix.diagnose import diagnose
    from counterfix.runtime.report import counts

    cases = _cases(args)
    if cases is None:
        return 2
    judge = None
    if args.judge:
        judge = _cli_judge(args.judge)
        if judge is None:
            return 2
    failed = [c for c in cases if c.failed]
    shown = cases if args.all else failed
    reports, explained = [], 0
    for case in shown:
        print(f"--- {case.name}" + (f" [{case.record.model}]" if case.record.model else ""))
        print(f"  {case.tool}: {'passed' if not case.failed else case.reason or 'failed'}")
        if case.errored:
            print("  the call itself errored, so there is no answer to diagnose")
            continue
        report = diagnose(case.record, judge=judge if judge is not None else False)
        report.judge = judge.name if judge is not None else None  # its source was printed
        reports.append(report)
        if report.model_failures:
            explained += case.failed
            print("  counterfix: " + report.explain().replace("\n", "\n  "))
        elif case.failed:
            print(
                "  counterfix: no cause found"
                + (
                    ""
                    if judge is not None
                    else " by the live checks (add --judge to check facts too)"
                )
            )
    print(
        f"\n{len(cases)} test case(s), {len(failed)} failed in the tool, "
        f"{explained} with a cause counterfix found."
    )
    for name, n in sorted(counts(reports).items()):
        print(f"  {name:<40} {n}")
    return 0


def cmd_fix_results(args: argparse.Namespace) -> int:
    """Search for verified fixes for the failures in an evaluation tool's results file."""
    from counterfix.core.presets import load_env_file
    from counterfix.fixes.engine import CheckError, fix

    load_env_file(Path.cwd() / ".env")
    cases = _cases(args)
    if cases is None:
        return 2
    judge: Any = False
    if args.judge:
        judge = _cli_judge(args.judge)
        if judge is None:
            return 2
    user_check = None
    if args.check:
        from counterfix.evals.data import load_callable

        try:
            user_check = load_callable(args.check)
        except (ImportError, AttributeError, ValueError, OSError) as exc:
            print(f"Cannot load the check {args.check}: {exc}")
            return 2
    system = None
    if args.system:
        path = Path(args.system)
        if path.is_file():
            system = path.read_text(encoding="utf-8")
        elif " " not in args.system and path.suffix.lower() in (".txt", ".md", ".prompt"):
            print(f"--system {args.system}: no such file")  # not sent as the prompt itself
            return 2
        else:
            system = args.system
    callers: dict[str, Any] = {}
    rows, fixed, skipped = [], 0, 0
    todo = [c for c in cases if c.failed and not c.errored]
    for case in todo:
        record = case.record.model_copy(deep=True)
        if system is not None:
            if any(m.role in ("system", "developer") for m in record.messages):
                print(
                    f"--- {case.name}: the prompt in the file already has a system message;"
                    " --system is not added"
                )
            else:
                record.messages.insert(0, Message(role="system", content=system))
        spec = args.caller or case.rerun
        if spec is None:
            skipped += 1
            print(
                f"--- {case.name}: cannot be re-run: the file does not say which model served"
                " it. Add --caller provider:model, e.g. --caller groq:llama-3.1-8b-instant"
            )
            continue
        if spec not in callers:
            callers[spec] = _rerun_caller(spec)
        check = user_check or case.check
        print(f"--- {case.name} [{spec}]\n  {case.tool}: {case.reason or 'failed'}")
        for note in case.notes:
            print(f"  note: {note}")
        try:
            result = fix(
                record, judge=judge, caller=callers[spec], max_runs=args.max_runs, check=check
            )
        except CheckError as exc:
            print(f"  {exc}")
            return 2
        if check is None and not result.diagnosis.model_failures:
            print(
                "  counterfix found no cause it can target, so no change was tried"
                + ("" if judge is not False else " (add --judge to check facts too)")
                + "\n"
            )
        else:
            print("  " + result.explain().replace("\n", "\n  ") + "\n")
        fixed += result.fix is not None
        rows.append(
            {
                "case": case.name,
                "tool_reason": case.reason,
                "record": record.model_dump(mode="json"),
                "fix": None if result.fix is None else result.fix.model_dump(mode="json"),
                "result": result.model_dump(mode="json", exclude={"diagnosis"}),
            }
        )
    print(
        f"{len(todo)} failed test case(s), {fixed} with a verified fix"
        + (f", {skipped} skipped (no model to re-run)." if skipped else ".")
    )
    if args.out:
        lines = [json.dumps(r, ensure_ascii=False) for r in rows]
        Path(args.out).write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        print(f"Saved to {args.out}; make tests with: counterfix export-tests {args.out}")
    return 2 if todo and skipped == len(todo) else 0
