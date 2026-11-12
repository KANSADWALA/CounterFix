"""Commands on prompts and logged calls: ``lint``, ``diagnose``, ``fix``, ``export-tests``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from counterfix.cli.common import (
    _cli_judge,
    _print_issue,
    _prompts,
    _rerun_caller,
)
from counterfix.judges.base import Judge
from counterfix.lint import JUDGE_RULES, RULES, lint


def cmd_lint(args: argparse.Namespace) -> int:
    rules = args.rules.split(",") if args.rules else None
    known = {**RULES, **JUDGE_RULES}
    unknown = [r for r in rules or [] if r not in known]
    if unknown:
        print(f"Unknown rule(s): {', '.join(unknown)}. Available: {', '.join(known)}")
        return 2
    judge: Judge | None = None
    if args.judge:
        judge = _cli_judge(args.judge)
        if judge is None:
            return 2
    elif any(r in JUDGE_RULES for r in rules or []):
        print("Rules L2, L5 and L6 need a judge: add --judge (picks one from your keys)")
        return 2
    total = warnings = flagged = 0
    for name in args.paths:
        path = Path(name)
        if not path.is_file():
            print(f"Not a file: {name}")
            return 2
        for label, prompt, params in _prompts(path):
            total += 1
            issues = lint(prompt, params=params, rules=rules, judge=judge)
            if not issues and not args.quiet:
                print(f"{label}: ok")
            if issues:
                flagged += 1
                print(f"{label}:")
                for issue in issues:
                    warnings += issue.severity == "warning"
                    _print_issue(issue)
    print(f"\n{total} prompt(s) checked, {flagged} with issues, {warnings} warning(s).")
    return 1 if args.fail_on_warning and warnings else 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Check every call in a JSONL log and print, save or render what was found."""
    from counterfix.diagnose import diagnose
    from counterfix.runtime.report import html_report, load_log, log_line, text_report

    try:
        records = load_log(args.log)
    except (OSError, ValueError) as exc:
        print(exc)
        return 2
    judge: Judge | None = None
    if args.judge:
        judge = _cli_judge(args.judge)
        if judge is None:
            return 2
    reports = []
    for n, record in enumerate(records, 1):
        if judge is not None:
            print(f"\rchecking {n}/{len(records)}...", end="", file=sys.stderr, flush=True)
        report = diagnose(record, judge=judge if judge is not None else False)
        report.judge = judge.name if judge is not None else None
        reports.append(report)
    if judge is not None:
        print(file=sys.stderr)
    print(text_report(reports, failures_only=not args.all))
    if args.json:
        lines = [log_line(r, d) for r, d in zip(records, reports, strict=True)]
        Path(args.json).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nJSON lines written to {args.json}")
    if args.html:
        Path(args.html).write_text(html_report(reports), encoding="utf-8")
        print(f"\nHTML report written to {args.html}")
    return 1 if args.fail_on_failure and any(not d.ok for d in reports) else 0


def cmd_fix(args: argparse.Namespace) -> int:
    """Run the counterfactual engine on every flagged call in a log; save what worked."""
    from counterfix.core.presets import load_env_file
    from counterfix.diagnose import diagnose
    from counterfix.fixes.engine import fix
    from counterfix.runtime.report import load_log

    load_env_file(Path.cwd() / ".env")
    try:
        records = load_log(args.log)
    except (OSError, ValueError) as exc:
        print(exc)
        return 2
    judge: Judge | bool = False
    if args.judge:
        found = _cli_judge(args.judge)
        if found is None:
            return 2
        judge = found
    caller = _rerun_caller(args.caller) if args.caller else None
    rows, fixed = [], 0
    for n, record in enumerate(records, 1):
        report = diagnose(record, judge=judge)
        if not report.model_failures:
            continue
        try:
            result = fix(report, judge=judge, caller=caller, max_runs=args.max_runs)
        except ValueError as exc:
            # the API's own message mentions caller=...; say it in command-line terms
            why = str(exc)
            if "caller=" in why:
                why = "the log does not say which provider served it"
            hint = (
                ""
                if args.caller
                else (
                    " Add --caller provider:model or --caller MODEL@http://host/v1,"
                    " e.g. --caller groq:qwen/qwen3.8-27b"
                )
            )
            print(f"Call {n} ({record.model or 'unknown model'}) cannot be re-run: {why}.{hint}")
            return 2
        fixed += result.fix is not None
        print(f"--- call {n} ({record.model})\n{result.explain()}\n")
        rows.append(
            {
                "record": record.model_dump(mode="json"),
                "fix": None if result.fix is None else result.fix.model_dump(mode="json"),
                "result": result.model_dump(mode="json", exclude={"diagnosis"}),
            }
        )
    print(f"{len(rows)} flagged call(s), {fixed} with a verified fix.")
    if args.out:
        lines = [json.dumps(r, ensure_ascii=False) for r in rows]
        Path(args.out).write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        print(f"Saved to {args.out}; make tests with: counterfix export-tests {args.out}")
    return 0


def cmd_export_tests(args: argparse.Namespace) -> int:
    from counterfix.testing import export_tests

    path = Path(args.fixes)
    if not path.is_file():
        print(f"Not a file: {path}")
        return 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    n = export_tests(rows, Path(args.out))
    print(f"Wrote {n} case(s) to {Path(args.out).with_suffix('.json')} and the test to {args.out}")
    return 0
