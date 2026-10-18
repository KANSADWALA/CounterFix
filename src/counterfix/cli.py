"""Command line: ``counterfix lint FILE...``, ``counterfix diagnose LOG``, ``counterfix setup``.

Accepted files:
  * .txt / .md      the whole file is one user prompt
  * .json           a list of chat messages, or an object with "messages" (and optional "params")
  * .jsonl          one such object per line (e.g. bench/seed_failures.jsonl)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from counterfix import __version__
from counterfix.judges.base import Judge
from counterfix.lint import JUDGE_RULES, RULES, LintIssue, lint


def _prompts(path: Path) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    """Yield (label, prompt, params) for every prompt in the file."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                obj = json.loads(line)
                label = f"{path.name}:{obj.get('id', n)}"
                yield label, obj["messages"], obj.get("params") or {}
    elif suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            yield path.name, obj, {}
        else:
            yield path.name, obj["messages"], obj.get("params") or {}
    else:
        yield path.name, text, {}


def _print_issue(issue: LintIssue) -> None:
    mark = "!" if issue.severity == "warning" else "i"
    causes = ", ".join(t.value for t in issue.failure_types)
    print(f"  [{mark}] {issue.rule} {issue.title}" + (f"  (risk: {causes})" if causes else ""))
    print(f"      {issue.message}")
    if issue.snippet:
        print(f'      at: "{issue.snippet}"')
    print(f"      fix: {issue.suggestion}")


def _cli_judge(choice: str) -> Judge | None:
    """The judge for ``--judge`` ("auto" picks one from your keys, reading .env); None on error."""
    from counterfix.judges.auto import NoJudgeError, resolve_judge
    from counterfix.presets import load_env_file

    load_env_file(Path.cwd() / ".env")
    try:
        judge, source = resolve_judge(None if choice == "auto" else choice)
    except NoJudgeError as exc:
        print(exc)
        return None
    except Exception as exc:
        print(f"Could not set up the judge {choice!r}: {exc}")
        return None
    print(f"Judge: {judge.name} ({source})\n")
    return judge


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
    from counterfix.report import html_report, load_log, log_line, text_report

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


def cmd_setup(args: argparse.Namespace) -> int:
    """Show what counterfix can use on this machine. Never prints key values."""
    import importlib.util
    import os
    import time

    from counterfix.judges.auto import (
        ENV_VAR,
        NoJudgeError,
        detect_judges,
        ollama_models,
        resolve_judge,
    )
    from counterfix.presets import PRESETS, load_env_file

    env_file = Path.cwd() / ".env"
    load_env_file(env_file)
    print(f"counterfix {__version__}\n")
    print("Live checks (guard):  ready, no key needed")
    sdk = importlib.util.find_spec("openai") is not None
    print(
        "OpenAI-compatible SDK: "
        + ("installed" if sdk else 'missing  ->  pip install "counterfix[openai]"')
    )
    print(f".env in this folder:  {'found (loaded)' if env_file.is_file() else 'none'}")
    print("\nAPI keys (values are never shown):")
    for _, key_var, _ in PRESETS.values():
        if key_var:
            print(f"  {key_var:<20} {'set' if os.environ.get(key_var) else '-'}")
    models = ollama_models()
    print(f"  {'Ollama (local)':<20} " + (", ".join(models[:5]) if models else "not running"))
    chosen = os.environ.get(ENV_VAR, "").strip()
    print(f"  {ENV_VAR:<20} {chosen or '-'}")
    print("\nJudge checks (counterfix.diagnose, lint --judge):")
    if chosen:
        print(f"  will use {chosen} ({ENV_VAR})")
    else:
        found = detect_judges()
        for n, (spec, why) in enumerate(found):
            print(f"  {'will use' if n == 0 else 'also can'} {spec}  ({why})")
        if not found:
            print("  no judge found. Without one, diagnose() lets the checked model judge")
            print("  itself (with a warning). For an independent judge:")
            print("    GROQ_API_KEY    free at https://console.groq.com/keys")
            print("    GEMINI_API_KEY  free at https://aistudio.google.com/apikey")
            print("    or Ollama       https://ollama.com, then: ollama pull qwen2.5:7b")
    if not args.test:
        print("\nRun `counterfix setup --test` to send the judge one tiny question.")
        return 0
    from counterfix.judges.base import JudgeTask

    try:
        judge, source = resolve_judge()
    except NoJudgeError as exc:
        print(f"\n{exc}")
        return 1
    task = JudgeTask(
        name="setup_test",
        instructions="Is the model answer correct?",
        material={"user_request": "What is 2 + 2?", "model_answer": "4"},
        labels=("correct", "incorrect"),
    )
    start = time.perf_counter()
    verdict = judge.ask(task)
    took = time.perf_counter() - start
    if verdict.failed:
        print(f"\nJudge test FAILED with {judge.name}: {verdict.reason}")
        return 1
    print(f"\nJudge test ok: {judge.name} said {verdict.label!r} in {took:.1f}s ({source})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="counterfix", description="Find why LLM calls fail.")
    parser.add_argument("--version", action="version", version=f"counterfix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p_lint = sub.add_parser("lint", help="check prompts for failure-prone patterns")
    p_lint.add_argument("paths", nargs="+", help=".txt, .md, .json or .jsonl files")
    p_lint.add_argument("--rules", help="comma-separated rule ids, e.g. L1,L4")
    p_lint.add_argument("--quiet", action="store_true", help="only print prompts with issues")
    p_lint.add_argument(
        "--judge",
        nargs="?",
        const="auto",
        help="also run L2, L5, L6. Alone it picks a judge from your keys (reads .env); "
        "or name one, e.g. --judge groq:qwen/qwen3.8-27b",
    )
    p_lint.add_argument(
        "--fail-on-warning", action="store_true", help="exit with code 1 if any warning (for CI)"
    )
    p_lint.set_defaults(func=cmd_lint)
    p_diag = sub.add_parser("diagnose", help="check a JSONL log of calls and report failures")
    p_diag.add_argument("log", help="JSONL log, e.g. from Config(on_diagnosis=log_to(path))")
    p_diag.add_argument(
        "--judge",
        nargs="?",
        const="auto",
        help="also run judge checks (extra calls). Alone it picks a judge from your keys; "
        "or name one, e.g. --judge groq:qwen/qwen3.8-27b",
    )
    p_diag.add_argument("--html", metavar="PATH", help="also write a shareable HTML report")
    p_diag.add_argument("--json", metavar="PATH", help="also write results as JSON lines")
    p_diag.add_argument("--all", action="store_true", help="list calls with no failure too")
    p_diag.add_argument(
        "--fail-on-failure", action="store_true", help="exit with code 1 if any call failed (CI)"
    )
    p_diag.set_defaults(func=cmd_diagnose)
    p_setup = sub.add_parser("setup", help="show which keys, local models and judge will be used")
    p_setup.add_argument("--test", action="store_true", help="ask the judge one tiny question")
    p_setup.set_defaults(func=cmd_setup)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
