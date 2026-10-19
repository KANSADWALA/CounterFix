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


def cmd_fix(args: argparse.Namespace) -> int:
    """Run the counterfactual engine on every flagged call in a log; save what worked."""
    from counterfix.core.caller import RateLimitedCaller
    from counterfix.diagnose import diagnose
    from counterfix.engine import fix
    from counterfix.presets import load_env_file, make_caller, rpm_for
    from counterfix.report import load_log

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
    caller = None
    if args.caller:
        caller = RateLimitedCaller(make_caller(args.caller), rpm_for(args.caller))
    rows, fixed = [], 0
    for n, record in enumerate(records, 1):
        report = diagnose(record, judge=judge)
        if not report.model_failures:
            continue
        try:
            result = fix(report, judge=judge, caller=caller, max_runs=args.max_runs)
        except ValueError:
            print(
                f"Call {n} ({record.model or 'unknown model'}) cannot be re-run: the log does "
                "not say which provider served it. Add --caller provider:model, e.g. "
                "--caller groq:qwen/qwen3.8-27b"
            )
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


def _rerun_caller(spec: str) -> Any:
    from counterfix.core.caller import RateLimitedCaller
    from counterfix.presets import load_env_file, make_caller, rpm_for

    load_env_file(Path.cwd() / ".env")
    return RateLimitedCaller(make_caller(spec, timeout=120), rpm_for(spec))


def _judge_arg(choice: str | None) -> Judge | bool | None:
    """False for no judge, a judge, or None when --judge was given but none could be set up."""
    if not choice:
        return False
    return _cli_judge(choice)


def cmd_profile(args: argparse.Namespace) -> int:
    """Run the probe suite on one model and save its profile."""
    from counterfix import profiles

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
    from counterfix import profiles

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
    from counterfix.report import load_log

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


def _short_prompt(record: Any) -> str:
    text = " ".join(record.prompt_text.split())
    return text if len(text) <= 60 else text[:57] + "..."


def cmd_drift(args: argparse.Namespace) -> int:
    """Re-run the probes; exit 1 when a task's failure rate moved since the stored profile."""
    from counterfix import profiles

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
    from counterfix import profiles

    loaded = [profiles.load(x) for x in args.profiles]
    missing = [x for x, p in zip(args.profiles, loaded, strict=True) if p is None]
    if missing:
        print("Profile not found: " + ", ".join(missing))
        return 2
    found = [p for p in loaded if p is not None]
    for task, model in profiles.route(found, args.max_failure).items():
        print(f"{task} {profiles.TASKS.get(task, ''):<36} {model or 'none passes'}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the MCP server on stdio (needs the mcp extra)."""
    try:
        from counterfix.mcp_server import main as serve
    except ImportError:
        print('The MCP server needs: pip install "counterfix[mcp]"')
        return 2
    serve()
    return 0


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
    p_fix = sub.add_parser("fix", help="find verified fixes for the failed calls in a log")
    p_fix.add_argument("log", help="JSONL log of calls")
    p_fix.add_argument("--judge", nargs="?", const="auto", help="also fix judge-found failures")
    p_fix.add_argument(
        "--caller", metavar="SPEC", help="re-run on this model, e.g. groq:qwen/qwen3.8-27b"
    )
    p_fix.add_argument("--max-runs", type=int, default=15, help="re-runs per call (default 15)")
    p_fix.add_argument("--out", metavar="PATH", help="save results as JSON lines")
    p_fix.set_defaults(func=cmd_fix)
    p_export = sub.add_parser("export-tests", help="turn verified fixes into pytest cases")
    p_export.add_argument("fixes", help="JSON lines from counterfix fix --out")
    p_export.add_argument(
        "--out", default="tests/test_prompt_regressions.py", help="pytest file to write"
    )
    p_export.set_defaults(func=cmd_export_tests)
    p_prof = sub.add_parser("profile", help="measure a model on the probe suite")
    p_prof.add_argument("spec", help="provider:model, e.g. groq:qwen/qwen3.8-27b")
    p_prof.add_argument("--judge", nargs="?", const="auto", help="add judge checks")
    p_prof.add_argument("--fix", action="store_true", help="also learn which fixes work")
    p_prof.add_argument("--probes", help="your own probes (.jsonl, same format)")
    p_prof.add_argument("--cost", type=float, help="price per million tokens, for route")
    p_prof.add_argument("--out", help="where to save (default ~/.counterfix/profiles)")
    p_prof.set_defaults(func=cmd_profile)
    p_cmp = sub.add_parser("compare", help="compare two profiles, or re-run a log on a model")
    p_cmp.add_argument("items", nargs="+", help="two profiles, or JSONL logs with --model")
    p_cmp.add_argument("--model", metavar="SPEC", help="re-run the logged calls on this model")
    p_cmp.add_argument("--judge", nargs="?", const="auto", help="add judge checks")
    p_cmp.set_defaults(func=cmd_compare)
    p_drift = sub.add_parser("drift", help="re-run the probes and alert on changes")
    p_drift.add_argument("spec", help="provider:model with a stored profile")
    p_drift.add_argument("--threshold", type=float, default=0.15, help="default 0.15")
    p_drift.add_argument("--judge", nargs="?", const="auto", help="add judge checks")
    p_drift.add_argument("--probes", help="your own probes (.jsonl)")
    p_drift.add_argument("--no-save", action="store_true", help="keep the stored profile")
    p_drift.set_defaults(func=cmd_drift)
    p_route = sub.add_parser("route", help="cheapest model per task from profiles")
    p_route.add_argument("profiles", nargs="+", help="profiles (files or model names)")
    p_route.add_argument("--max-failure", type=float, default=0.2, help="default 0.2")
    p_route.set_defaults(func=cmd_route)
    p_mcp = sub.add_parser("mcp", help="run the MCP server for coding agents (stdio)")
    p_mcp.set_defaults(func=cmd_mcp)
    p_setup = sub.add_parser("setup", help="show which keys, local models and judge will be used")
    p_setup.add_argument("--test", action="store_true", help="ask the judge one tiny question")
    p_setup.set_defaults(func=cmd_setup)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
