"""Command line: ``counterfix lint FILE...``.

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
from counterfix.lint import RULES, LintIssue, lint


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


def cmd_lint(args: argparse.Namespace) -> int:
    rules = args.rules.split(",") if args.rules else None
    unknown = [r for r in rules or [] if r not in RULES]
    if unknown:
        print(f"Unknown rule(s): {', '.join(unknown)}. Available: {', '.join(RULES)}")
        return 2
    total = warnings = flagged = 0
    for name in args.paths:
        path = Path(name)
        if not path.is_file():
            print(f"Not a file: {name}")
            return 2
        for label, prompt, params in _prompts(path):
            total += 1
            issues = lint(prompt, params=params, rules=rules)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="counterfix", description="Find why LLM calls fail.")
    parser.add_argument("--version", action="version", version=f"counterfix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p_lint = sub.add_parser("lint", help="check prompts for failure-prone patterns")
    p_lint.add_argument("paths", nargs="+", help=".txt, .md, .json or .jsonl files")
    p_lint.add_argument("--rules", help="comma-separated rule ids, e.g. L1,L4")
    p_lint.add_argument("--quiet", action="store_true", help="only print prompts with issues")
    p_lint.add_argument(
        "--fail-on-warning", action="store_true", help="exit with code 1 if any warning (for CI)"
    )
    p_lint.set_defaults(func=cmd_lint)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
