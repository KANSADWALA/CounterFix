"""Helpers shared by the commands: reading prompt files, picking a judge, parsing
KEY=VALUE options."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from counterfix.judges.base import Judge
from counterfix.lint import LintIssue


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
    print(f"      try: {issue.suggestion}")


def _cli_judge(choice: str) -> Judge | None:
    """The judge for ``--judge`` ("auto" picks one from your keys, reading .env); None on error."""
    from counterfix.core.presets import load_env_file
    from counterfix.judges.auto import NoJudgeError, resolve_judge

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


def _key_values(pairs: Sequence[str] | None, what: str) -> dict[str, Any]:
    """["max_tokens=512", "temperature=0"] -> {"max_tokens": 512, "temperature": 0}."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"{what} needs NAME=VALUE, got {pair!r}")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def _no_answers(result: Any) -> str:
    """A message when no item got an answer (every call failed): a CI gate must not pass."""
    if result.items and all(i.record.error is not None for i in result.items):
        first = result.items[0].record.error or ""
        return f"\nERROR: no item got an answer; every call failed (first: {first[:200]})"
    return ""


def _rerun_caller(spec: str) -> Any:
    """A paced caller for any model spec the CLI takes (provider:model, anthropic:model,
    MODEL@http://host/v1); prints the problem and exits 2 on a spec it cannot use."""
    from counterfix.core.caller import RateLimitedCaller
    from counterfix.core.presets import caller_for, pace_for

    try:
        return RateLimitedCaller(caller_for(spec, timeout=120), pace_for(spec))
    except ValueError as exc:
        print(f"cannot use the model {spec!r}: {exc}")
        raise SystemExit(2) from None


def _judge_arg(choice: str | None) -> Judge | bool | None:
    """False for no judge, a judge, or None when --judge was given but none could be set up."""
    if not choice:
        return False
    return _cli_judge(choice)


def _short_prompt(record: Any) -> str:
    text = " ".join(record.prompt_text.split())
    return text if len(text) <= 60 else text[:57] + "..."
