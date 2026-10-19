"""Run the counterfactual engine on every labelled seed failure and report the fix rate.

    python bench/fix_seed.py              # resumable: finished items are kept in fixes.jsonl
    python bench/fix_seed.py --report     # rebuild bench/results/fixes.md from saved rows
    python bench/fix_seed.py --limit 5 --type F1
    python bench/fix_seed.py --learn      # add the fix statistics to the shipped profiles
    python bench/fix_seed.py --type F1 --rows bench/results/fixes_profiled.jsonl
                                          # a second run, kept apart (profiles now in use)

Each failure is re-run on the model that produced it (Groq gpt-oss-120b, local Ollama
qwen2.5:3b), judged by a panel of two fast judges (cached). Ctrl+C keeps what is done.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.eval_judges import CACHE, labelled_records  # noqa: E402
from counterfix import CallRecord, fix  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.judges import CachedJudge, JudgePanel, LLMJudge  # noqa: E402
from counterfix.presets import (  # noqa: E402
    judge_params,
    load_env_file,
    make_caller,
    rpm_for,
    supports_json_mode,
)

ROWS = ROOT / "bench" / "results" / "fixes.jsonl"
JUDGES = ["groq:qwen/qwen3.8-27b", "gemini:gemini-3.5-flash-lite"]
_lock = threading.Lock()


def target_spec(key: str, model: str) -> str:
    return f"{key.split('__', 1)[0]}:{model}"  # run names start with the provider


def make_panel() -> JudgePanel:
    judges = []
    for spec in JUDGES:
        caller = RateLimitedCaller(make_caller(spec), rpm_for(spec))
        judge = LLMJudge(caller, params=judge_params(spec), json_mode=supports_json_mode(spec))
        judges.append(CachedJudge(judge, CACHE))
    return JudgePanel(judges)


def run_one(record: CallRecord, truth: set[str], key: str, panel: Any, callers: dict) -> dict:
    spec = target_spec(key, record.model or "")
    with _lock:
        if spec not in callers:
            callers[spec] = RateLimitedCaller(make_caller(spec, timeout=120), rpm_for(spec))
    start = time.monotonic()
    try:
        result = fix(record, judge=panel, caller=callers[spec])
    except Exception as exc:
        return {"key": key, "truth": sorted(truth), "error": f"{type(exc).__name__}: {exc}"}
    found = [f.failure_type.value for f in result.diagnosis.model_failures]
    return {
        "key": key,
        "truth": sorted(truth),
        "found": found,
        "cause_right": bool(found) and found[0] in truth,
        "retry": None if result.retry is None else [result.retry.passed, result.retry.runs],
        "attempts": [[a.name, a.passed, a.runs] for a in result.attempts],
        "fix": None if result.fix is None else result.fix.name,
        "fix_type": None if result.fix is None else result.fix.failure_type,
        "calls": result.calls,
        "stopped": result.stopped,
        "seconds": round(time.monotonic() - start, 1),
    }


def report(rows: list[dict]) -> str:
    ok = [r for r in rows if "error" not in r]
    found = [r for r in ok if r["found"]]
    right = [r for r in found if r["cause_right"]]
    retry_ok = [r for r in ok if r["retry"] and r["retry"][0] >= 2]
    fixed = [r for r in ok if r["fix"]]
    fixed_right = [r for r in fixed if r["fix_type"] in r["truth"]]
    needs_fix = [r for r in right if r not in retry_ok]
    calls = sum(r["calls"] for r in ok)

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({a / b:.0%})" if b else "0/0"

    lines = [
        "# Counterfactual engine on the seed failures",
        "",
        f"{len(rows)} labelled failures ({len(rows) - len(ok)} errors). Judges: "
        f"{', '.join(JUDGES)} (panel). A fix counts only if the failure is gone, with no new "
        "failure, in 2 of 3 re-runs; a plain retry runs first.",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Failure detected | {pct(len(found), len(ok))} |",
        f"| Top cause matches the label | {pct(len(right), len(ok))} |",
        f"| Plain retry already passes (no fix claimed) | {pct(len(retry_ok), len(ok))} |",
        "| Verified fix, cause right, retry did not pass | "
        f"{pct(len(fixed_right), len(needs_fix))} |",
        f"| Verified fix (any) | {pct(len(fixed), len(ok))} |",
        f"| Re-runs of the target model | {calls} ({calls / max(len(ok), 1):.1f} per failure) |",
        "",
    ]
    by_type: dict[str, Counter[str]] = {}
    for r in ok:
        t = r["truth"][0]
        c = by_type.setdefault(t, Counter())
        c["n"] += 1
        c["retry"] += r in retry_ok
        c["fixed"] += r["fix"] is not None
    lines += ["| Type | Failures | Retry passes | Verified fix |", "| --- | --- | --- | --- |"]
    for t in sorted(by_type):
        c = by_type[t]
        lines.append(f"| {t} | {c['n']} | {c['retry']} | {c['fixed']} |")
    used = Counter(r["fix"] for r in fixed)
    lines += ["", "Fixes that verified: " + ", ".join(f"{k} {v}" for k, v in used.most_common())]
    lines += [
        "",
        "| Item | Label | Found | Retry | Tried | Fix |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['key']} | {','.join(r['truth'])} | error | | | {r['error'][:60]} |")
            continue
        retry = f"{r['retry'][0]}/{r['retry'][1]}" if r["retry"] else "-"
        tried = "; ".join(f"{n} {p}/{k}" for n, p, k in r["attempts"]) or "-"
        lines.append(
            f"| {r['key'].split('__')[1]}:{r['key'].split(':')[-1]} | {','.join(r['truth'])} "
            f"| {','.join(r['found']) or '-'} | {retry} | {tried} | {r['fix'] or '-'} |"
        )
    return "\n".join(lines) + "\n"


def load_rows(path: Path = ROWS) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["key"]: r for r in rows}


def learn(rows: list[dict]) -> list[Path]:
    """Set each model's fix statistics in its shipped profile from these rows (rebuilt from
    scratch each time, so running it twice does not double the counts)."""
    from counterfix.engine import INTERVENTIONS
    from counterfix.profiles import SHIPPED, Profile, load, profile_path, save

    kind = {i.name: i.failure_type.value for i in INTERVENTIONS}
    by_model: dict[str, Profile] = {}
    for r in rows:
        model = (
            r["key"]
            .split("__")[1]
            .replace("openai-", "openai/")
            .replace("qwen2.5-3b", "qwen2.5:3b")
        )
        if model not in by_model:
            by_model[model] = load(model) or Profile(model=model)
            by_model[model].fixes = {}
        profile = by_model[model]
        for name, passed, _runs in r.get("attempts", []):
            stats = profile.fixes.setdefault(kind[name], {})
            verified, tried = stats.get(name, [0, 0])
            stats[name] = [verified + (passed >= 2), tried + 1]
    return [save(p, SHIPPED / profile_path(m).name) for m, p in by_model.items()]


def main() -> None:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--type", help="only labels of this type, e.g. F1")
    parser.add_argument("--report", action="store_true", help="rebuild the report only")
    parser.add_argument("--redo-errors", action="store_true", help="re-run rows that errored")
    parser.add_argument("--learn", action="store_true", help="update the shipped profiles")
    parser.add_argument("--rows", type=Path, default=ROWS, help="results file (JSON lines)")
    args = parser.parse_args()
    rows_path, report_path = args.rows, args.rows.with_suffix(".md")
    if args.learn:
        for path in learn(list(load_rows(rows_path).values())):
            print(f"Updated {path}")
        return

    done = load_rows(rows_path)
    if not args.report:
        items = [
            (r, t, k)
            for r, t, k in labelled_records(False)
            if t and (args.type is None or args.type in t)
        ][: args.limit]
        todo = [
            i for i in items if i[2] not in done or (args.redo_errors and "error" in done[i[2]])
        ]
        print(f"{len(items)} failures, {len(todo)} to run", flush=True)
        panel, callers = make_panel(), {}
        groups: dict[str, list] = {}
        for item in todo:  # one worker per target provider, so free tiers are not flooded
            groups.setdefault(item[2].split("__", 1)[0], []).append(item)

        def worker(batch: list) -> None:
            for record, truth, key in batch:
                row = run_one(record, truth, key, panel, callers)
                with _lock:
                    done[key] = row
                    with rows_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        f"[{len(done)}] {key}: "
                        f"{row.get('fix') or row.get('stopped') or row.get('error')}",
                        flush=True,
                    )

        try:
            with ThreadPoolExecutor(max_workers=len(groups) or 1) as pool:
                list(pool.map(worker, groups.values()))
        except KeyboardInterrupt:
            print("\nstopped; finished rows are saved")
    labelled = {k for _, t, k in labelled_records(False) if t}
    rows = [r for k, r in done.items() if k in labelled]
    report_path.write_text(report(rows), encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
