"""Measure the RAG checks against the labelled RAG sets.

Two sets: bench/rag_seed.jsonl (development: the judge rules were refined while looking
at results on it, so its scores are optimistic) and bench/rag_holdout.jsonl (held out:
never used to change the rules; quote this one).

    python bench/eval_rag.py                         (judges picked from the keys in .env)
    python bench/eval_rag.py --judge groq:qwen/qwen3.8-27b
    python bench/eval_rag.py --offline               (report from cached verdicts only)

For every item: the free label check (F10 from gold labels), the judge's context-use
check (ok / F10 retrieval miss / F11 context ignored) and the faithfulness metric, with
each judge alone and, when there are several, as a majority-vote panel (no extra calls).
Accuracy comes with a 95% Wilson interval, clear and hard items are reported apart, and
every disagreement with a label is listed. Writes bench/results/rag.md. Verdicts are
cached in bench/cache/judges.jsonl (shared with eval_judges.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.eval_judges import CACHE, _NamedOnly, default_judges  # noqa: E402
from counterfix import Config, run_detectors  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.detect import ContextUseCheck, RetrievalLabelCheck  # noqa: E402
from counterfix.detect.retrieval import record  # noqa: E402
from counterfix.judges import CachedJudge, JudgePanel, LLMJudge  # noqa: E402
from counterfix.judges.base import Judge  # noqa: E402
from counterfix.metrics import Faithfulness  # noqa: E402
from counterfix.presets import (  # noqa: E402
    judge_params,
    load_env_file,
    make_caller,
    rpm_for,
    supports_json_mode,
)

SEED = ROOT / "bench" / "rag_seed.jsonl"
HOLDOUT = ROOT / "bench" / "rag_holdout.jsonl"
SETS = {
    "held-out": (HOLDOUT, "never used to change the judge rules: quote these numbers"),
    "development": (SEED, "the judge rules were refined on it: these numbers are optimistic"),
}
OUT = ROOT / "bench" / "results" / "rag.md"
CAUSES = ("ok", "F10", "F11")


def load(path: Path = SEED) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def as_record(row: dict[str, Any]) -> Any:
    return record(
        row["question"],
        row["response"],
        row["contexts"],
        relevant=row["relevant"],
        reference=row["answer"],
    )


def wilson(hits: int, n: int) -> str:
    """95% Wilson score interval for a proportion, as 'a to b'."""
    if n == 0:
        return "-"
    z, p = 1.96, hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{max(0.0, centre - half):.2f} to {min(1.0, centre + half):.2f}"


def predicted_cause(rec: Any, check: ContextUseCheck) -> str:
    d = run_detectors(rec, [check], Config())
    if d.skipped or d.errors:
        return "error"
    codes = {f.failure_type.value for f in d.findings}
    return "F10" if "F10" in codes else "F11" if "F11" in codes else "ok"


def confusion(truth: list[str], pred: list[str]) -> list[str]:
    pairs = Counter(zip(truth, pred, strict=True))
    head = "| Label \\ judged | " + " | ".join(CAUSES) + " | error |"
    lines = [head, "| --- | --- | --- | --- | --- |"]
    for t in CAUSES:
        cells = " | ".join(str(pairs[(t, p)]) for p in (*CAUSES, "error"))
        lines.append(f"| {t} | {cells} |")
    lines += ["", "| Cause | Precision | Recall |", "| --- | --- | --- |"]
    for c in ("F10", "F11"):
        tp = pairs[(c, c)]
        fp = sum(pairs[(t, c)] for t in CAUSES if t != c)
        fn = sum(pairs[(c, p)] for p in (*CAUSES, "error") if p != c)
        prec = f"{tp / (tp + fp):.2f}" if tp + fp else "-"
        rec = f"{tp / (tp + fn):.2f}" if tp + fn else "-"
        lines.append(f"| {c} | {prec} | {rec} |")
    return lines


class Progress:
    """One stderr line for several judges running side by side, with time left."""

    def __init__(self, names: list[str], total: int):
        self.done = dict.fromkeys(names, 0)
        self.total, self.start = total, time.monotonic()
        self.lock = threading.Lock()
        self.width = 0

    def step(self, name: str) -> None:
        with self.lock:
            self.done[name] += 1
            parts = []
            for n, k in self.done.items():
                left = ""
                if 0 < k < self.total:
                    secs = (time.monotonic() - self.start) / k * (self.total - k)
                    left = f" ~{secs / 60:.0f} min left" if secs >= 60 else f" ~{secs:.0f} s left"
                parts.append(f"{n.removeprefix('judge ').split('/')[-1]} {k}/{self.total}{left}")
            # stay inside an 80-column terminal: a wrapped line cannot be redrawn with \r
            line = " | ".join(parts)[:78]
            print("\r" + line.ljust(self.width), end="", file=sys.stderr, flush=True)
            self.width = len(line)


def section(
    name: str,
    judge: Judge,
    rows: list[dict[str, Any]],
    recs: list[Any],
    progress: Progress | None = None,
    free: list[str] | None = None,
) -> list[str]:
    check = ContextUseCheck(judge)
    faith = Faithfulness().bind(judge)
    truth = [r["label"]["cause"] for r in rows]
    pred, faith_rows = [], []
    for row, rec in zip(rows, recs, strict=True):
        if progress is not None:
            progress.step(name)
        pred.append(predicted_cause(rec, check))
        want = row["label"]["faithful"]
        if want is None:
            continue
        r = faith.score(rec)
        if r.reason.startswith("judge failed"):
            faith_rows.append((row, want, None, r.reason))
            continue
        faith_rows.append((row, want, r.value is None or r.value == 1.0, r.reason))

    lines = [
        "",
        f"### {name}",
        "",
        "| Items | Cause accuracy | 95% interval | Not judged |",
        "| --- | --- | --- | --- |",
    ]
    for label, keep in (
        ("all", lambda r: True),
        ("clear", lambda r: r["kind"] == "clear"),
        ("hard", lambda r: r["kind"] != "clear"),
    ):
        idx = [i for i, r in enumerate(rows) if keep(r)]
        judged = [i for i in idx if pred[i] != "error"]  # an error is not a wrong answer
        hits = sum(pred[i] == truth[i] for i in judged)
        lines.append(
            f"| {label} ({len(idx)}) | {hits}/{len(judged)} | {wilson(hits, len(judged))} | "
            f"{len(idx) - len(judged)} |"
        )
    if free is not None:
        # What counterfix reports: the free F10 from labels takes precedence over the
        # judge (F10 supersedes F11), so a judge that confuses a look-alike passage is
        # corrected whenever gold labels exist.
        combined = ["F10" if f == "F10" else p for f, p in zip(free, pred, strict=True)]
        judged = [i for i, c in enumerate(combined) if c != "error"]
        hits = sum(combined[i] == truth[i] for i in judged)
        lines.append(
            f"| counterfix: free check + judge ({len(rows)}) | {hits}/{len(judged)} | "
            f"{wilson(hits, len(judged))} | {len(rows) - len(judged)} |"
        )
    errors = pred.count("error")
    if errors:
        lines += [
            "",
            f"**{errors} item(s) could not be judged** (a judge error, or not in the cache "
            "with --offline); accuracy is over the judged items only.",
        ]
    scored = [f for f in faith_rows if f[2] is not None]
    agree = sum(f[1] == f[2] for f in scored)
    lines += [
        "",
        f"Faithfulness (every claim supported, or not) agrees with the label on {agree} of "
        f"{len(scored)} items ({wilson(agree, len(scored))}); "
        f"{sum(r['label']['faithful'] is None for r in rows)} items have no faithfulness "
        f"label and {len(faith_rows) - len(scored)} could not be judged.",
        "",
        *confusion(truth, pred),
    ]
    wrong = [
        f"- {r['id']} ({r['kind']}): label {t}, judged {p}"
        for r, t, p in zip(rows, truth, pred, strict=True)
        if p not in (t, "error")
    ]
    wrong += [
        f"- {row['id']} ({row['kind']}): faithfulness label {want}, judged {got}: {why[:120]}"
        for row, want, got, why in scored
        if want != got
    ]
    if wrong:
        lines += ["", "Disagreements with the labels:", "", *wrong]
    return lines


def report_set(
    title: str, note: str, rows: list[dict[str, Any]], judges: dict[str, Judge], panel: bool
) -> list[str]:
    truth = [r["label"]["cause"] for r in rows]
    recs = [as_record(r) for r in rows]
    kinds = Counter(r["kind"] for r in rows if r["kind"] != "clear")
    hard_kinds = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    labels_check = RetrievalLabelCheck()
    free = ["F10" if run_detectors(r, [labels_check], Config()).findings else "ok" for r in recs]
    pairs = list(zip(free, truth, rows, strict=True))
    caught = [r["id"] for f, t, r in pairs if f == "F10" and t == "F10"]
    missed = [f"{r['id']} ({r['kind']})" for f, t, r in pairs if f != "F10" and t == "F10"]
    alarms = [r["id"] for f, t, r in pairs if f == "F10" and t != "F10"]
    lines = [
        "",
        f"## {title} set: {note}",
        "",
        f"{len(rows)} items: {truth.count('ok')} ok, {truth.count('F10')} F10 retrieval miss, "
        f"{truth.count('F11')} F11 context ignored; "
        f"{sum(r['kind'] == 'clear' for r in rows)} clear, {sum(kinds.values())} hard "
        f"({hard_kinds}).",
        "",
        "Free check from gold labels (F10 only, no LLM): "
        f"caught {len(caught)} of {truth.count('F10')} retrieval misses; false alarms: "
        f"{len(alarms)}{' (' + ', '.join(alarms) + ')' if alarms else ''}."
        + (
            f" Not caught: {', '.join(missed)} (the free check fires only when no gold "
            "passage was retrieved; partial retrieval needs the judge)."
            if missed
            else ""
        ),
    ]
    # Each judge has its own provider rate limit, so they run side by side: the run takes
    # as long as the slowest judge. Verdicts are saved as they arrive (resumable).
    progress = Progress([f"judge {spec}" for spec in judges], len(rows))
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        futures = {
            spec: pool.submit(section, f"judge {spec}", judge, rows, recs, progress, free)
            for spec, judge in judges.items()
        }
        sections = {spec: f.result() for spec, f in futures.items()}
    print(file=sys.stderr)
    for spec in judges:
        lines += sections[spec]
    if panel and len(judges) > 1:
        views = [j.cache_only() for j in judges.values() if isinstance(j, CachedJudge)]
        panel_judge = JudgePanel(views)
        lines += section("panel of all judges (majority vote)", panel_judge, rows, recs, None, free)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--judge", action="append", help="provider:model (repeatable)")
    parser.add_argument("--offline", action="store_true", help="cached verdicts only, no calls")
    parser.add_argument("--no-panel", action="store_true", help="skip the majority-vote panel")
    parser.add_argument(
        "--set", choices=["both", *SETS], default="both", help="which labelled set (default both)"
    )
    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    specs = args.judge or default_judges()
    if not specs:
        sys.exit("no judge: add a key to .env or pass --judge provider:model")
    judges: dict[str, Judge] = {}
    for spec in specs:
        if args.offline:
            judges[spec] = CachedJudge(_NamedOnly(spec), CACHE, offline=True)
            continue
        caller = RateLimitedCaller(make_caller(spec), rpm_for(spec))
        llm = LLMJudge(caller, params=judge_params(spec), json_mode=supports_json_mode(spec))
        judges[spec] = CachedJudge(llm, CACHE)
    lines = [
        "# RAG checks against the labelled RAG sets",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `bench/eval_rag.py`. "
        "All passages are written for this project about fictional organisations, so a judge "
        "cannot answer from memory; labels are reference labels following the rules in the "
        "judge task. With this many items, read the 95% intervals, not the point estimates.",
    ]
    for title, (path, note) in SETS.items():
        if args.set in ("both", title):
            print(f"{title} set:", file=sys.stderr)
            lines += report_set(title, note, load(path), judges, not args.no_panel)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
