"""Measure how well the judges agree with your hand labels (Phase 3 'done when').

    python bench/eval_judges.py          (judges picked from the API keys in .env)
    python bench/eval_judges.py --judge groq:openai/gpt-oss-20b --judge ollama:qwen2.5:3b

Runs the judge-based checks with each judge alone and with all of them as a voting panel,
plus the free live checks as a baseline, on every labelled response. Writes
bench/results/judges.md and judges.json. Verdicts are cached in bench/cache/, so a re-run
(or adding a judge) only pays for new calls. Free-tier limits are respected with --rpm.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.label import LABELS, load_labels  # noqa: E402
from bench.seed import load_seed  # noqa: E402
from counterfix import CallRecord, Config, FailureType, Message, run_detectors  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.core.detector import Detector  # noqa: E402
from counterfix.detectors import default_detectors, judge_detectors  # noqa: E402
from counterfix.detectors.fmt import SCHEMA_METADATA_KEY  # noqa: E402
from counterfix.judges import CachedJudge, JudgePanel, LLMJudge  # noqa: E402
from counterfix.judges.base import Judge  # noqa: E402
from counterfix.judges.tasks import REFERENCE_KEY  # noqa: E402
from counterfix.presets import load_env_file, make_caller, parse_spec  # noqa: E402

RESULTS = ROOT / "bench" / "results"
CACHE = ROOT / "bench" / "cache" / "judges.jsonl"
DEFAULT_RPM = {"groq": 28.0, "gemini": 12.0, "ollama": 0.0, "openai": 60.0, "openrouter": 15.0}
WORKERS_PER_JUDGE = 4  # calls in flight per judge; the rate limiter still spaces them
TYPES = [t for t in FailureType if t.is_model_failure]


class _NamedOnly:
    """Stands in for a judge in --offline mode; only its name is used (as the cache key)."""

    def __init__(self, spec: str):
        self._name = "typesafe:jev" if spec in ("jev", "typesafe:jev") else _caller_name(spec)

    @property
    def name(self) -> str:
        return self._name

    def ask(self, task: Any) -> Any:
        raise RuntimeError("offline mode never calls a judge")


def _caller_name(spec: str) -> str:
    provider, model = parse_spec(spec)
    return f"{provider}:{model}"


def default_judges() -> list[str]:
    """Three model families when the keys allow, chosen from what is configured."""
    import os

    judges = []
    if os.environ.get("GROQ_API_KEY"):
        judges += ["groq:openai/gpt-oss-20b", "groq:qwen/qwen3.8-27b"]
    if os.environ.get("GEMINI_API_KEY"):
        judges.append("gemini:gemini-3.5-flash-lite")
    if os.environ.get("TYPESAFE_API_KEY"):
        judges.append("jev")
    return judges


def _short(name: str) -> str:
    """Compact setup names for the one-line progress display."""
    if name.startswith("live"):
        return "live"
    if name.startswith("panel"):
        return "panel"
    return name.replace("judge ", "").split(":")[-1].split("/")[-1][:14]


def labelled_records(with_reference: bool) -> list[tuple[CallRecord, set[str], str]]:
    """(record, true failure codes, label key) for every labelled response."""
    seed = {i.id: i for i in load_seed()}
    runs: dict[str, dict[str, Any]] = {}
    out = []
    for key, label in load_labels().items():
        run_name = label["run"]
        if run_name not in runs:
            path = next((RESULTS / "phase2").glob(f"{run_name}.json"), None)
            if path is None:
                print(f"warning: run file for {key} not found; skipped")
                continue
            runs[run_name] = json.loads(path.read_text(encoding="utf-8"))
        row = next(r for r in runs[run_name]["rows"] if r["id"] == label["id"])
        item = seed[label["id"]]
        record = CallRecord(
            id=key,
            model=label["model"],
            messages=[Message.model_validate(m.model_dump()) for m in item.messages],
            params={"max_tokens": 4096, **item.params},
            response_text=row["answer_full"],
            finish_reason=row.get("finish_reason"),
        )
        if item.schema_ is not None:
            record.metadata[SCHEMA_METADATA_KEY] = item.schema_
        if with_reference and item.gold:
            record.metadata[REFERENCE_KEY] = item.gold
        out.append((record, set(label["labels"]), key))
    return out


def review_note() -> str:
    path = ROOT / "bench" / "labels" / "review_summary.json"
    if not path.is_file():
        return " No human spot-check of the Claude labels yet (run bench/label.py --review 15)."
    review = json.loads(path.read_text(encoding="utf-8"))
    return (
        f" Human spot-check: agreed with {review['agreed']} of {review['checked']} "
        f"Claude labels ({review['agreement']:.0%})."
    )


def label_sources() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in load_labels().values():
        counts[item["source"]] += 1
    return dict(counts)


def score(predicted: dict[str, set[str]], truth: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for t in TYPES:
        code = t.value
        tp = sum(code in predicted[k] and code in truth[k] for k in truth)
        fp = sum(code in predicted[k] and code not in truth[k] for k in truth)
        fn = sum(code not in predicted[k] and code in truth[k] for k in truth)
        positives = tp + fn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / positives if positives else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if positives else None)
        )
        table[code] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "positives": positives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return table


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def main() -> None:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        help="provider:model, repeatable (3 families recommended)",
    )
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="give judges the gold answer (an easier, upper-bound setting)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="build the report from cached verdicts only (no API calls)",
    )
    args = parser.parse_args()
    if not args.judge:
        args.judge = default_judges()
        if not args.judge:
            sys.exit(
                "no judge available: add GROQ_API_KEY or GEMINI_API_KEY to .env, "
                "or pass --judge ollama:<model>"
            )
        print("Judges (picked from your API keys): " + ", ".join(args.judge))

    data = labelled_records(args.with_reference)[: args.limit]
    if not data:
        sys.exit(f"no labels found in {LABELS}; run bench/label.py first")
    truth = {key: codes for _, codes, key in data}

    judges: dict[str, Judge] = {}
    for spec in args.judge:
        if args.offline:  # no clients, no keys needed: verdicts come from the cache
            judges[spec] = CachedJudge(_NamedOnly(spec), CACHE, offline=True)
            continue
        if spec in ("jev", "typesafe:jev"):  # optional, needs counterfix[jev] + TYPESAFE_API_KEY
            from counterfix.judges.jev import JevJudge

            judges["typesafe:jev"] = CachedJudge(JevJudge(), CACHE)
            continue
        provider, _ = parse_spec(spec)
        caller = RateLimitedCaller(make_caller(spec), DEFAULT_RPM.get(provider, 0.0))
        judges[spec] = CachedJudge(LLMJudge(caller), CACHE)
    setups: dict[str, list[Detector]] = {"live checks (no LLM)": default_detectors()}
    for spec, judge in judges.items():
        setups[f"judge {spec}"] = judge_detectors(judge)
    if len(judges) > 1 and not args.no_panel:
        # The panel only combines verdicts the judges already gave: no new calls, no retries.
        offline_views = [j.cache_only() for j in judges.values() if isinstance(j, CachedJudge)]
        setups["panel of all judges"] = judge_detectors(JudgePanel(offline_views))

    progress: dict[str, int] = {name: 0 for name in setups}
    started = time.monotonic()

    def evaluate(name: str, detectors: list[Detector], workers: int) -> dict[str, Any]:
        def one(entry: tuple[CallRecord, set[str], str]) -> tuple[str, Any]:
            record, _, key = entry
            diagnosis = run_detectors(record, detectors, Config())
            progress[name] += 1
            return key, diagnosis

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, data))
        predicted = {
            key: {f.failure_type.value for f in d.findings if f.failure_type.is_model_failure}
            for key, d in results
        }
        failures = sum(sum("judge failed" in r for r in d.skipped.values()) for _, d in results)
        errors = sum(len(d.errors) for _, d in results)
        return {"scores": score(predicted, truth), "judge_failures": failures, "errors": errors}

    done = threading.Event()

    def show_progress() -> None:
        while not done.wait(2.0):
            elapsed = time.monotonic() - started
            parts = [f"{_short(n)} {c}" for n, c in progress.items()]
            line = f"{' | '.join(parts)} of {len(data)} [{elapsed / 60:.1f}m]"
            print("\r" + line[:78].ljust(78), end="", flush=True)  # fits one terminal line

    printer = threading.Thread(target=show_progress, daemon=True)
    printer.start()
    report: dict[str, Any] = {}
    single = [n for n in setups if n.startswith("judge ")]
    report["live checks (no LLM)"] = evaluate(
        "live checks (no LLM)", setups["live checks (no LLM)"], 1
    )
    # Each judge has its own provider/model rate limit, so all judges run at the same time.
    with ThreadPoolExecutor(max_workers=max(1, len(single))) as pool:
        futures = {n: pool.submit(evaluate, n, setups[n], WORKERS_PER_JUDGE) for n in single}
        for n, future in futures.items():
            report[n] = future.result()
    if "panel of all judges" in setups:  # mostly cache hits by now
        report["panel of all judges"] = evaluate(
            "panel of all judges", setups["panel of all judges"], WORKERS_PER_JUDGE
        )
    done.set()
    print(f"\nFinished in {(time.monotonic() - started) / 60:.1f} min.\n")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    counts = {t.value: sum(t.value in c for c in truth.values()) for t in TYPES}
    lines = [
        "# Judge accuracy against hand labels",
        "",
        f"Generated {stamp} by `bench/eval_judges.py` on {len(data)} labelled responses"
        + (" (judges were given the gold answer)" if args.with_reference else "")
        + ".",
        "",
        "Labelled failures per type: "
        + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        + ".",
        "Precision = of the responses flagged, how many truly failed. Recall = of the true",
        "failures, how many were flagged. With few positives, treat numbers as rough.",
        "",
    ]
    for name, result in report.items():
        lines += [
            f"## {name}",
            "",
            f"Judge failures (skipped checks): {result['judge_failures']}. "
            f"Internal errors: {result['errors']}.",
            "",
            "| Type | Positives | TP | FP | FN | Precision | Recall | F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for code, s in result["scores"].items():
            if s["positives"] or s["fp"]:
                lines.append(
                    f"| {code} | {s['positives']} | {s['tp']} | {s['fp']} | {s['fn']} | "
                    f"{fmt(s['precision'])} | {fmt(s['recall'])} | {fmt(s['f1'])} |"
                )
        lines.append("")
    (RESULTS / "judges.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "judges.json").write_text(
        json.dumps(
            {
                "generated": stamp,
                "n": len(data),
                "label_counts": counts,
                "with_reference": args.with_reference,
                "setups": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"Saved {(RESULTS / 'judges.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
