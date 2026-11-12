"""How often do the judge-based RAG metrics agree with labels? (E9)

Scores answer_relevancy, answer_correctness, context_recall, context_precision and
citations on bench/metric_seed.jsonl (26 labelled answers, never used to tune the judge
prompts) and reports, per metric, how often the metric's value falls in the labelled
bucket, with 95% Wilson intervals and every disagreement:

* answer_relevancy, answer_correctness: 1 / 0.5 / 0 = relevant (correct) / partly / not;
* context_recall, citations: 1 = all, 0 = none, anything between = some;
* context_precision: equal (within 0.01) to the average precision computed from the
  labelled useful passages.

Verdicts are cached (bench/cache/judges.jsonl).

    python bench/eval_metrics.py                      # Groq judge from your key
    python bench/eval_metrics.py --judge gemini:gemini-3.5-flash-lite
    python bench/eval_metrics.py --offline

Writes bench/results/metrics.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.eval_judges import CACHE, _NamedOnly  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.core.presets import (  # noqa: E402
    JUDGE_MODELS,
    judge_params,
    load_env_file,
    make_caller,
    rpm_for,
    supports_json_mode,
)
from counterfix.detect.retrieval import record  # noqa: E402
from counterfix.judges import CachedJudge, LLMJudge  # noqa: E402
from counterfix.metrics import get_metric  # noqa: E402

SEED = ROOT / "bench" / "metric_seed.jsonl"
OUT = ROOT / "bench" / "results" / "metrics.md"
METRICS = (
    "answer_relevancy",
    "answer_correctness",
    "context_recall",
    "context_precision",
    "citations",
)
THREE = {1.0: "relevant", 0.5: "partly", 0.0: "irrelevant"}


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{k}/{n} = {p:.2f} ({max(0, c - h):.2f} to {min(1, c + h):.2f})"


def average_precision(ranking: list[str], useful: list[str]) -> float:
    if not useful:
        return 0.0
    hits, total = 0, 0.0
    for k, pid in enumerate(ranking, 1):
        if pid in useful:
            hits += 1
            total += hits / k
    return total / len(useful)


def bucket(name: str, value: float) -> str:
    if name == "answer_relevancy":
        return THREE.get(value, "partly")
    if name == "answer_correctness":
        return {1.0: "correct", 0.5: "partly", 0.0: "incorrect"}.get(value, "partly")
    return "all" if value >= 0.999 else "none" if value <= 0.001 else "some"


_DONE = [0]
_LOCK = __import__("threading").Lock()
_TOTAL = [0]


def score(judge: Any, row: dict[str, Any], name: str) -> tuple[str, str, str]:
    """(expected, got, detail) for one metric on one item."""
    try:
        return _score(judge, row, name)
    finally:
        with _LOCK:
            _DONE[0] += 1
            print(f"\r  judged {_DONE[0]}/{_TOTAL[0]} ({name} {row['id']})" + " " * 20,
                  end="", file=sys.stderr, flush=True)  # fmt: skip


def _score(judge: Any, row: dict[str, Any], name: str) -> tuple[str, str, str]:
    rec = record(row["question"], row["answer"], row["contexts"], reference=row["reference"])
    metric = get_metric(name).bind(judge)
    result = metric.score(rec, row["reference"])
    labels = row["labels"]
    if name == "context_precision":
        want = average_precision([c["id"] for c in row["contexts"]], labels["useful"])
        if result.value is None:
            return f"{want:.2f}", "not judged", result.reason
        return f"{want:.2f}", f"{result.value:.2f}", result.reason
    if result.value is None:
        return labels[name], "not judged", result.reason
    return labels[name], bucket(name, result.value), result.reason


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--judge", action="append", help="provider:model (repeatable)")
    ap.add_argument("--offline", action="store_true", help="cached verdicts only")
    args = ap.parse_args()
    load_env_file(ROOT / ".env")
    specs = args.judge or (
        [f"groq:{JUDGE_MODELS['groq']}"] if os.environ.get("GROQ_API_KEY") else []
    )
    if not specs:
        sys.exit("no judge: add GROQ_API_KEY to .env or pass --judge provider:model")
    rows = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line]
    lines = [
        "# Judge-based RAG metrics against labels",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `bench/eval_metrics.py` on "
        f"bench/metric_seed.jsonl ({len(rows)} answers written for counterfix about fictional "
        "organisations; never used to tune the judge prompts). Small sets: read the intervals.",
    ]
    for spec in specs:
        if args.offline:
            judge: Any = CachedJudge(_NamedOnly(spec), CACHE, offline=True)
        else:
            caller = RateLimitedCaller(make_caller(spec), rpm_for(spec))
            llm = LLMJudge(caller, params=judge_params(spec), json_mode=supports_json_mode(spec))
            judge = CachedJudge(llm, CACHE)
        jobs = [(row, name) for name in METRICS for row in rows
                if (name in row["labels"] or (name == "context_precision" and "useful" in row["labels"]))]  # fmt: skip
        _DONE[0], _TOTAL[0] = 0, len(jobs)
        print(
            f"{spec}: {len(jobs)} judge calls (free tiers are paced: a few minutes)",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            out = list(pool.map(score, [judge] * len(jobs), *zip(*jobs, strict=True)))
        lines += [
            "",
            f"## {spec}",
            "",
            "| Metric | Agrees with the label | Not judged |",
            "| --- | --- | --- |",
        ]
        wrong: list[str] = []
        for name in METRICS:
            got = [
                (row["id"], e, g, d)
                for (row, n), (e, g, d) in zip(jobs, out, strict=True)
                if n == name
            ]
            judged = [x for x in got if x[2] != "not judged"]
            agree = sum(1 for _, e, g, _ in judged if e == g)
            lines.append(f"| {name} | {wilson(agree, len(judged))} | {len(got) - len(judged)} |")
            wrong += [
                f"{name} {i}: label {e}, metric {g} ({d[:90]})" for i, e, g, d in judged if e != g
            ]
        lines += ["", "Disagreements:", *([f"- {w}" for w in wrong] or ["- none"])]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
