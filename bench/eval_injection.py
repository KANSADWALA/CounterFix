"""How well does the F13 judge check find prompt injections that were followed? (E7)

Runs :class:`counterfix.detect.InjectionCheck` on the 20 labelled answers in
bench/injection_seed.jsonl (8 followed, 8 ignored, 4 with no planted instruction) and
reports precision and recall for "followed", with 95% Wilson intervals, the confusion
table and every disagreement. Verdicts are cached (bench/cache/judges.jsonl).

    python bench/eval_injection.py                       # Groq judge from your key
    python bench/eval_injection.py --judge gemini:gemini-3.5-flash-lite
    python bench/eval_injection.py --offline             # cached verdicts only

Writes bench/results/injection.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.eval_judges import CACHE, _NamedOnly  # noqa: E402
from counterfix import Config  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.core.detector import SkipCheck  # noqa: E402
from counterfix.core.record import CallRecord, Message  # noqa: E402
from counterfix.detect import InjectionCheck  # noqa: E402
from counterfix.judges import CachedJudge, LLMJudge  # noqa: E402
from counterfix.judges.tasks import CONTEXT_KEY, injection_followed  # noqa: E402
from counterfix.presets import (  # noqa: E402
    JUDGE_MODELS,
    judge_params,
    load_env_file,
    make_caller,
    rpm_for,
    supports_json_mode,
)

SEED = ROOT / "bench" / "injection_seed.jsonl"
OUT = ROOT / "bench" / "results" / "injection.md"
LABELS = ("followed_injection", "ignored_injection", "no_injection")


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{k}/{n} = {p:.2f} ({max(0, c - h):.2f} to {min(1, c + h):.2f})"


def as_record(row: dict[str, Any]) -> CallRecord:
    return CallRecord(
        messages=[Message(role="user", content=row["question"])],
        response_text=row["answer"],
        metadata={CONTEXT_KEY: row["untrusted"], "rag_question": row["question"]},
    )


def predict(check: InjectionCheck, row: dict[str, Any]) -> str:
    try:
        findings = check.check(as_record(row), Config())
    except SkipCheck as exc:
        return f"error: {exc}"[:80]
    if findings:
        return "followed_injection"
    task = injection_followed(as_record(row), row["untrusted"], row["question"])
    verdict = check.judge.ask(task)  # cached: this repeats no call
    return verdict.label if verdict.label in LABELS else "ignored_injection"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--judge", action="append", help="provider:model (repeatable)")
    ap.add_argument("--offline", action="store_true", help="cached verdicts only")
    args = ap.parse_args()
    load_env_file(ROOT / ".env")
    import os

    specs = args.judge or (
        [f"groq:{JUDGE_MODELS['groq']}"] if os.environ.get("GROQ_API_KEY") else []
    )
    if not specs:
        sys.exit("no judge: add GROQ_API_KEY to .env or pass --judge provider:model")
    rows = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line]
    lines = [
        "# F13 prompt-injection check against labelled answers",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `bench/eval_injection.py` "
        f"on bench/injection_seed.jsonl ({len(rows)} answers written for counterfix: "
        f"{sum(r['label'] == 'followed_injection' for r in rows)} followed, "
        f"{sum(r['label'] == 'ignored_injection' for r in rows)} ignored, "
        f"{sum(r['label'] == 'no_injection' for r in rows)} without a planted instruction). "
        "Positive class: followed. Read the intervals: 20 items is a small set.",
    ]
    for spec in specs:
        if args.offline:
            judge: Any = CachedJudge(_NamedOnly(spec), CACHE, offline=True)
        else:
            caller = RateLimitedCaller(make_caller(spec), rpm_for(spec))
            llm = LLMJudge(caller, params=judge_params(spec), json_mode=supports_json_mode(spec))
            judge = CachedJudge(llm, CACHE)
        check = InjectionCheck(judge)
        with ThreadPoolExecutor(max_workers=4) as pool:
            preds = list(pool.map(predict, [check] * len(rows), rows))
        truth = [r["label"] for r in rows]
        tp = sum(p == t == "followed_injection" for p, t in zip(preds, truth, strict=True))
        pred_pos = sum(p == "followed_injection" for p in preds)
        true_pos = sum(t == "followed_injection" for t in truth)
        errors = sum(p.startswith("error") for p in preds)
        lines += [
            "",
            f"## {spec}",
            "",
            f"- precision (said followed, was followed): {wilson(tp, pred_pos)}",
            f"- recall (followed, and found): {wilson(tp, true_pos)}",
            f"- exact label: {wilson(sum(p == t for p, t in zip(preds, truth, strict=True)), len(rows))}",
            f"- not judged (errors): {errors}",
            "",
            "| truth \\ judged | " + " | ".join(LABELS) + " | error |",
            "| --- |" + " --- |" * (len(LABELS) + 1),
        ]
        for t in LABELS:
            counts = Counter(
                p if p in LABELS else "error" for p, tt in zip(preds, truth, strict=True) if tt == t
            )
            lines.append(
                f"| {t} | " + " | ".join(str(counts[x]) for x in (*LABELS, "error")) + " |"
            )
        wrong = [(r["id"], t, p) for r, t, p in zip(rows, truth, preds, strict=True) if t != p]
        lines += [
            "",
            "Disagreements: " + (", ".join(f"{i} ({t} -> {p})" for i, t, p in wrong) or "none"),
        ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
