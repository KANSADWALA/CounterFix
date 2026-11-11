"""How well does the F14 judge check tell finished agent runs from failed ones? (E8)

Runs :class:`counterfix.detectors.AgentTaskCheck` on the 20 labelled runs in
bench/agent_seed.jsonl (10 completed, 10 not, with the cause) and reports precision and
recall for "not completed" with 95% Wilson intervals, how often the named cause matches
the label, and every disagreement. Verdicts are cached (bench/cache/judges.jsonl).

    python bench/eval_agents.py                     # Groq judge from your key
    python bench/eval_agents.py --judge gemini:gemini-3.5-flash-lite
    python bench/eval_agents.py --offline

Writes bench/results/agents.md.
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
from counterfix import agents  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.judges import CachedJudge, LLMJudge  # noqa: E402
from counterfix.judges.tasks import agent_task  # noqa: E402
from counterfix.presets import (  # noqa: E402
    JUDGE_MODELS,
    judge_params,
    load_env_file,
    make_caller,
    rpm_for,
    supports_json_mode,
)

SEED = ROOT / "bench" / "agent_seed.jsonl"
OUT = ROOT / "bench" / "results" / "agents.md"


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{k}/{n} = {p:.2f} ({max(0, c - h):.2f} to {min(1, c + h):.2f})"


def judge_run(judge: Any, row: dict[str, Any]) -> tuple[str, str]:
    traj = agents.from_messages(row["messages"], tools=row["tools"], task=row["task"])
    v = judge.ask(agent_task(traj))
    if v.failed:
        return "error", ""
    return v.label, str(v.extra.get("cause") or "")


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
        "# F14 agent-task check against labelled runs",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `bench/eval_agents.py` on "
        f"bench/agent_seed.jsonl ({len(rows)} runs written for counterfix: "
        f"{sum(r['label'] == 'completed' for r in rows)} completed, "
        f"{sum(r['label'] == 'not_completed' for r in rows)} not). Positive class: not completed. "
        "20 runs is a small set: read the intervals.",
    ]
    for spec in specs:
        if args.offline:
            judge: Any = CachedJudge(_NamedOnly(spec), CACHE, offline=True)
        else:
            caller = RateLimitedCaller(make_caller(spec), rpm_for(spec))
            llm = LLMJudge(caller, params=judge_params(spec), json_mode=supports_json_mode(spec))
            judge = CachedJudge(llm, CACHE)
        with ThreadPoolExecutor(max_workers=4) as pool:
            preds = list(pool.map(judge_run, [judge] * len(rows), rows))
        truth = [r["label"] for r in rows]
        labels = [p for p, _ in preds]
        tp = sum(p == t == "not_completed" for p, t in zip(labels, truth, strict=True))
        flagged = sum(p == "not_completed" for p in labels)
        failed = sum(t == "not_completed" for t in truth)
        cause_ok = sum(
            1 for r, (p, c) in zip(rows, preds, strict=True)
            if r["label"] == p == "not_completed" and c == r["cause"]
        )  # fmt: skip
        errors = sum(p == "error" for p in labels)
        wrong = [(r["id"], r["label"] + (f"/{r['cause']}" if r["cause"] else ""), p + (f"/{c}" if c else ""))
                 for r, (p, c) in zip(rows, preds, strict=True) if p != r["label"] or (p == "not_completed" and c != r["cause"])]  # fmt: skip
        lines += [
            "",
            f"## {spec}",
            "",
            f"- precision (said not completed, was not): {wilson(tp, flagged)}",
            f"- recall (not completed, and caught): {wilson(tp, failed)}",
            f"- exact label: {wilson(sum(p == t for p, t in zip(labels, truth, strict=True)), len(rows))}",
            f"- cause named right, of the failed runs caught: {wilson(cause_ok, tp)}",
            f"- not judged (errors): {errors}",
            "",
            "Disagreements (label -> judged): "
            + (", ".join(f"{i} ({a} -> {b})" for i, a, b in wrong) or "none"),
        ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
