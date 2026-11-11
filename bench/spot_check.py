"""Human spot check of the reference labels behind the newer accuracy numbers (E9).

The labels in bench/rag_holdout.jsonl, injection_seed.jsonl, agent_seed.jsonl and
metric_seed.jsonl were written for counterfix as reference labels. Before the README
claims any accuracy from them, a person checks a sample: this shows a fixed random
sample (5 per set by default), each item with its label, and you answer y (agree), n
(disagree, then type what it should be), s (skip) or q (quit). Answers are saved after
each item to bench/results/spot_check_answers.jsonl, so you can stop and resume; the summary goes
to bench/results/spot_check.md. For the main seed set, also run
``python bench/label.py --review 15``; this report includes its result.

    python bench/spot_check.py            # 5 per set, about 10 minutes
    python bench/spot_check.py --per-set 8
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SAVED = (
    ROOT / "results" / "spot_check_answers.jsonl"
)  # not in labels/: label.py reads every file there
OUT = ROOT / "results" / "spot_check.md"
SETS = {
    "rag_holdout": ROOT / "rag_holdout.jsonl",
    "injection_seed": ROOT / "injection_seed.jsonl",
    "agent_seed": ROOT / "agent_seed.jsonl",
    "metric_seed": ROOT / "metric_seed.jsonl",
}


def wrap(text: str) -> str:
    return textwrap.fill(str(text), 100, subsequent_indent="    ")


def show(name: str, row: dict[str, Any]) -> str:
    """Print the item; return its label as text."""
    print("\n" + "=" * 100)
    print(f"{name}  {row['id']}")
    if name == "rag_holdout":
        print(wrap(f"QUESTION: {row['question']}"))
        for c in row["contexts"]:
            print(wrap(f"  [{c['id']}] {c['text']}"))
        print(wrap(f"ANSWER: {row['response']}"))
        print(wrap(f"reference answer: {row.get('answer')}; gold passages: {row.get('relevant')}"))
        label = row["label"]
        return f"cause={label['cause']} (ok / F10 retrieval miss / F11 context ignored), faithful={label['faithful']}"
    if name == "injection_seed":
        print(wrap(f"QUESTION: {row['question']}"))
        print(wrap(f"UNTRUSTED TEXT: {row['untrusted']}"))
        print(wrap(f"ANSWER: {row['answer']}"))
        return f"{row['label']} (followed_injection / ignored_injection / no_injection)"
    if name == "agent_seed":
        print(wrap(f"TASK: {row['task']}"))
        for m in row["messages"][1:]:
            if m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                print(wrap(f"  call {fn['name']}({fn['arguments']})"))
            elif m["role"] == "tool":
                print(wrap(f"    -> {m['content']}"))
            else:
                print(wrap(f"FINAL: {m['content']}"))
        cause = f", cause={row['cause']}" if row.get("cause") else ""
        return f"{row['label']}{cause}"
    print(wrap(f"QUESTION: {row['question']}"))
    for n, c in enumerate(row["contexts"], 1):
        print(wrap(f"  [{n}] ({c['id']}) {c['text']}"))
    print(wrap(f"ANSWER: {row['answer']}"))
    print(wrap(f"REFERENCE: {row['reference']}"))
    return ", ".join(f"{k}={v}" for k, v in row["labels"].items())


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{k}/{n} = {p:.2f} ({max(0, c - h):.2f} to {min(1, c + h):.2f})"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--per-set", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    done: dict[str, dict[str, Any]] = {}
    if SAVED.exists():
        for line in SAVED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    queue = []
    for name, path in SETS.items():
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        for row in random.Random(f"{args.seed}-{name}").sample(rows, min(args.per_set, len(rows))):
            queue.append((name, row))
    todo = [(n, r) for n, r in queue if f"{n}:{r['id']}" not in done]
    print(f"{len(queue)} items in the sample, {len(queue) - len(todo)} already checked.")
    for k, (name, row) in enumerate(todo, 1):
        label = show(name, row)
        print(f"LABEL: {label}")
        answer = ""
        while answer not in ("y", "n", "s", "q"):
            answer = (
                input(f"[{k}/{len(todo)}] agree with the label? y / n / s(kip) / q(uit) > ")
                .strip()
                .lower()
            )
        if answer == "q":
            break
        if answer == "s":
            continue
        note = input("what should it be? > ").strip() if answer == "n" else ""
        entry = {
            "key": f"{name}:{row['id']}",
            "set": name,
            "label": label,
            "agree": answer == "y",
            "note": note,
        }
        done[entry["key"]] = entry
        SAVED.parent.mkdir(exist_ok=True)
        with SAVED.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    lines = ["# Human spot check of reference labels", ""]
    total_ok = total = 0
    lines += ["| Set | Agreed |", "| --- | --- |"]
    for name in SETS:
        rows = [v for v in done.values() if v["set"] == name]
        ok = sum(v["agree"] for v in rows)
        total_ok, total = total_ok + ok, total + len(rows)
        lines.append(f"| {name} | {wilson(ok, len(rows))} |")
    review = ROOT / "labels" / "review_summary.json"
    if review.exists():
        r = json.loads(review.read_text(encoding="utf-8"))
        lines.append(
            f"| main seed (bench/label.py --review) | {wilson(r['agreed'], r['checked'])} |"
        )
        total_ok, total = total_ok + r["agreed"], total + r["checked"]
    lines += ["", f"All checked labels: {wilson(total_ok, total)}", ""]
    lines += [
        f"- {v['key']}: label {v['label']!r}; reviewer: {v['note']}"
        for v in done.values()
        if not v["agree"]
    ] or ["No disagreements."]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
