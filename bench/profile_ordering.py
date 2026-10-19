"""Does a model profile cut re-runs? Fix the same failures with and without one.

    python bench/profile_ordering.py            # local Ollama qwen2.5:3b, no keys needed

Uses 6 held-out truncation prompts (not in the seed set the profile learned from), live
checks only. Writes bench/results/profile_ordering.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from counterfix import Message, fix, profiles  # noqa: E402
from counterfix.presets import make_caller  # noqa: E402

SPEC = "ollama:qwen2.5:3b"
PROMPTS = [
    "Describe the water cycle in detail, covering every stage with an example.",
    "Write a step-by-step beginner's guide to baking sourdough bread.",
    "Explain how a bill becomes a law in India, from drafting to assent.",
    "List twenty common Python built-in functions, each with a one-line example.",
    "Write a short story of about 500 words about a lighthouse keeper.",
    "Compare five popular databases (PostgreSQL, MySQL, SQLite, MongoDB, Redis) in detail.",
]


def main() -> None:
    caller = make_caller(SPEC, timeout=180)
    profile = profiles.load(SPEC)
    if profile is None:
        sys.exit(f"no profile for {SPEC}")
    rows = []
    for prompt in PROMPTS:
        messages = [Message(role="user", content=prompt)]
        record = caller.complete(messages, max_tokens=64)
        record.provider = "ollama"
        if record.finish_reason != "length":
            print(f"skipped (not cut off): {prompt[:50]}")
            continue
        plain = fix(record, judge=False, caller=caller, profile=None)
        ordered = fix(record, judge=False, caller=caller, profile=profile)
        rows.append((prompt, plain, ordered))
        print(f"{plain.calls:>2} -> {ordered.calls:>2} re-runs  {prompt[:60]}", flush=True)
    if not rows:
        sys.exit("no prompt was cut off")
    a = sum(p.calls for _, p, _ in rows) / len(rows)
    b = sum(o.calls for _, _, o in rows) / len(rows)
    fixed_a = sum(p.fix is not None for _, p, _ in rows)
    fixed_b = sum(o.fix is not None for _, _, o in rows)
    lines = [
        "# Profile ordering on held-out failures",
        "",
        f"{len(rows)} truncated answers from `{SPEC}` (max_tokens 64), prompts not in the seed "
        "set. Live checks only. Same failures fixed twice: default intervention order, then "
        "the order from the model's shipped profile.",
        "",
        "| | Re-runs per failure | Verified fixes |",
        "| --- | --- | --- |",
        f"| Default order | {a:.1f} | {fixed_a}/{len(rows)} |",
        f"| Profile order | {b:.1f} | {fixed_b}/{len(rows)} |",
        "",
        "| Prompt | Default | Profile | Fix (profile order) |",
        "| --- | --- | --- | --- |",
    ]
    for prompt, plain, ordered in rows:
        name = ordered.fix.name if ordered.fix else "-"
        lines.append(f"| {prompt[:50]} | {plain.calls} | {ordered.calls} | {name} |")
    out = ROOT / "bench" / "results" / "profile_ordering.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{a:.1f} -> {b:.1f} re-runs per failure. Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
