"""Run the seed failure set against a real model and report what the live checks catch.

    python bench/run_seed.py groq --model openai/gpt-oss-120b
    python bench/run_seed.py ollama --model qwen2.5:3b --types F1,F2,F3
    python bench/run_seed.py groq --limit 10          # quick smoke test

Each prompt is sent once, without counterfix's help. The live detectors and the prompt
linter are then applied, and the results are saved to bench/results/phase2/.
Keys come from the project's .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.seed import SeedItem, load_seed  # noqa: E402
from counterfix import Config, FailureType, lint, record_from_openai, run_detectors  # noqa: E402
from counterfix.detectors import default_detectors  # noqa: E402
from counterfix.detectors.fmt import SCHEMA_METADATA_KEY  # noqa: E402
from counterfix.presets import PRESETS, load_env_file, make_client  # noqa: E402

RESULTS = ROOT / "bench" / "results" / "phase2"
LIVE_TYPES = {FailureType.TRUNCATION, FailureType.FORMAT_BREAK, FailureType.OVER_REFUSAL}


def run_item(client: Any, model: str, item: SeedItem, retries: int = 2) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in item.messages],
        **item.params,
    }
    # Generous limit so only the F1 items (which set their own low limit) hit it.
    request.setdefault("max_tokens", 4096)
    row: dict[str, Any] = {"id": item.id, "expected": item.failure_type.value}
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**request)
            break
        except Exception as error:
            name = type(error).__name__
            if name == "RateLimitError" and attempt < retries:
                time.sleep(20)
                continue
            row["error"] = f"{name}: {getattr(error, 'message', None) or error}"
            return row
    latency = (time.perf_counter() - start) * 1000
    record = record_from_openai(request, response, latency_ms=latency)
    if item.schema_ is not None:
        record.metadata[SCHEMA_METADATA_KEY] = item.schema_
    diagnosis = run_detectors(record, default_detectors(), Config())
    row.update(
        finish_reason=record.finish_reason,
        latency_s=round(latency / 1000, 2),
        reasoned=record.has_reasoning,
        answer=record.answer_text[:160],
        answer_full=record.answer_text,
        reasoning_chars=len(record.reasoning_text or ""),
        found=[f.failure_type.value for f in diagnosis.findings],
        evidence=[e.detail for f in diagnosis.findings for e in f.evidence],
        skipped=diagnosis.skipped,
        lint=[i.rule for i in lint(item.messages, params=item.params)],
    )
    return row


def summarise(rows: list[dict[str, Any]]) -> list[str]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["expected"]].append(row)
    lines = [
        f"{'type':<5} {'n':>3} {'err':>4} {'caught':>7}  {'other flags':<26} lint",
        "-" * 64,
    ]
    for ft in sorted(by_type):
        group = by_type[ft]
        ok = [r for r in group if "error" not in r]
        caught = sum(ft in r["found"] for r in ok)
        others = Counter(t for r in ok for t in r["found"] if t != ft)
        linted = sum(bool(r["lint"]) for r in ok)
        caught_text = f"{caught}/{len(ok)}" if FailureType(ft) in LIVE_TYPES else "-"
        other_text = ", ".join(f"{t}x{n}" for t, n in sorted(others.items())) or "-"
        lines.append(
            f"{ft:<5} {len(group):>3} {len(group) - len(ok):>4} {caught_text:>7}  "
            f"{other_text:<26} {linted}/{len(ok)}"
        )
    lines += [
        "",
        "caught = the live checks found the expected type (only F1-F3 have live checks).",
        "A miss can mean the model did not fail on that prompt, not that the check is wrong;",
        "read the per-item rows in the saved JSON to tell the two apart.",
    ]
    return lines


def main() -> None:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", choices=sorted(PRESETS))
    parser.add_argument("--model")
    parser.add_argument("--types", help="comma-separated, e.g. F1,F2,F3 (default: all)")
    parser.add_argument("--limit", type=int, help="run only the first N items")
    parser.add_argument("--sleep", type=float, help="seconds between calls (default 2.5 for groq)")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    try:
        client = make_client(args.provider, timeout=args.timeout)
    except Exception as exc:
        sys.exit(str(exc))
    default_model = PRESETS[args.provider][2]
    model = args.model or default_model
    sleep = args.sleep if args.sleep is not None else (2.5 if args.provider == "groq" else 0.5)

    items = load_seed()
    if args.types:
        wanted = {t.strip().upper() for t in args.types.split(",")}
        items = [i for i in items if i.failure_type.value in wanted]
    items = items[: args.limit] if args.limit else items

    print(f"Running {len(items)} seed prompts on {args.provider}:{model}\n")
    rows = []
    for n, item in enumerate(items, 1):
        row = run_item(client, model, item)
        rows.append(row)
        status = row.get("error") or (", ".join(row["found"]) or "clean")
        print(f"[{n:>2}/{len(items)}] {item.id}  {status}")
        if n < len(items):
            time.sleep(sleep)

    print()
    summary = summarise(rows)
    print("\n".join(summary))
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    out = RESULTS / f"{args.provider}__{safe_model}__{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "provider": args.provider,
                "model": model,
                "created_utc": stamp,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
