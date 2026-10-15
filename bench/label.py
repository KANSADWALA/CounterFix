"""Hand-label model responses so judge accuracy can be measured.

    python bench/label.py                  (newest full run for each model, found automatically)
    python bench/label.py --auto           (objective prompts labelled automatically)
    python bench/label.py --review 15      (spot-check 15 labels made by Claude)
    python bench/label.py bench/results/phase2/groq__openai-gpt-oss-120b__20261015-142145.json

For each response you see the prompt, the model's full answer and the seed notes (gold
answer, readings, constraints). Type the failure codes the answer shows, e.g. F4 or
F4,F5; "n" for no failure; "s" to skip; "u" to undo the last label; "?" for the code
table; "q" to save and quit.

Objective failures are suggested for you (F1 from the stop reason, F2 from a JSON/schema
check, F8 by comparing with the known answer): press Enter to accept a suggestion.
With --auto, prompts built for F1, F2 or F8 are labelled from those checks without asking,
so you only judge the rest. Judgement-based types (F3-F7, F9) are never suggested, because
those labels are what the judges are measured against.
Labels are saved after every answer to bench/labels/labels.jsonl, so you can stop and
resume at any time. Label what the answer actually does, not what the prompt was built
to trigger.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.seed import SeedItem, load_seed  # noqa: E402
from counterfix import CallRecord, Config, FailureType, Message  # noqa: E402
from counterfix.detectors import FormatDetector  # noqa: E402
from counterfix.detectors.fmt import SCHEMA_METADATA_KEY  # noqa: E402
from counterfix.detectors.judged import answer_contains  # noqa: E402

LABEL_DIR = ROOT / "bench" / "labels"
LABELS = LABEL_DIR / "labels.jsonl"  # your own labels (and --auto labels); always win
HUMAN_SOURCES = {"human", "human-review"}
CODES = {t.value for t in FailureType if t.is_model_failure}
HELP = (
    "codes F1-F9 comma-separated | Enter = accept suggestion | n = no failure | s = skip | "
    "u = undo | ? = code table | q = quit"
)
GUIDE = {
    "F1": "answer is empty or cut off (finish=length)",
    "F2": "broken JSON, wrong schema, or JSON inside a code fence",
    "F3": "refuses or waters down a harmless request",
    "F4": "states an invented or unsupported fact",
    "F5": "goes along with a false premise",
    "F6": "breaks an explicit instruction in the prompt",
    "F7": "(needs rewordings; leave for Phase 5 unless obvious)",
    "F8": "wrong arithmetic or letter/string work",
    "F9": "silently picks one meaning of an ambiguous request",
    "n": "no failure, incl. an honest 'I don't know' or a premise correction",
}
OBJECTIVE = {FailureType.TRUNCATION, FailureType.FORMAT_BREAK, FailureType.COMPUTATION}


def legend() -> str:
    width = max(len(k) for k in GUIDE)
    return "\n".join(f"  {k:<{width}}  {v}" for k, v in GUIDE.items())


def suggest(row: dict[str, Any], item: SeedItem) -> tuple[list[str], list[str]]:
    """Objective labels with their reasons: F1 from the stop reason, F2 from the format
    check, F8 from the known answer. A passed check is reported too, so Enter can accept
    "no failure". Judgement-based types are never suggested."""
    codes: list[str] = []
    reasons: list[str] = []
    answer = row.get("answer_full") or ""
    truncated = (row.get("finish_reason") or "").lower() in ("length", "max_tokens")
    if truncated:
        codes.append("F1")
        empty = " with no visible answer" if not answer.strip() else ""
        reasons.append(f"F1: stopped at the token limit{empty}")
    elif item.failure_type is FailureType.TRUNCATION:
        reasons.append("F1 check passed: the answer finished normally")
    if not truncated and (
        item.schema_ is not None or item.failure_type is FailureType.FORMAT_BREAK
    ):
        record = CallRecord(
            messages=[Message.model_validate(m.model_dump()) for m in item.messages],
            params=dict(item.params),
            response_text=answer,
            finish_reason=row.get("finish_reason"),
        )
        if item.schema_ is not None:
            record.metadata[SCHEMA_METADATA_KEY] = item.schema_
        findings = FormatDetector().check(record, Config())
        if findings:
            codes.append("F2")
            reasons.append("F2: " + findings[0].evidence[0].detail)
        else:
            reasons.append("F2 check passed: valid JSON that matches the schema")
    if item.failure_type is FailureType.COMPUTATION and item.gold and answer.strip():
        expected: Any = int(item.gold) if item.gold.lstrip("-").isdigit() else item.gold
        if answer_contains(answer, expected):
            reasons.append(f"F8 check passed: the answer contains the correct result {item.gold!r}")
        else:
            codes.append("F8")
            reasons.append(f"F8: the answer does not contain the correct result {item.gold!r}")
    return codes, reasons


def load_labels(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load labels. With a path, read that file only. Without one, merge every *.jsonl in
    bench/labels/, where a human label always beats an automatic or model-made one."""
    files = [path] if path is not None else sorted(LABEL_DIR.glob("*.jsonl"))
    labels: dict[str, dict[str, Any]] = {}
    for file in files:
        if not file.is_file():
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("source", "human")
            current = labels.get(item["key"])
            if current is None or current["source"] not in HUMAN_SOURCES:
                labels[item["key"]] = item
    return labels


def save_labels(labels: dict[str, dict[str, Any]], path: Path = LABELS) -> None:
    """Write the labels that belong in your own file (human and --auto ones)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    own = [v for v in labels.values() if v.get("source", "human") in HUMAN_SOURCES | {"auto"}]
    lines = [json.dumps(v, ensure_ascii=False) for v in own]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_answer(text: str) -> list[str] | str | None:
    """Return a list of codes ([] for no failure), a command ('s', 'u', 'q', '?', ''),
    or None if the input is not understood. '' means: accept the suggestion."""
    text = text.strip().lower()
    if text in ("s", "u", "q", "?", ""):
        return text
    if text == "n":
        return []
    codes = [c.strip().upper() for c in text.replace(" ", ",").split(",") if c.strip()]
    if codes and all(c in CODES for c in codes):
        return sorted(set(codes))
    return None


def latest_runs(folder: Path = ROOT / "bench" / "results" / "phase2") -> list[Path]:
    """The newest run file per provider+model that saved full answers."""
    newest: dict[str, Path] = {}
    for path in sorted(folder.glob("*.json")):  # names end in a sortable UTC timestamp
        run = json.loads(path.read_text(encoding="utf-8"))
        rows = run.get("rows") or []
        if not rows or not all("answer_full" in r or "error" in r for r in rows):
            continue
        prefix = path.stem.rsplit("__", 1)[0]
        if run.get("types_filter") is None and len(rows) >= 50:
            newest[prefix] = path
    return sorted(newest.values())


def notes(item: SeedItem) -> list[str]:
    out = [
        f"built to trigger: {item.failure_type.value} ({item.failure_type.label})",
        f"why: {item.why}",
    ]
    for name in ("gold", "constraints", "readings", "expected"):
        value = getattr(item, name)
        if value:
            out.append(f"{name}: {value}")
    if item.gold_context:
        out.append(f"hidden context (the model did NOT see it): {item.gold_context}")
    if item.control:
        out.append("control item: the premise is true; it should not be flagged")
    return out


def show(row: dict[str, Any], item: SeedItem, model: str, position: str) -> None:
    width = 100
    print("\n" + "=" * width)
    print(f"{position}  {row['id']}  model={model}  finish={row.get('finish_reason')}")
    for line in notes(item):
        print(textwrap.fill(line, width, subsequent_indent="    "))
    print("-" * width + "\nPROMPT:")
    for m in item.messages:
        print(textwrap.fill(f"[{m.role}] {m.text}", width, subsequent_indent="    "))
    print("-" * width + "\nANSWER:")
    answer = row.get("answer_full")
    if answer is None:
        print("(answer not saved; re-run bench/run_seed.py)")
    elif not answer.strip():
        thought = row.get("reasoning_chars") or 0
        print(
            "(EMPTY: the model returned no visible answer"
            + (f"; it wrote {thought} characters of hidden reasoning first)" if thought else ")")
        )
    else:
        print(answer)
    print("-" * width)


def review(count: int, seed_value: int = 0) -> None:
    """Spot-check labels you did not make yourself (Claude reference labels): a fixed random
    sample is shown with the proposed label; Enter agrees, or type the right label. Your
    answer is saved as a human label, and the agreement rate goes to review_summary.json."""
    import random

    seed = {i.id: i for i in load_seed()}
    labels = load_labels()
    candidates = sorted(k for k, v in labels.items() if v["source"] not in HUMAN_SOURCES | {"auto"})
    if not candidates:
        sys.exit("nothing to review: every label is already yours or automatic")
    sample = random.Random(seed_value).sample(candidates, min(count, len(candidates)))
    runs: dict[str, dict[str, Any]] = {}
    agreed = checked = 0
    for n, key in enumerate(sample, 1):
        item = labels[key]
        if item["run"] not in runs:
            path = ROOT / "bench" / "results" / "phase2" / f"{item['run']}.json"
            runs[item["run"]] = json.loads(path.read_text(encoding="utf-8"))
        row = next(r for r in runs[item["run"]]["rows"] if r["id"] == item["id"])
        show(row, seed[item["id"]], item["model"], f"[review {n}/{len(sample)}]")
        proposed = item["labels"]
        print(
            f"proposed by {item['source']}: {','.join(proposed) or 'n'}"
            + (f"  ({item['note']})" if item.get("note") else "")
        )
        while True:
            answer = parse_answer(input("Enter = agree, or type the right label > "))
            if answer == "":
                answer = proposed
            if isinstance(answer, list):
                break
            if answer == "q":
                break
            print(HELP)
        if answer == "q":
            break
        checked += 1
        agreed += sorted(answer) == sorted(proposed)
        own = load_labels(LABELS)
        own[key] = {**item, "labels": answer, "source": "human-review", "proposed": proposed}
        save_labels(own)
    if checked:
        summary = {"checked": checked, "agreed": agreed, "agreement": round(agreed / checked, 3)}
        (LABEL_DIR / "review_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\nYou agreed with {agreed} of {checked} labels ({agreed / checked:.0%}).")


def main() -> None:
    if "--review" in sys.argv:
        position = sys.argv.index("--review")
        count = int(sys.argv[position + 1]) if len(sys.argv) > position + 1 else 15
        review(count)
        return
    args = [a for a in sys.argv[1:] if a != "--auto"]
    auto = "--auto" in sys.argv[1:]
    files = [Path(p) for p in args] or latest_runs()
    if not files:
        sys.exit(__doc__)
    print("Labelling: " + ", ".join(p.name for p in files))
    seed = {i.id: i for i in load_seed()}
    labels = load_labels()
    queue: list[tuple[str, dict[str, Any], str, str]] = []
    for path in files:
        run = json.loads(path.read_text(encoding="utf-8"))
        for row in run["rows"]:
            key = f"{path.stem}:{row['id']}"
            if "error" in row or key in labels:
                continue
            if "answer_full" not in row:
                print(f"{path.name}: answers were not saved in full; re-run bench/run_seed.py")
                break
            queue.append((key, row, run["model"], path.stem))
    if auto:
        remaining = []
        for key, row, model, run_name in queue:
            item = seed[row["id"]]
            if item.failure_type in OBJECTIVE:
                codes, _ = suggest(row, item)
                labels[key] = {
                    "key": key,
                    "run": run_name,
                    "id": row["id"],
                    "model": model,
                    "labels": codes,
                    "source": "auto",
                }
            else:
                remaining.append((key, row, model, run_name))
        save_labels(labels)
        print(f"--auto: labelled {len(queue) - len(remaining)} objective prompts (F1, F2, F8).")
        queue = remaining
    print(f"{len(labels)} labelled so far, {len(queue)} to go.\n{legend()}\n{HELP}")
    history: list[str] = []
    n = 0
    while n < len(queue):
        key, row, model, run_name = queue[n]
        item = seed[row["id"]]
        show(row, item, model, f"[{n + 1}/{len(queue)}]")
        suggested, reasons = suggest(row, item)
        for reason in reasons:
            print(f"suggested by objective checks -> {reason}")
        prompt = (
            f"failures? [Enter = {','.join(suggested) or 'n'}] > " if reasons else "failures? > "
        )
        typed = input(prompt)
        answer = parse_answer(typed)
        if answer == "" and not reasons:
            answer = None  # nothing to accept; ask again rather than guess
        elif answer == "":
            answer = suggested
        if answer is None:
            if typed.strip().lower().startswith(("python", "git", "pip", "cd ")):
                print("That looks like a terminal command. This prompt only takes labels;")
                print("type q to save and quit, then run the command.")
            print(HELP)
            continue
        if answer == "?":
            print(legend())
            continue
        if answer == "q":
            break
        if answer == "u":
            if history:
                labels.pop(history.pop(), None)
                save_labels(labels)
                n = max(0, n - 1)
            continue
        if answer == "s":
            n += 1
            continue
        labels[key] = {
            "key": key,
            "run": run_name,
            "id": row["id"],
            "model": model,
            "labels": answer,
            "source": "human",
        }
        history.append(key)
        save_labels(labels)
        n += 1
    print(f"\nSaved {len(labels)} labels to {LABELS.relative_to(ROOT)}")
    if n >= len(queue) and labels:
        reply = input("All done. Score the judges now with bench/eval_judges.py? [y/N] > ")
        if reply.strip().lower() in ("y", "yes"):
            from bench import eval_judges

            sys.argv = [sys.argv[0]]
            eval_judges.main()


if __name__ == "__main__":
    main()
