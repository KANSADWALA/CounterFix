"""Do counterfix's fixes beat what a developer would try anyway? (build plan, Phase 5 baselines)

One model answers IFEval prompts; the answers that break an instruction (IFEval's own
strict checks, Google's code via lm-eval) are the failures. Each failure then gets five
treatments, and each treatment is scored on ``--runs`` fresh calls that were not used to
find it, by the same IFEval checks, which are independent of counterfix's detectors:

- ``retry``          the same request again
- ``careful``        a generic system prompt: be careful, follow every instruction
- ``self_critique``  the model sees its failed answer and is asked to check and rewrite it
- ``counterfix``     ``counterfix.fix(record, judge=False)``: live checks only, as a
                     developer without a judge key would run it; when it finds no fix,
                     the request is sent unchanged
- ``counterfix_eval`` the fix search of ``counterfix eval``/``benchmark``, driven by the
                     IFEval check itself (the most generous setting for counterfix)

Per treatment: the mean pass rate over failures, the paired difference from ``retry``
with a 95% bootstrap interval, wins/ties/losses against ``retry``, and the calls it took
to find the change. A fix is tested only on the prompt it was found on: whether it helps
other prompts is not measured here.

    python bench/compare_baselines.py --model ollama:qwen2.5:3b
    python bench/compare_baselines.py --model ollama:qwen2.5:3b --limit 200 --max-failures 40
    python bench/compare_baselines.py --model ollama:qwen2.5:3b --report   # rebuild the report

Resumable: answers and finished failures are kept next to the report; Ctrl+C keeps what is
done. Needs ``counterfix[benchmarks]`` (datasets, lm-eval[ifeval]). Writes
bench/results/compare_<model>.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from counterfix import CallRecord, Message, diagnose, fix  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.core.presets import caller_for, load_env_file, pace_for  # noqa: E402
from counterfix.evals.benchmarks import get_benchmark  # noqa: E402
from counterfix.evals.benchmarks.ifeval import IfEvalStrict, score_instructions  # noqa: E402
from counterfix.evals.runner import _with_metric_findings, fix_item, score  # noqa: E402
from counterfix.fixes.engine import _with_system  # noqa: E402

RESULTS = ROOT / "bench" / "results"
CAREFUL = (
    "Be careful and accurate. Read the request again before answering and follow every "
    "instruction in it exactly."
)
CRITIQUE = (
    "Check your answer against every instruction in my request above. If it misses any, "
    "rewrite the whole answer so that it follows all of them. Reply with the answer only."
)
ARMS = ["retry", "careful", "self_critique", "counterfix", "counterfix_eval"]
_lock = threading.Lock()


class Counting:
    """Wraps the model caller to show a live progress line: calls made and their speed."""

    def __init__(self, caller: Any) -> None:
        self.caller, self.name = caller, getattr(caller, "name", "model")
        self.calls, self.errors, self.start = 0, 0, time.monotonic()
        self.note = ""

    def complete(self, messages: Any, **params: Any) -> CallRecord:
        try:
            return self.caller.complete(messages, **params)  # type: ignore[no-any-return]
        except Exception:
            with _lock:
                self.errors += 1
            raise
        finally:
            with _lock:
                self.calls += 1
                rate = self.calls / max(1e-9, time.monotonic() - self.start) * 60
                err = f", {self.errors} errors" if self.errors else ""
                print(f"\r  {self.calls} calls ({rate:.0f}/min{err}){self.note}   ",
                      end="", flush=True)  # fmt: skip


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", spec).strip("_")


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}


def append(path: Path, row: dict[str, Any]) -> None:
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def passes(answer: str | None, reference: dict[str, Any]) -> bool:
    return answer is not None and score_instructions(answer, reference).passed is True


def held_out(
    caller: Any, messages: list[Message], params: dict[str, Any], ref: dict[str, Any], runs: int
) -> dict[str, Any]:
    """Fresh runs of one treatment, scored by IFEval. Calls that error are not evidence.
    Keeps the first failing and first passing answer, so a result can be checked by eye."""
    ok = done = errors = 0
    samples: dict[str, str] = {}
    for _ in range(runs):
        try:
            new = caller.complete(messages, **params)
        except Exception:
            errors += 1
            continue
        done += 1
        good = passes(new.answer_text, ref)
        ok += good
        samples.setdefault("pass" if good else "fail", (new.answer_text or "")[:800])
    return {"passed": ok, "runs": done, "errors": errors, "samples": samples}


def treat(item: dict[str, Any], first: CallRecord, caller: Any, runs: int) -> dict[str, Any]:
    """All five treatments for one failure."""
    ref = item["reference"]
    msgs, params = list(first.messages), dict(first.params)
    out: dict[str, Any] = {"id": item["id"], "missed": first.metadata.get("missed", "")}

    out["retry"] = held_out(caller, msgs, params, ref, runs)
    out["careful"] = held_out(caller, _with_system(msgs, CAREFUL), params, ref, runs)
    critique = [
        *msgs,
        Message(role="assistant", content=first.answer_text),
        Message(role="user", content=CRITIQUE),
    ]
    out["self_critique"] = held_out(caller, critique, params, ref, runs)

    # counterfix with live checks only: what a developer runs without a judge key
    found = fix(first, judge=False, caller=caller)
    change: dict[str, Any] = {"fix": None, "detected": bool(found.diagnosis.model_failures)}
    use_msgs, use_params = msgs, params
    if found.fix is not None:
        change["fix"] = found.fix.name
        use_msgs = found.fix.messages or msgs
        use_params = dict(found.fix.params)
    out["counterfix"] = {
        **held_out(caller, use_msgs, use_params, ref, runs),
        **change,
        "search_calls": found.calls,
        "found": [f.failure_type.value for f in found.diagnosis.model_failures],
    }

    # the fix search of `counterfix eval`, driven by the IFEval check itself
    metrics = [IfEvalStrict()]
    diag = _with_metric_findings(diagnose(first, judge=False), score(first, ref, metrics))
    search = fix_item(first, ref, metrics, diag, caller)
    use_msgs, use_params = msgs, params
    name = None
    if search.fix is not None:
        name = search.fix.name
        use_msgs = search.fix.messages or msgs
        use_params = {**params, **search.fix.changed}
    out["counterfix_eval"] = {
        **held_out(caller, use_msgs, use_params, ref, runs),
        "fix": name,
        "search_calls": search.calls,
    }
    return out


def rate(arm: dict[str, Any]) -> float | None:
    return arm["passed"] / arm["runs"] if arm.get("runs") else None


def bootstrap(diffs: list[float], seed: int = 0, n: int = 4000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def report(
    spec: str,
    settings: dict[str, Any],
    answers: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    asked = len(answers)
    failed = [a for a in answers.values() if not a.get("passed") and "error" not in a]
    lines = [
        f"# Fixes against simple alternatives: {spec}",
        "",
        f"IFEval prompts asked: {asked}; broke at least one instruction: {len(failed)} "
        f"({len(failed) / asked:.0%}); compared below: {len(rows)}. Settings: "
        + ", ".join(f"{k}={v}" for k, v in settings.items())
        + ". Each treatment is scored on fresh runs not used to find it, by IFEval's own "
        "strict checks. A fix is tested on the prompt it was found on only.",
        "",
    ]
    if not rows:
        return "\n".join([*lines, "No failures compared yet."]) + "\n"
    lines += [
        "| Treatment | Failures compared | Mean pass rate | vs retry (95% CI) "
        "| Wins / ties / losses vs retry | Calls to find the change |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    base = [rate(r["retry"]) for r in rows]
    for arm in ARMS:
        search = ""
        if arm.startswith("counterfix"):
            calls = [r[arm].get("search_calls", 0) for r in rows]
            search = f"{sum(calls) / len(calls):.1f} per failure"
        pairs: list[tuple[float, float]] = []
        for r, b in zip(rows, base, strict=True):
            if arm.startswith("counterfix") and not r[arm].get("fix"):
                continue  # no fix: the request went unchanged, a second retry, not a treatment
            a = rate(r[arm])
            if a is not None and b is not None:
                pairs.append((a, b))
        if not pairs:
            lines.append(f"| {arm} | 0 | - | no fix found, nothing to compare | - | "
                         f"{search or 'none'} |")  # fmt: skip
            continue
        mean = sum(a for a, _ in pairs) / len(pairs)
        if arm == "retry":
            lines.append(f"| {arm} | {len(pairs)} | {mean:.2f} | (baseline) | - | none |")
            continue
        diffs = [a - b for a, b in pairs]
        low, high = bootstrap(diffs)
        wins = sum(d > 0 for d in diffs)
        ties = sum(d == 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        diff = sum(diffs) / len(diffs)
        lines.append(
            f"| {arm} | {len(pairs)} | {mean:.2f} | {diff:+.2f} ({low:+.2f} to {high:+.2f}) "
            f"| {wins} / {ties} / {losses} | {search or 'none'} |"
        )
    detected = sum(r["counterfix"].get("detected", False) for r in rows)
    found = sum(r["counterfix"].get("fix") is not None for r in rows)
    found_eval = sum(r["counterfix_eval"].get("fix") is not None for r in rows)
    lines += [
        "",
        f"counterfix (live checks): failure detected on {detected}/{len(rows)}, a verified "
        f"fix found on {found}/{len(rows)}. counterfix_eval (IFEval-driven search): a "
        f"verified fix found on {found_eval}/{len(rows)}. The counterfix rows compare only the "
        "failures where a fix was found and applied: elsewhere the request went unchanged, "
        "which is a second plain retry, not a treatment.",
        "",
        "Read a difference as real only when its interval excludes zero, and treat any row "
        "with few failures compared as anecdote.",
        "",
        "## Per failure",
        "",
        "| Item | Missed | " + " | ".join(ARMS) + " | counterfix fix | counterfix_eval fix |",
        "| --- | --- | " + " | ".join("---" for _ in ARMS) + " | --- | --- |",
    ]
    for r in rows:
        cells = [f"{r[a]['passed']}/{r[a]['runs']}" for a in ARMS]
        cells += [r["counterfix"].get("fix") or "-", r["counterfix_eval"].get("fix") or "-"]
        lines.append(f"| {r['id']} | {r.get('missed', '')[:60]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, make_caller: Callable[[str], Any] | None = None) -> int:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="e.g. ollama:qwen2.5:3b")
    parser.add_argument("--limit", type=int, default=150, help="IFEval prompts to ask")
    parser.add_argument("--max-failures", type=int, default=30, help="failures to compare")
    parser.add_argument("--runs", type=int, default=5, help="fresh runs per treatment")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=1, help="failures treated at once")
    parser.add_argument("--seed", type=int, default=0, help="which prompts, in a fixed order")
    parser.add_argument("--report", action="store_true", help="rebuild the report only")
    parser.add_argument("--out", type=Path, default=RESULTS, help="results folder")
    args = parser.parse_args(argv)

    base = f"compare_{slug(args.model)}"  # e.g. compare_ollama_qwen2.5_3b (no suffix games)
    answers_path, rows_path = args.out / f"{base}.answers.jsonl", args.out / f"{base}.jsonl"
    report_path = args.out / f"{base}.md"
    settings = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "runs per treatment": args.runs,
        "seed": args.seed,
    }
    answers, done = load_jsonl(answers_path), load_jsonl(rows_path)
    if args.report:
        rows = [done[k] for k in done]
        report_path.write_text(report(args.model, settings, answers, rows), "utf-8")
        print(f"Saved {report_path}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    caller = Counting(
        (make_caller or (lambda s: RateLimitedCaller(caller_for(s, timeout=180),
                                                     pace_for(s))))(args.model)
    )  # fmt: skip
    items = get_benchmark("ifeval").load(args.seed)
    random.Random(args.seed).shuffle(items)
    items = items[: args.limit]
    params = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    # 1. one answer per prompt; keep the ones that break an instruction
    todo = [i for i in items if i["id"] not in answers]
    print(f"{len(items)} prompts, {len(todo)} to ask", flush=True)
    asked = [0]

    def ask(item: dict[str, Any]) -> None:
        try:
            rec = caller.complete([Message(role="user", content=item["prompt"])], **params)
            verdict = score_instructions(rec.answer_text, item["reference"])
            row = {"id": item["id"], "passed": verdict.passed is True, "missed": verdict.reason,
                   "record": rec.model_dump(mode="json")}  # fmt: skip
        except Exception as exc:
            row = {"id": item["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
        with _lock:
            answers[item["id"]] = row
            asked[0] += 1
        append(answers_path, row)
        caller.note = f", answered {asked[0]}/{len(todo)}"

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(ask, todo))
    except KeyboardInterrupt:
        print("\nStopped; what is done is kept. Run the same command to continue.")
        return 1
    print(flush=True)

    # 2. the treatments, on the first --max-failures failures
    by_id = {i["id"]: i for i in items}
    failures = [
        answers[i["id"]] for i in items
        if i["id"] in answers and not answers[i["id"]].get("passed")
        and "error" not in answers[i["id"]]
    ][: args.max_failures]  # fmt: skip
    left = [a for a in failures if a["id"] not in done]
    print(
        f"{len(failures)} failures to compare, {len(left)} left; each takes about 70-90 "
        "calls, and a line is printed when one finishes",
        flush=True,
    )
    caller.note = ""

    def one(a: dict[str, Any]) -> None:
        first = CallRecord.model_validate(a["record"])
        first.params = {**params, **first.params}  # fix() re-runs with these
        first.metadata["missed"] = a.get("missed", "")
        try:
            row = treat(by_id[a["id"]], first, caller, args.runs)
        except Exception as exc:
            print(f"  {a['id']}: {type(exc).__name__}: {exc}"[:300], flush=True)
            return
        done[a["id"]] = row
        append(rows_path, row)
        summary = ", ".join(f"{k} {row[k]['passed']}/{row[k]['runs']}" for k in ARMS)
        print(f"\n[{len(done)}/{len(failures)}] {a['id']}: {summary}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(one, left))
    except KeyboardInterrupt:
        print("\nStopped; what is done is kept. Run the same command to continue.")
    rows = [done[a["id"]] for a in failures if a["id"] in done]
    report_path.write_text(report(args.model, settings, answers, rows), "utf-8")
    print(f"Saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
