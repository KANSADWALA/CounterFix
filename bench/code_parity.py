"""Does counterfix's own code checker agree with EvalPlus? (E4 "done when")

For every HumanEval+ problem it scores three answers: the reference solution, a stub
that returns None, and a solution cut after its first line. Each answer is scored twice:
by counterfix's ``tests`` metric (the original HumanEval tests, in counterfix's sandbox)
and by EvalPlus's own base tests (``evalplus.evaluate.check_correctness``). Every
disagreement is listed. The ``humaneval+`` / ``mbpp+`` benchmarks score with EvalPlus
itself, so their pass@1 is EvalPlus's by construction; this checks counterfix's checker.

    python bench/code_parity.py          # needs counterfix[code]; runs code (subprocess)

Writes bench/results/code_parity.md.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("COUNTERFIX_ALLOW_CODE", "1")
os.environ.setdefault("COUNTERFIX_SANDBOX", "subprocess")

from counterfix.benchmarks.evalplus import EvalPlusTests, base_tests, truth
from counterfix.core.record import CallRecord
from counterfix.detect.code import check_code

OUT = Path(__file__).resolve().parent / "results" / "code_parity.md"


def answers(p: dict) -> dict[str, str]:
    first = p["canonical_solution"].strip().splitlines()[0]
    return {
        "reference solution": f"```python\n{p['prompt']}{p['canonical_solution']}\n```",
        "returns None": "    return None\n",
        "first line only": f"```python\n{p['prompt']}    {first}\n```",
    }


def main() -> int:
    probs, _ = truth("humaneval")
    evalplus_base = EvalPlusTests(plus=False)
    rows: list[tuple[str, str, bool, bool]] = []
    start = time.time()
    for n, (task_id, p) in enumerate(probs.items(), 1):
        for kind, text in answers(p).items():
            ours = check_code(text, base_tests(p), timeout=10).passed
            ref = {"task_id": task_id, "dataset": "humaneval"}
            theirs = bool(evalplus_base.score(CallRecord(response_text=text), ref).passed)
            rows.append((task_id, kind, ours, theirs))
        print(f"\r{n}/{len(probs)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    same = sum(1 for *_, a, b in rows if a == b)
    diff = [r for r in rows if r[2] != r[3]]
    lines = [
        "# counterfix code checker vs EvalPlus base tests (HumanEval+)",
        "",
        f"{len(probs)} problems x 3 answers = {len(rows)} verdicts; agree on {same} "
        f"({same / len(rows):.1%}); {time.time() - start:.0f}s.",
        "",
        "| answer | counterfix pass | EvalPlus pass |",
        "| --- | --- | --- |",
    ]
    for kind in ("reference solution", "returns None", "first line only"):
        sel = [r for r in rows if r[1] == kind]
        mine, evalplus = sum(r[2] for r in sel), sum(r[3] for r in sel)
        lines.append(f"| {kind} | {mine}/{len(sel)} | {evalplus}/{len(sel)} |")
    lines += ["", "## Disagreements", ""]
    lines += [f"- {t} ({k}): counterfix {a}, EvalPlus {b}" for t, k, a, b in diff] or ["None."]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
