"""Do counterfix's benchmark scores agree with lm-evaluation-harness? (E5 "done when")

Both tools get the same model, the same questions (GSM8K test and MMLU abstract algebra
test, first N), the same prompt text, temperature 0 and the same 1,024-token budget:
lm-eval runs two task files written here with counterfix's prompts, and reads the answer
with its own filters (the last number for GSM8K, the last "Answer: X" letter for MMLU).
So any difference comes from the harness (client, answer reading, scoring), which is
what this checks. Per item, it lists where the two disagree.

An earlier version used lm-eval's stock tasks; their prompts and limits differ so much
that the numbers are not comparable (stock ``mmlu_*_generative`` needs a bare letter and
stops at the first newline: 0/100 for a chat model that explains first; stock
``gsm8k_cot_zeroshot`` caps answers at 256 tokens, which cut 44 of 100 qwen2.5:3b answers).

    python bench/lm_eval_parity.py --model ollama:qwen2.5:3b --limit 100 [--only gsm8k]

Needs counterfix[benchmarks] and lm-eval with its API backend (``pip install
"lm-eval[api]"``), and the model on an OpenAI-compatible URL (Ollama:
http://localhost:11434/v1). Writes bench/results/lm_eval_parity_last.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from counterfix.benchmarks import get_benchmark
from counterfix.core.presets import PRESETS, parse_spec
from counterfix.evals import evaluate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "lm_eval_parity_last.md"  # the curated summary: lm_eval_parity.md
WORK = HERE / "results" / "lm_eval_parity"

# lm-eval task files with counterfix's exact prompts (counterfix.benchmarks, template v1).
TASK_FILES = {
    "gsm8k": """task: cf_gsm8k
dataset_path: openai/gsm8k
dataset_name: main
test_split: test
output_type: generate_until
doc_to_text: "{{question.strip()}}\\n\\nSolve it step by step, then give the final answer on the last line as 'Answer: <number>'."
doc_to_target: "{{answer.split('####')[-1].strip()}}"
generation_kwargs: {until: ["<|im_end|>"], do_sample: false, temperature: 0.0, max_gen_toks: 1024}
filter_list:
  - name: last-number
    filter:
      - {function: regex, group_select: -1, regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"}
      - {function: take_first}
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    regexes_to_ignore: [",", "\\\\$", "\\\\.$"]
metadata: {version: 1.0}
""",
    "mmlu": """task: cf_mmlu_abstract_algebra
dataset_path: cais/mmlu
dataset_name: abstract_algebra
test_split: test
output_type: generate_until
doc_to_text: "The following is a multiple choice question about abstract algebra.\\n\\n{{question.strip()}}\\nA. {{choices[0]}}\\nB. {{choices[1]}}\\nC. {{choices[2]}}\\nD. {{choices[3]}}\\n\\nAnswer with the letter of the correct option on the last line, as 'Answer: X'."
doc_to_target: "{{['A', 'B', 'C', 'D'][answer]}}"
generation_kwargs: {until: ["<|im_end|>"], do_sample: false, temperature: 0.0, max_gen_toks: 1024}
filter_list:
  - name: answer-letter
    filter:
      - {function: regex, group_select: -1, regex_pattern: "[Aa]nswer:?\\\\s*\\\\(?([A-D])\\\\b"}
      - {function: take_first}
metric_list:
  - {metric: exact_match, aggregation: mean, higher_is_better: true}
metadata: {version: 1.0}
""",
}
TASK_NAMES = {"gsm8k": "cf_gsm8k", "mmlu": "cf_mmlu_abstract_algebra"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half


def progress(n: int, total: int, label: str) -> None:
    print(f"\r  counterfix {n}/{total} {label[:40]:<40}", end="", file=sys.stderr, flush=True)


def ours(name: str, model: str, limit: int) -> list[bool]:
    bench = get_benchmark(name)
    rows = bench.load(0)
    if name == "mmlu":  # the subject the lm-eval task runs, in dataset order
        rows = [r for r in rows if r["group"] == "abstract_algebra"]
    rows = rows[:limit]
    keep = OUT.parent / f"lm_eval_parity_{name}_answers.jsonl"  # resumes after Ctrl+C
    print(f"{name}: counterfix on {model} ({limit} items; answers kept in {keep.name})")
    result = evaluate(rows, model=model, metrics=bench.metrics(), input="prompt",
                      reference="reference", params={**bench.params, "max_tokens": 1024}, name=name,
                      out=keep, progress=progress)  # fmt: skip
    print(file=sys.stderr)
    return [i.passed for i in result.items]


def theirs(name: str, model: str, limit: int) -> list[bool]:
    """lm-eval's per-item verdicts (doc order), from its logged samples; a finished run
    is reused."""
    task = TASK_NAMES[name]
    out = WORK / f"{task}_{limit}"
    found = sorted(out.rglob(f"samples_{task}_*.jsonl")) if out.exists() else []
    if not found:
        provider, model_name = parse_spec(model)
        base = (PRESETS[provider][0] or "https://api.openai.com/v1").rstrip("/")
        tasks_dir = WORK / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / f"{task}.yaml").write_text(TASK_FILES[name], encoding="utf-8")
        print(f"{task}: lm-eval on {model_name} ({limit} items; lm-eval shows its progress)")
        cmd = [
            sys.executable, "-m", "lm_eval", "--model", "local-chat-completions",
            "--model_args",
            f"model={model_name},base_url={base}/chat/completions,num_concurrent=1,max_gen_toks=1024",
            "--include_path", str(tasks_dir), "--tasks", task, "--apply_chat_template",
            "--limit", str(limit), "--log_samples", "--output_path", str(out),
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
        found = sorted(out.rglob(f"samples_{task}_*.jsonl"))
    rows = [json.loads(line) for line in found[-1].read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: int(r["doc_id"]))
    return [float(r.get("exact_match", 0)) == 1.0 for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama:qwen2.5:3b")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--only", choices=list(TASK_FILES), help="run one benchmark")
    args = ap.parse_args()
    lines = [
        f"# counterfix vs lm-eval on {args.model} (first {args.limit} items, same prompts)",
        "",
        "| benchmark | counterfix | lm-eval | intervals overlap | items scored the same |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in TASK_FILES:
        if args.only and name != args.only:
            continue
        a, b = ours(name, args.model, args.limit), theirs(name, args.model, args.limit)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        lo, hi = wilson(sum(a), n)
        lo2, hi2 = wilson(sum(b), n)
        same = sum(x == y for x, y in zip(a, b, strict=True))
        overlap = "yes" if lo <= hi2 and lo2 <= hi else "NO"
        lines.append(
            f"| {name} | {sum(a) / n:.3f} ({lo:.2f}-{hi:.2f}) | {sum(b) / n:.3f} "
            f"({lo2:.2f}-{hi2:.2f}) | {overlap} | {same}/{n} |"
        )
        diff = [k for k in range(n) if a[k] != b[k]]
        if diff:
            lines.append(f"|  | differ on items {', '.join(map(str, diff[:20]))} | | | |")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
