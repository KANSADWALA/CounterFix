# Benchmarks

Run a standard benchmark on any model counterfix can call (Groq, Gemini, OpenAI, Anthropic,
Ollama, any OpenAI-compatible URL) or on your own app, and get the score with a 95%
confidence interval, the misses grouped by cause, and verified fixes for them.

```bash
pip install "counterfix[benchmarks]"           # + "counterfix[code]" for HumanEval+ / MBPP+
counterfix benchmark --list
counterfix benchmark gsm8k --model ollama:qwen2.5:3b --limit 200
counterfix benchmark mmlu --model groq:openai/gpt-oss-120b --per-group 10 --fix 5
counterfix benchmark humaneval+ --model groq:openai/gpt-oss-120b --allow-code
```

| Benchmark | What is scored | Data (licence) |
| --- | --- | --- |
| `mmlu` | option letter, 57 subjects | cais/mmlu, test (MIT) |
| `gsm8k` | final number | openai/gsm8k, test (MIT) |
| `arc` | option letter, ARC-Challenge | allenai/ai2_arc, test (CC-BY-SA-4.0) |
| `truthfulqa` | option letter, MC1 (one true answer; options shuffled with the seed) | truthfulqa/truthful_qa, validation (Apache-2.0) |
| `ifeval` | strict prompt-level accuracy, Google's official checks (via lm-eval) | google/IFEval (Apache-2.0) |
| `tools` | right tool and arguments, or no tool when none is needed: 20 probes ([Agents](agents.md)) | written for counterfix (MIT) |
| `injection` | probes resisted: 20 planted instructions in passages and tool results ([Safety](safety.md)) | written for counterfix (MIT) |
| `humaneval` | original HumanEval tests, counterfix's checker (any OS) | HumanEval (MIT), via EvalPlus's data |
| `mbpp` | original MBPP asserts, counterfix's checker (any OS) | MBPP (CC-BY-4.0), via EvalPlus's data |
| `humaneval+` | EvalPlus base + plus tests | EvalPlus (Apache-2.0); HumanEval (MIT) |
| `mbpp+` | EvalPlus base + plus tests | EvalPlus (Apache-2.0); MBPP (CC-BY-4.0) |

Datasets are downloaded when you run them and never shipped with counterfix.

EvalPlus's scorer (`humaneval+`, `mbpp+`) uses Unix signals and resource limits, so it
runs on Linux and macOS only; on Windows use WSL, or `humaneval` / `mbpp`, which score the
same problems against their original tests with counterfix's checker (it agrees with
EvalPlus's base tests on 491 of 492 verdicts).

## Subsets that fit a free tier

`--per-group 20` takes 20 questions per group (per MMLU subject: 1,140 in all) and
`--limit N` takes N in all, both drawn with `--seed` (default 0), so the same command picks
the same questions every time. The confidence interval in the report says how far a
subset can be from the full set; the weakest groups are listed. `--out answers.jsonl`
keeps every answer, so a run stopped by a daily limit resumes where it stopped.

## How the questions are asked

Chat models answer in text, so every benchmark is generative: the prompt ends with
"Answer: X" (letters) or "Answer: <number>" (GSM8K), and counterfix reads the last explicit
answer. Temperature is 0 and the prompt template is versioned (`template v1` in the
report), so results stay comparable across counterfix versions. The answer budget is
4,096 tokens, because reasoning models (gpt-oss, Qwen3, DeepSeek-R1) spend much of it
thinking: at 1,536 tokens gpt-oss-120b ran out on 2 of 20 HumanEval problems. Set another
with `--param max_tokens=8192`. Models that answer briefly stop early and use less.

The classic MMLU number is log-likelihood scoring, which needs token probabilities chat
APIs do not return. Generative scores are usually close but not identical; for
log-likelihood scoring of local models, use lm-evaluation-harness directly.

## From Python

```python
from counterfix.evals.benchmarks import run_benchmark

result = run_benchmark("gsm8k", model="ollama:qwen2.5:3b", limit=200, fix=5)
print(result.text())
result.info  # source, licence, template, seed, subset, per-group accuracy
```

## Checking the numbers

`bench/lm_eval_parity.py --model ollama:qwen2.5:3b --limit 100` runs GSM8K and MMLU
through counterfix and through lm-evaluation-harness (`local-chat-completions`, task
files with counterfix's prompts) on the same first 100 items. Measured on qwen2.5:3b:
GSM8K 0.81 vs 0.79 (94/100 items scored the same), MMLU abstract algebra 0.56 vs 0.56
(100/100); see [Measured results](results.md#benchmarks-against-lm-evaluation-harness).
counterfix compares numbers, so "75.00" counts as 75; lm-eval compares text.
