# Probe sets

counterfix ships two small probe sets for failure types it diagnoses and fixes. Run them on
any model counterfix can call (Groq, Gemini, OpenAI, Anthropic, Ollama, any
OpenAI-compatible URL) or on your own app, and get the score with a 95% confidence
interval, the misses grouped by cause, and verified fixes for them.

```bash
counterfix benchmark --list
counterfix benchmark injection --model ollama:qwen2.5:3b --fix 3
counterfix benchmark tools --model groq:openai/gpt-oss-120b
```

| Probe set | What is scored | Data (licence) |
| --- | --- | --- |
| `injection` | probes resisted: 20 planted instructions in passages and tool results, F13 ([Safety](safety.md)) | written for counterfix (MIT) |
| `tools` | right tool and arguments, or no tool when none is needed: 20 probes, F14 ([Agents](agents.md)) | written for counterfix (MIT) |

For standard benchmarks, use the tools built for them:
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for MMLU,
GSM8K, ARC, TruthfulQA and IFEval, and [EvalPlus](https://github.com/evalplus/evalplus) for
HumanEval(+) and MBPP(+). To find out why a model failed their questions, see
[DeepEval, Ragas and Promptfoo](eval-tools.md) and [Evaluate a dataset](evals.md).

## Options

`--limit N` takes N probes and `--per-group N` takes N per group, both drawn with `--seed`
(default 0), so the same command picks the same probes every time. Temperature is 0 and the
prompt template is versioned (`template v1` in the report), so results stay comparable
across counterfix versions. The answer budget is 4,096 tokens, because reasoning models
spend much of it thinking; set another with `--param max_tokens=8192`. `--out answers.jsonl`
keeps every answer, so a run stopped by a daily limit resumes where it stopped.

## From Python

```python
from counterfix.evals.benchmarks import run_benchmark

result = run_benchmark("injection", model="ollama:qwen2.5:3b", fix=3)
print(result.text())
result.info  # source, licence, template, seed, subset, per-group accuracy
```
