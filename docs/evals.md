# Evaluate a dataset

`counterfix eval` runs a model over a dataset, scores every answer, then does what a plain
eval tool does not: it groups the misses by **cause** (F1 to F14) and, on a sample of
misses, searches for a **verified fix**.

```bash
counterfix eval capitals.jsonl --model groq:openai/gpt-oss-120b --metric exact_match \
  --judge --fix 5 --html report.html
```

```text
eval capitals · groq:openai/gpt-oss-120b · 50 item(s), 9 missed · judged by gemini:...

  metric                        mean  95% CI                  pass
  exact_match                  0.820  0.720 to 0.920         41/50

Misses by cause (9)
  F4 Hallucination from missing knowledge       4
  F1 Truncation                                  2
  none found                                     3

Verified fixes (searched on 5 missed or flagged item(s))
  raise_max_tokens_more                          2
  allow_unknown                                  1
  no verified fix                                2
```

`--fix N` searches up to N items: missed ones first, then flagged ones (see below). A fix
counts only when the item **passes its metrics** with no new failure clearly more often
than on a plain retry ([how](fixes.md#what-verified-means))
(for a flagged item, the flagged failure must also be gone), after a plain retry was tried first (if a plain retry passes, the miss was not
repeatable and no change is claimed).

## From Python

```python
import counterfix

result = counterfix.evaluate(
    "capitals.jsonl",
    model="groq:openai/gpt-oss-120b",  # or any ModelCaller
    metrics=["exact_match", "latency:3000"],
    judge=True,  # diagnose misses with a judge (live checks always run)
    fix=5,  # fix search on up to 5 misses
)
print(result.text())
result.metric("exact_match").pass_rate
```

`model` takes a preset (`groq:...`, `gemini:...`, `ollama:...`, `openai:...`,
`openrouter:...`), `anthropic:MODEL` (Anthropic SDK, `ANTHROPIC_API_KEY`), or
`MODEL@http://host/v1` for any OpenAI-compatible server (vLLM, LM Studio, a LiteLLM proxy;
key from `OPENAI_API_KEY`). To evaluate your own application instead of a bare model, pass
`app=fn` (a function question -> answer; see [RAG evaluation](rag.md#your-own-pipeline)).
`concurrency=8` (`--concurrency 8`) answers 8 items at once; results keep their order and
free tiers stay paced.

In pytest, gate on a metric:

```python
from counterfix.testing import assert_metric

assert_metric(result, "exact_match", ">=", 0.8)
```

## Datasets

JSONL, JSON (a list) or CSV, one item per row.

| What | Taken from |
| --- | --- |
| Prompt | `messages`, else `--template` filled from the row, else `--input FIELD`, else `prompt`, `question` or `input` |
| Reference | `--reference FIELD`, else `reference`, `answer`, `expected`, `target` or `ideal` |
| Recorded answer | `response` or `output`: with no `--model`, these are scored and no model is called |
| Request fields | `--param max_tokens=512` for all items; a row's own `params` object adds to it |

`--out answers.jsonl` keeps each answer as it arrives: a run stopped by a daily limit
resumes where it stopped. A kept answer is reused only if the item's messages and params
are unchanged, so editing a prompt or template re-asks that item. `--limit N` runs the
first N items. Item ids must be unique.

Exit codes: 0 when the run finished and every gate passed, 1 when a `--fail-under` or
`--fail-on-flagged` gate failed, 2 when the run could not be done (a bad argument, a
missing file, or every call failed, e.g. a wrong model name), so CI never passes a run that
got no answers.

## How the numbers are counted

- **Pass** counts items at or past the metric's threshold. A call that failed (timeout,
  server error) counts as a miss, with the cause `INFRA`, never blamed on the model.
- **Not scored** items have no value for that metric (no reference answer, no TTFT on a
  call that was not streamed). They are shown next to the pass count and left out of the
  mean, which is over scored items only.
- **Flagged** items passed every metric, but a free live check still found a model
  failure, such as an answer cut off by `max_tokens` that met a latency gate. They are
  listed by cause so a fast but broken answer cannot pass unnoticed;
  `--fix N` searches verified fixes for them too, `--fail-on-flagged` (or
  `fail_on_flagged: true` in a suite) makes them fail CI, and `result.flagged` lists them
  in Python.
- **95% CI** is a seeded bootstrap of the mean (1,000 resamples), so it repeats run to run.
- If the judge cannot check a miss (an error, a used-up daily limit), the report says how
  many misses were not judged, since their cause may then be missing.

## Suite files

Keep the settings next to the data and run with one command (`.json`, or `.yaml` with
`pip install "counterfix[eval]"`):

```yaml
name: gsm8k-sample
data: gsm8k_50.jsonl          # relative to this file
model: groq:openai/gpt-oss-120b
metrics: [number, latency:5000]
template: "{question}\nGive the final answer after ####."
reference: answer
judge: true
fix: 5
fail_under: {number: 0.7}     # exit code 1 below this (CI)
fail_on_flagged: true         # exit code 1 if a live check flags a passing item
```

```bash
counterfix eval suite.yaml --html report.html
```

## Metrics

Write a metric as `name` or `name:threshold`. Right-or-wrong metrics pass at 1.0 by
default; graded ones report a number and fail nothing unless you give a threshold.

| Metric | Measures | Needs |
| --- | --- | --- |
| `exact_match` | normalised answer equals a reference (SQuAD rules) | core |
| `f1` | word overlap with the best reference (SQuAD F1) | core |
| `contains` | a reference appears in the answer, as whole words | core |
| `rouge_l` | ROUGE-L F-measure; same numbers as Google's `rouge-score` | core |
| `bleu`, `chrf` | sentence BLEU and chrF from `sacrebleu`, scaled 0 to 1 | `[metrics]` |
| `choice` | the option letter chosen ("The answer is (B)") equals the reference | core |
| `number` | the final number (after `####`, in `\boxed{}`, or last) equals the reference | core |
| `json_valid` | the answer is JSON, and matches `schema=` when given; a miss is F2 | core |
| `json_match` | share of the reference's fields the answer gets right; a miss is F2 | core |
| `embedding_similarity` | cosine similarity of sentence embeddings (local model) | `[semantic]` |
| `bertscore` | BERTScore F1 (local model) | `[semantic]` |
| `answer_correctness` | a judge decides whether the answer agrees with the reference in meaning (1, 0.5, 0) | a judge |
| `latency`, `ttft` | milliseconds; pass at or under the threshold, e.g. `latency:2000` | core |
| `hit`, `recall`, `precision`, `mrr`, `ndcg` (and `@k`), `faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`, `citations` | RAG: see [RAG evaluation](rag.md) | labels or a judge |

With several references (a list), a metric takes the best match. A metric that names a
failure type (`json_valid` and `json_match` name F2) adds that cause to a miss, so the
fix search can target it.

Your own metric is a subclass of `counterfix.metrics.Metric` with a `name` and a
`compute(record, reference)` that returns `(value, reason)`; pass the object in
`metrics=[...]`. Set `default_threshold = 1.0` on the class (or pass `threshold=` when you
create it) for a pass/fail metric; without one it is reported as a graded number and
fails nothing:

```python
from counterfix.evals.metrics import Metric


class ShortAnswer(Metric):
    name = "short"
    default_threshold = 1.0  # pass/fail

    def compute(self, record, reference):
        n = len(record.response_text.split())
        return (1.0 if n <= 5 else 0.0), f"{n} words"


counterfix.evaluate("qa.jsonl", model="ollama:qwen2.5:3b", metrics=["contains", ShortAnswer()])
```
