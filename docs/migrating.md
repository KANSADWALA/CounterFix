# Upgrading to 0.4

0.4 removes what 0.3 had marked for removal. If your code ran on 0.3 without warnings, it
runs on 0.4 unchanged.

## Old module paths are gone

0.3 moved every module into a layered package and kept the 0.2 paths working with a
warning. In 0.4 the old paths no longer import; use the new ones:

| Old path | New path |
| --- | --- |
| `counterfix.adapters` | `counterfix.integrations` |
| `counterfix.agents` | `counterfix.detect.agents` |
| `counterfix.background` | `counterfix.runtime.background` |
| `counterfix.benchmarks` | `counterfix.evals.benchmarks` |
| `counterfix.capabilities` | `counterfix.integrations.capabilities` |
| `counterfix.code` | `counterfix.detect.code` |
| `counterfix.confidence` | `counterfix.detect.confidence` |
| `counterfix.context` | `counterfix.runtime.context` |
| `counterfix.detectors` | `counterfix.detect` |
| `counterfix.engine` | `counterfix.fixes.engine` |
| `counterfix.mcp_server` | `counterfix.integrations.mcp_server` |
| `counterfix.metrics` | `counterfix.evals.metrics` |
| `counterfix.otel` | `counterfix.integrations.otel` |
| `counterfix.perf` | `counterfix.runtime.perf` |
| `counterfix.presets` | `counterfix.core.presets` |
| `counterfix.profiles` | `counterfix.fixes.profiles` |
| `counterfix.rag` | `counterfix.detect.retrieval` |
| `counterfix.report` | `counterfix.runtime.report` |
| `counterfix.rules` | `counterfix.fixes` |
| `counterfix.rules.store` | `counterfix.core.rules` |
| `counterfix.safe_eval` | `counterfix._internal.safe_eval` |
| `counterfix.text` | `counterfix._internal.text` |
| `counterfix.verify` | `counterfix.fixes.verify` |

`counterfix.context()`, `counterfix.evaluate`, `counterfix.otel`, `counterfix.perf`,
`counterfix.background` and `counterfix.metrics` remain as short names.

## Evaluation features other tools already provide are gone

counterfix explains why a call failed and verifies fixes; it works alongside evaluation
tools rather than replacing them. These are removed in 0.4:

| Removed | Use instead |
| --- | --- |
| metrics `bleu`, `chrf`, `rouge_l` | sacrebleu, rouge-score, or your evaluation tool |
| metrics `bertscore`, `embedding_similarity` | bert-score, sentence-transformers, or your evaluation tool |
| metrics `faithfulness`, `answer_relevancy`, `answer_correctness`, `context_precision`, `context_recall`, `citations` | Ragas or DeepEval |
| metric `pii`, `counterfix.evals.safety` | Microsoft Presidio |
| metric `toxicity` | Detoxify |
| benchmarks `mmlu`, `gsm8k`, `arc`, `truthfulqa`, `ifeval`, metric `ifeval` | lm-evaluation-harness |
| benchmarks `humaneval`, `humaneval+`, `mbpp`, `mbpp+`, metric `evalplus` | EvalPlus |
| `counterfix.evals.perplexity` | lm-evaluation-harness |
| extras `metrics`, `semantic`, `safety`, `benchmarks`, `code` | (not needed) |
| the ignored `need=` option of `fix()` and `speed_fix()` | (drop it) |

Staying: the simple checks (`exact_match`, `contains`, `f1`, `number`, `choice`,
`json_valid`, `json_match`, `latency`, `ttft`), the label-based retrieval metrics, `tests`
(F12), `tool_call` (F14), `injection` (F13), `confidence`, and the `injection` and `tools`
probe sets.

To keep a removed tool's verdict in the loop, bring its results file
([DeepEval, Ragas and Promptfoo](eval-tools.md)) or wrap it as your own check
([Fix with your own test](your-check.md)).
