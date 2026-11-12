# Migrating to 0.3

Nothing that worked in 0.2 stops working in 0.3. Three things change how it looks.

## 1. A smaller top level

`import counterfix` now lists 12 names: `guard`, `watch`, `diagnose`, `fix`, `configure`,
`Config`, `Diagnosis`, `FixResult`, `FailureType`, `CallRecord`, `Message` and `log_to`.
Every other 0.2 name (`counterfix.evaluate`, `counterfix.RuleStore`,
`counterfix.litellm_callback`, ...) still imports from `counterfix`, without a warning; it
loads on first use, so `import counterfix` is faster.

Common settings no longer need a `Config`:

```python
client = counterfix.guard(OpenAI(), log="calls.jsonl", sample_rate=0.05, mode="retry")
```

## 2. New module paths

Every module now sits in a layered package. The old paths still import, giving the same
module objects (so `isinstance` and monkeypatching behave as before), with a
`DeprecationWarning`; they are planned to go in 0.4. Type checkers such as mypy do not
follow the old paths, so typed code should switch now.

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
| `counterfix.perplexity` | `counterfix.evals.perplexity` |
| `counterfix.presets` | `counterfix.core.presets` |
| `counterfix.profiles` | `counterfix.fixes.profiles` |
| `counterfix.rag` | `counterfix.detect.retrieval` |
| `counterfix.report` | `counterfix.runtime.report` |
| `counterfix.rules` | `counterfix.fixes` |
| `counterfix.rules.store` | `counterfix.core.rules` |
| `counterfix.safe_eval` | `counterfix._internal.safe_eval` |
| `counterfix.safety` | `counterfix.evals.safety` |
| `counterfix.text` | `counterfix._internal.text` |
| `counterfix.verify` | `counterfix.fixes.verify` |

`counterfix.context()`, `counterfix.otel`, `counterfix.perf`, `counterfix.background` and
`counterfix.metrics` keep working as short names, without a warning.

## 3. Evaluation features that other tools already provide

counterfix explains why a call failed and verifies fixes; it works alongside evaluation
tools rather than replacing them. These still work in 0.3 but warn (`FutureWarning`) and
are planned for removal in 0.4:

| Deprecated | Use instead |
| --- | --- |
| metrics `bleu`, `chrf`, `rouge_l` | sacrebleu, rouge-score, or your evaluation tool |
| metrics `bertscore`, `embedding_similarity` | bert-score, sentence-transformers, or your evaluation tool |
| metrics `faithfulness`, `answer_relevancy`, `answer_correctness`, `context_precision`, `context_recall`, `citations` | Ragas or DeepEval |
| metric `pii`, `counterfix.evals.safety.find_pii` | Microsoft Presidio |
| metric `toxicity`, `counterfix.evals.safety.toxicity_scores` | Detoxify |
| benchmarks `mmlu`, `gsm8k`, `arc`, `truthfulqa`, `ifeval`, metric `ifeval` | lm-evaluation-harness |
| benchmarks `humaneval`, `humaneval+`, `mbpp`, `mbpp+`, metric `evalplus` | EvalPlus |
| `counterfix.evals.perplexity` | lm-evaluation-harness |

Staying: the simple checks (`exact_match`, `contains`, `f1`, `number`, `choice`,
`json_valid`, `json_match`, `latency`, `ttft`), the label-based retrieval metrics, `tests`
(F12), `tool_call` (F14), `injection` (F13), `confidence`, and the `injection` and `tools`
probe benchmarks.

To see every warning your code triggers, run it once with `python -W error::FutureWarning
-W error::DeprecationWarning`.
