# Command line

| Command | Does |
| --- | --- |
| `counterfix setup [--test]` | which keys, local models and judge will be used (values never printed) |
| `counterfix lint FILE... [--judge] [--fail-on-warning]` | prompt linter |
| `counterfix diagnose LOG [--judge] [--html P] [--json P] [--fail-on-failure]` | report on a log of calls, or on a DeepEval, Ragas or Promptfoo results file ([guide](eval-tools.md)) |
| `counterfix fix LOG [--judge] [--caller SPEC] [--check FILE:FN] [--system TEXT] [--threshold M=V] [--max-runs N] [--out P]` | verified fixes for the failed calls in a log or an evaluation tool's results; `--check` uses your own test ([guide](your-check.md), [tools](eval-tools.md)) |
| `counterfix eval DATA [--model SPEC] [--app MODULE:FN] [--metric M]... [--concurrency N] [--samples N] [--allow-code] [--sandbox docker] [--judge] [--fix N] [--out P] [--fail-under M=V] [--fail-on-flagged]` | run and score a dataset, misses by cause, verified fixes ([guide](evals.md)) |
| `counterfix benchmark NAME --model SPEC [--limit N] [--per-group N] [--seed S] [--fix N]` | the prompt-injection and tool-use probe sets, with misses by cause ([guide](benchmarks.md)); `--list` shows them |
| `counterfix export-tests FIXES [--out P]` | turn verified fixes into pytest cases |
| `counterfix profile SPEC [--judge] [--fix] [--cost X]` | measure a model on the probe suite |
| `counterfix compare A B` / `compare LOG --model SPEC` | two profiles, or a model upgrade on your calls |
| `counterfix drift SPEC [--threshold 0.15]` | re-run the probes, exit 1 on a shift |
| `counterfix route PROFILE... [--max-failure 0.2]` | cheapest model per task type |
| `counterfix mcp` | MCP server on stdio |

`SPEC` is `provider:model`, for example `groq:qwen/qwen3.8-27b`, `gemini:gemini-3.5-flash-lite`
or `ollama:qwen2.5:3b`; also `anthropic:MODEL` or `MODEL@http://host/v1` for any
OpenAI-compatible server. Keys come from the environment or a `.env` file in the current
folder. `--judge` alone picks a judge from your keys; `--judge SPEC` names one.

Exit codes: 0 when the run finished and every gate passed; 1 when a gate failed
(`--fail-under`, `--fail-on-flagged`, `--fail-on-failure`, `--fail-on-warning`, a drift);
2 when the run could not be done (a bad argument, a missing file, or every call failed).

