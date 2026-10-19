# Command line

| Command | Does |
| --- | --- |
| `counterfix setup [--test]` | which keys, local models and judge will be used (values never printed) |
| `counterfix lint FILE... [--judge] [--fail-on-warning]` | prompt linter |
| `counterfix diagnose LOG [--judge] [--html P] [--json P] [--fail-on-failure]` | report on a log of calls |
| `counterfix fix LOG [--judge] [--caller SPEC] [--out P]` | verified fixes for the failed calls in a log |
| `counterfix export-tests FIXES [--out P]` | turn verified fixes into pytest cases |
| `counterfix profile SPEC [--judge] [--fix] [--cost X]` | measure a model on the probe suite |
| `counterfix compare A B` / `compare LOG --model SPEC` | two profiles, or a model upgrade on your calls |
| `counterfix drift SPEC [--threshold 0.15]` | re-run the probes, exit 1 on a shift |
| `counterfix route PROFILE... [--max-failure 0.2]` | cheapest model per task type |
| `counterfix mcp` | MCP server on stdio |

`SPEC` is `provider:model`, for example `groq:qwen/qwen3.8-27b`, `gemini:gemini-3.5-flash-lite`
or `ollama:qwen2.5:3b`. Keys come from the environment or a `.env` file in the current
folder. `--judge` alone picks a judge from your keys; `--judge SPEC` names one.
