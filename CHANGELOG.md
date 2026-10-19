# Changelog

All notable changes to counterfix. Versions follow [semantic versioning](https://semver.org).

## 0.1.0 (2026-10-19)

First real release.

- `guard(client)` for any OpenAI-compatible client: live checks on every call with no extra
  LLM calls (F1 truncation incl. reasoning budgets, F2 format and tool-call arguments,
  F3 refusal, F6 exact instruction checks, empty output, provider filter). Fail-open.
- `Config(mode="retry")`: one automatic retry with a known fix, keeping the better response.
- `counterfix.diagnose()`: live checks plus judge checks (F3, F4, F5, F6, F8, F9), one cause
  per mistake; `judge=False` for live checks only; `Diagnosis.explain()` for readable output.
- `@counterfix.watch`: live checks on any function that calls a model (sync or async).
- `guard(Anthropic())`: the Anthropic SDK's `messages.create` (sync and async, tool_use
  arguments checked, thinking kept as reasoning; retries change only `max_tokens`).
  `stop_reason="refusal"` counts as a provider filter, not F3.
- `litellm_callback()` and `langchain_callback()`: observe-only live checks for apps already
  on LiteLLM or LangChain.
- `counterfix.fix()`: the counterfactual engine. Tries one change per cause, re-runs the
  call, and returns a fix only if it passes 2 of 3 re-runs; a plain retry runs first, and
  `max_runs` caps the cost. Unverified causes get an escalation list.
- Streams: `stream=True` calls are checked when the stream ends, chunks pass through.
- `counterfix.context(source=..., reference=...)`: the RAG hook for every plug.
- Prompt regression tests: `counterfix fix LOG --out`, `counterfix export-tests`, a pytest
  plugin (`--counterfix-judge`, `--counterfix-runs`, `counterfix_caller` fixture) and
  `counterfix.testing.assert_ok()`.
- Model profiles: `counterfix profile`, `compare` (profiles, or a log re-run on a new
  model), `drift`, `route`; `fix()` orders interventions by the model's profile; measured
  profiles ship with the package.
- OpenTelemetry: `counterfix.otel.export()` and `counterfix.otel.SpanChecker()`.
- MCP server: `counterfix mcp` (`pip install "counterfix[mcp]"`).
- Docs site (`mkdocs.yml`, `docs/`) and a Colab notebook (`examples/quickstart.ipynb`).
- `fix()` stops after the first failed call when the model cannot be reached and says so
  (`result.unreachable`) instead of advising untested changes. Daily quota limits fail
  fast with `DailyLimitError`.
- Regression tests: `--counterfix-caller SPEC` / `COUNTERFIX_CALLER` (including
  `MODEL@http://host/v1` for your own server); cases that cannot run fail in CI.
- Bring your own judge key: `configure()`, `COUNTERFIX_JUDGE`, detected Groq / Gemini /
  OpenAI / OpenRouter keys or local Ollama; voting panels.
- `counterfix setup [--test]` and `counterfix lint` (rules L1 to L7).
- Reports: `log_to(path)` logs every guarded call to JSONL; `counterfix diagnose LOG`
  prints a terminal summary and can write `--html` (one shareable file) and `--json`.
- Background judge checks: `Config(sample_rate=..., max_judged_per_hour=..., judge=...,
  on_judged=...)` judges flagged calls and a sample of the rest off the request path.
  `Config.judge_model` (never used) is replaced by `Config.judge`.
- Examples: `examples/quickstart.py` (no key), `examples/diagnose_with_judge.py`.

## 0.0.1

Name reserved on PyPI.
