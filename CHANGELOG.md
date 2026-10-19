# Changelog

All notable changes to counterfix. Versions follow [semantic versioning](https://semver.org).

## 0.1.2 (2026-10-19)

Documentation only, no code changes.

- Install instructions lead with `pip install counterfix`; the `openai`, `anthropic`,
  `litellm` and `langchain` extras are explained as optional, so the PyPI page no longer
  reads as if counterfix came from a model provider.

## 0.1.1 (2026-10-19)

Fixes from a pre-release code review. No API was removed.

- `fix()` with an async client (`AsyncOpenAI`, `AsyncAnthropic`) or on a streamed call
  reported "a plain retry passes too": the re-run was never awaited, or came back as a
  stream, and an unreadable re-run counted as a pass. Re-runs now use a sync twin of the
  client (or await it), never stream, and a re-run that returns nothing readable is an
  error, never a pass. The same applies to judge calls that fall back to your client.
- `with guard(OpenAI()) as client:` and `async with` raised `AttributeError`; they work.
- `client.with_options(...)` and `client.copy(...)` returned an unguarded client; the copy
  is guarded now. `chat.completions.parse()` (structured outputs) is checked too.
- `guard()` is typed as returning the client you passed, so editors keep completions.
  `guard(guarded, config=...)` applies the new settings instead of ignoring them.
- The provider is read from the client's base URL (Groq, Gemini, OpenAI, OpenRouter,
  Ollama), so `fix` and exported regression tests can re-run logged calls without
  `--caller`.
- Logs (`log_to`) mask secret-looking headers and keys (`Authorization`, `api_key`, ...),
  and a call with a parameter JSON cannot hold (`NOT_GIVEN`, a timeout object, a pydantic
  class) is still logged instead of silently dropped. "Not given" placeholders are no
  longer stored as parameters.
- An answer that mentions `<think>` mid-text was cut there before checking; only reasoning
  blocks that open the response are removed now.
- `@watch` on a method read `self` as the prompt; it takes the first string or list
  argument.
- Deeply nested JSON (thousands of levels) is reported as a format failure instead of
  crashing the check; the background hourly cap and the response registry are thread-safe.

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
