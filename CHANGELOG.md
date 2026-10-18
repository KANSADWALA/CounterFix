# Changelog

All notable changes to counterfix. Versions follow [semantic versioning](https://semver.org).

## 0.1.0 (unreleased)

First real release.

- `guard(client)` for any OpenAI-compatible client: live checks on every call with no extra
  LLM calls (F1 truncation incl. reasoning budgets, F2 format and tool-call arguments,
  F3 refusal, F6 exact instruction checks, empty output, provider filter). Fail-open.
- `Config(mode="retry")`: one automatic retry with a known fix, keeping the better response.
- `counterfix.diagnose()`: live checks plus judge checks (F3, F4, F5, F6, F8, F9), one cause
  per mistake; `judge=False` for live checks only; `Diagnosis.explain()` for readable output.
- `@counterfix.watch`: live checks on any function that calls a model (sync or async).
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
