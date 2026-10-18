# counterfix: notes for Claude

counterfix is a Python library that plugs into any LLM client, detects why a call failed
(taxonomy F1 to F9), and returns a fix. It is built **for engineers and developers to use**,
not as a research paper: optimise for easy install, clear output, low cost and latency,
and fixes people can trust. Repo: github.com/KANSADWALA/CounterFix (private). PyPI name
`counterfix` is registered (0.0.1 uploaded).

## Rules from the owner (always follow)

- Build everything from scratch per the plan. **Never read or reuse the old
  `LLM_Failure_debugger` folder.**
- **Never create a virtual environment.** Use the existing conda env: `conda activate cuda`.
- Do not use the word "doctor" anywhere (code, docs, names).
- Never read, print or commit `.env` or key values. Keys live only in `.env` (git-ignored).
- Jev (TypeSafe) stays optional.
- No week-by-week schedules; the owner works on it when there is time.
- Automate anything repetitive (labelling, checks); don't hand the owner manual chores.
- Ask before `git commit` / `git push` unless told to go ahead.
- Keep the agent work in this one session: no subagents unless the owner asks
  (they cost more of the owner's usage limits).

## Environment

Windows, PowerShell, project at `D:\My Projects\counterfix`, conda env `cuda`.
Free API keys in `.env`: GROQ_API_KEY, GEMINI_API_KEY. Ollama is installed (qwen2.5:3b).

```powershell
conda activate cuda
pip install -e ".[dev,openai]"        # once, into the cuda env
pytest                                  # 237 passed
ruff check . ; ruff format --check . ; mypy   # CI runs all three (Python 3.10-3.13)
python -m counterfix setup              # which keys / judge are available
python bench\eval_judges.py             # judge accuracy vs labels (cached; Ctrl+C safe)
python bench\eval_judges.py --offline   # same report from cache only, instant
python bench\try_judge.py gemini:gemma-4-31b-it   # debug one judge's request options
```

## Layout

- `src/counterfix/core/`: CallRecord, Diagnosis (findings, suppressed, judge, private record
  and client), Config, detector runner, `precedence.py` (one mistake one cause: F8/F5 beat
  F4; F1/F3 beat F2/F6; same-type findings merge), callers (`RateLimitedCaller` honours
  429 retry-after, retries 5xx twice).
- `src/counterfix/adapters/openai_compat.py`: `guard(client)` middleware, fail-open, retry mode.
- `src/counterfix/detectors/`: live checks (truncation, format, refusal, empty, filter,
  `instructions.py` = IFEval-style exact checks for F6) and `judged.py` (F3, F4, F5, F6
  content-only, F8 via judge-written expression + `safe_eval`, F9); `sampling.py`.
- `src/counterfix/judges/`: LLMJudge (falls back without JSON mode / system message),
  JudgePanel (majority vote), CachedJudge, tasks (judge prompts), `auto.py` (bring your own
  key: passed > `configure()` > `COUNTERFIX_JUDGE` > detected keys/Ollama, other family
  preferred > self-judge with SelfJudgeWarning), optional `jev.py`.
- `src/counterfix/diagnose.py`: `counterfix.diagnose(response_or_record, judge=None)`.
- `src/counterfix/cli.py`: `counterfix lint ... [--judge]`, `counterfix setup [--test]`.
- `src/counterfix/lint/`: prompt linter L1-L7. `rules/`: FixRule / RuleStore for retry mode.
- `bench/`: 54-prompt seed set, runner, auto + reference labels (`bench/labels/`, history
  in `bench/labels/history/`), `eval_judges.py` (writes `bench/results/judges.md`),
  cache `bench/cache/judges.jsonl` (git-ignored). Treat the eval as a **regression gate**.

## Status (2026-10-19, later)

- Committed 4bebf16: v0.1.0 prep (quickstart, examples, CHANGELOG, `diagnose(judge=False)`,
  `explain()`, `report.py` + `counterfix diagnose LOG`, `background.py`, `@watch`, judge run 3).
  Uncommitted since then: Anthropic in `guard()` (`adapters/anthropic.py`, auto-detected by
  `messages` without `chat`), `adapters/callbacks.py` (`litellm_callback`,
  `langchain_callback`), `refusal` stop reason as provider filter.
- Judge run 3 done: every Gemma check filled (3 timeouts). Live checks + panel: F1, F2, F6,
  F8 1.00/1.00; F4 P 1.00 R 0.93 (15 positives); weak: F9 2 FP, F5 1 FP, F3 1 FP / 1 FN.
  `judges.json` now lists each setup's FP/FN keys per type.
- Roadmap steps 1-4 done (plugs: OpenAI-compatible, Anthropic, LiteLLM, LangChain, @watch,
  async). Not done: PyPI upload of 0.1.0 (needs the owner's OK); next is step 5, the
  counterfactual engine. Plugs are duck-typed: CI installs no SDKs, tests use fakes.
- In this shell `conda activate` does not take; call
  `C:/Users/Admin/anaconda3/envs/cuda/python.exe` directly.
- The build plan is a Claude Doc, "LLM Failure Debugger: Build Plan"
  (https://claude.ai/code/artifact/a4489e20-b78d-427e-b2e5-c5fabe50ffd7); an export sits
  untracked at `LLM Failure Debugger Build Plan.md`.

## Roadmap (developer-first, agreed 2026-10-19)

1. First real release v0.1.0: quickstart README, examples, polished `diagnose()`/`setup`.
2. Readable reports: terminal summary, JSON for logs, shareable HTML report.
3. Production-safe diagnose: sample N% of traffic in the background, cost cap, no added latency.
4. Plugs: Anthropic SDK, LiteLLM, LangChain, decorator for any function, async.
5. Verified fixes: the counterfactual engine (try a fix, re-run, suggest only what worked).
6. Prompt regression tests for CI (pytest plugin).
7. Observability: OpenTelemetry export.
