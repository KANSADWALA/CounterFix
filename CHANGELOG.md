# Changelog

All notable changes to counterfix. Versions follow [semantic versioning](https://semver.org).

## Unreleased

A smaller front door: counterfix is now presented as a reliability layer for single LLM
calls ("find out why an LLM call failed, and prove the fix works"), with evaluation as the
same checks run over a dataset. Nothing that worked in 0.2.1 stops working.

- **Shortcuts on `guard()` and `watch()`:** `guard(client, mode="retry", log="calls.jsonl",
  judge=..., sample_rate=0.05)` needs no `Config`. With a `config`, the shortcuts override
  its matching fields; the object you passed is not changed. `watch()` takes `log`, `judge`
  and `sample_rate`.
- **`counterfix.__all__` lists 12 names** (`guard`, `watch`, `diagnose`, `fix`,
  `configure`, `Config`, `Diagnosis`, `FixResult`, `FailureType`, `CallRecord`, `Message`,
  `log_to`). The other 0.2 names (`evaluate`, `metrics`, `RuleStore`, `litellm_callback`,
  ...) still import from `counterfix` with no warning; they now load on first use.
- **`import counterfix` is about 40% faster** (it no longer loads the metrics, evaluation,
  background and OpenTelemetry modules until used).
- **New extras:** `counterfix[evals]` (dataset runner, metrics, benchmarks, code, reports,
  CLI) and `counterfix[all]`.
- README and docs reorganised around the three-step loop: wrap, see why, get a verified
  fix. The docs menu now reads Core loop, In production, Evaluation (optional).
- Fixed: `counterfix.fix(response)` raised `NoJudgeError` on a first run with no judge
  key, even for a cut-off answer that needs no judge. It now falls back to judging with
  the checked model (with `SelfJudgeWarning`, as `diagnose()` already did), and with no
  model at all it fixes what the live checks find and warns that facts and premises were
  not checked. A judge you name explicitly that cannot be built still raises.
- A test now freezes the 0.2.1 public surface (every top-level name, module path and
  subpackage export, and both entry points), so later restructuring cannot break imports
  unnoticed.

"Verified" now has a stated error rate.

- **A change must beat a plain retry to count as a fix.** The plain retry and each change
  are re-run up to 6 times; a change is verified only if a one-sided Fisher exact test
  says it passed clearly more often (p at or below 0.05 divided by the number of changes
  the search may try). The old rule (2 of 3 re-runs) called a change that does nothing
  "verified" in 11% to 36% of searches on a model that fails at random; the new one in
  under 1%. Applies to `fix()`, `speed_fix()` and the fix search in `counterfix eval`.
  The statistics live in `counterfix.verify`.
- `explain()` shows the retry's count, each change's count and its p-value; attempts gain
  `p_value`, and `FixResult.level` is the bar each change had to reach.
- Default `max_runs` is now 30 (was 15 for `fix()` and 9 for `eval`, MCP and profiles):
  a deterministic fix such as a larger `max_tokens` now takes about 13 re-runs instead of
  6. Intermittent failures cost about the same, because the retry stops early once it
  passes too often for any change to beat.
- `need=` on `fix()`, `speed_fix()` and `fix_item()` is ignored with a
  `DeprecationWarning`. `check_retry=False` now assumes the plain retry never passes.
- Fixed: `counterfix fix --caller MODEL@http://host/v1` (and `compare`, `profile`,
  `drift`) crashed with "unknown provider"; they now take every spec `eval` takes, and an
  unusable spec prints one line and exits 2. `counterfix fix` also shows the real reason a
  call cannot be re-run instead of always blaming the log.
- Fixed: the F8 expression evaluator had no bound on nested loops (one expression ran for
  over two minutes); it now stops after 100,000 steps.
- Docs: "so luck does not count" and similar claims replaced by what the test does and
  does not show.
- Re-measured on the 38 seed failures (`bench/results/fixes_v022.md`): 14 verified fixes
  (cause right and retry not passing: 14 of 22, 64%), against 18 (18 of 23, 78%) under the
  old rule. The 5 that dropped out (`allow_unknown`, `work_it_out`) were on failures that,
  re-run more often, did not repeat reliably. 6.8 re-runs per failure (was 3.7).
- `bench/fix_seed.py`: `--learn` counts only the change the engine verified (it counted
  any attempt with 2 passes), and the report counts "plain retry passes" from where the
  engine stopped, not from a pass count.

## 0.2.1 (2026-11-11)

Wording that claims no more than counterfix does.

- New tagline everywhere (PyPI, README, docs, CLI, MCP server): "LLM evaluation that
  diagnoses failures and re-runs fixes to see which ones work". The old one ("get a fix
  that was tested to work") implied a fix every time; often no change passes, and
  counterfix says so.
- `diagnose()` output, the HTML report and `counterfix lint` show an untested suggestion as
  `try:`; only a suggestion that passed re-runs is shown as `fix (verified):`. `fix()`
  output is unchanged (it prints `fix:` only for a verified change).
- The README and docs intros say counterfix looks for the likely cause and tests fixes,
  and that when none passes it says so and lists what to try next.
- Docs: the confidence example shows the guarded client.

## 0.2.0 (2026-11-11)

counterfix grows from diagnosing single calls into one-stop evaluation, where every miss
gets a cause and, where one works, a verified fix: an eval runner with reference metrics,
speed and cost; RAG evaluation (F10, F11); code evaluation and standard benchmarks (F12);
prompt injection (F13), personal data, toxicity and token confidence; agents and tool
calls (F14); and a measured accuracy for every judged check, with a human spot check of
the labels. Also the OpenAI Responses API, evaluating your own app, and fixes from using
the library end to end as a new developer would.

- `counterfix eval DATA` and `counterfix.evaluate()`: run a model over a JSONL, JSON or
  CSV dataset (or score recorded answers), report each metric with a 95% bootstrap
  confidence interval, group the misses by cause (F1 to F14), and with `--fix N` search
  verified fixes on up to N misses (a fix counts when the item passes its metrics in 2 of
  3 re-runs). Suite files (`.json`, or `.yaml` with the `eval` extra), `--out` to resume
  after a daily limit (kept answers are reused only for unchanged prompts),
  `--fail-under METRIC=VALUE` for CI, text and HTML reports. Failed calls count as misses
  with cause INFRA; items a metric cannot score are shown as "not scored"; misses the
  judge could not check are reported. Items that pass their metrics but that a live check
  flags (e.g. a cut-off answer under a latency gate) are listed as "flagged";
  `--fix` searches fixes for them too (the flagged failure must be gone), and
  `--fail-on-flagged` fails CI on them.
- `counterfix.metrics`: `exact_match`, `f1`, `contains`, `rouge_l` (same numbers as
  `rouge-score`), `bleu` and `chrf` (sacrebleu, `metrics` extra), `choice`, `number`,
  `json_valid`, `json_match`, `embedding_similarity` and `bertscore` (`semantic` extra),
  `latency`, `ttft`; answer extraction helpers (`extract_choice`, `extract_number`,
  `extract_code`). `counterfix.testing.assert_metric` gates a result in pytest.
- Speed: streamed calls record `record.timing` (TTFT, first reasoning chunk, last chunk,
  chunk count). `counterfix.perf`: latency/TTFT/tokens-per-second percentiles per model,
  cost from prices you set (none are shipped), and `speed_fix()`, verified changes that
  bring a slow call under a target without a new failure. Reports show a speed table.
- RAG evaluation (wave 2): passages on records (`counterfix.context(chunks=...)`,
  `counterfix.rag.record()`, `record_from_llamaindex()`, LangChain retriever capture);
  retrieval metrics from gold labels (`hit`, `recall`, `precision`, `mrr`, `ndcg`, each
  with `@k`); judge metrics with RAGAS definitions (`faithfulness`, `context_precision`,
  `context_recall`, `answer_relevancy`, `citations`) and `answer_correctness` (any task with
  a reference); new failure types F10 retrieval miss
  and F11 context ignored (free F10 from labels, judge check for both, both supersede F4);
  verified RAG fixes (`gold_passage`, `key_passage_first`, `fewer_passages`,
  `quote_then_answer`); eval datasets take `contexts`, `relevant` and `gold_passages`.
  `bench/rag_seed.jsonl` (33 labelled answers, 15 hard; development) and
  `bench/rag_holdout.jsonl` (15 answers in new domains, never used to tune the judge
  rules) with `bench/eval_rag.py` (accuracy with 95% intervals, clear and hard apart,
  judge panel, judges run side by side, every disagreement listed) measure the checks.
- Judge hardening: text inside the judged material is data (the judge is told to ignore
  instructions in it), and tags inside material are neutralised so a retrieved page cannot
  close a section of the judge prompt. Long contexts are shortened evenly per passage
  (60,000 characters) and answers capped (20,000) before they reach a judge. The
  LangChain callback no longer keeps retrievals that no model call used.
- OpenTelemetry: `counterfix.otel.metrics()` records `gen_ai.client.operation.duration`,
  `gen_ai.client.operation.time_to_first_chunk` and `gen_ai.client.token.usage`; diagnosis
  spans also carry token counts, latency and TTFT. `counterfix.hooks()` chains hooks.
- OpenAI Responses API: `guard()` checks `client.responses.create` / `.parse` (sync,
  async, streamed, with TTFT). The call is recorded in chat terms, so every check, judge
  and fix works; `status="incomplete"` with `max_output_tokens` is F1; retry mode changes
  Responses fields. `counterfix.record_from_responses()` for logged calls.
- Evaluate your own app: `evaluate(app=fn)` / `counterfix eval --app module:function`
  calls your function per question and reads what it returns (string, LangChain or
  LlamaIndex output, `(answer, passages)`, a guarded response, a `CallRecord`;
  `counterfix.rag.record_from_output()`). Gold `relevant` labels now apply without
  `contexts` in the data, so an app's own retrieval is scored.
- `evaluate(concurrency=N)` / `--concurrency N`: N items at once, order kept, resumable
  `--out` still safe.
- Model specs: `anthropic:MODEL` and `MODEL@http://host/v1` work in `counterfix eval`
  (same as `--counterfix-caller`); `counterfix.presets.caller_for()`.
- Newer OpenAI models: re-runs and judges resend with `max_completion_tokens` (and
  without `temperature` / `top_p`) when the model refuses them, and remember it; F1
  fixes raise whichever limit field the call used.
- TTFT from the LangChain callback (`on_llm_new_token`) and LiteLLM
  (`completion_start_time`) on streamed calls.
- Python 3.14 in CI and the classifiers.
- Token confidence (wave 4): records keep a summary of token logprobs when the provider
  returns them (OpenAI Chat Completions, Ollama 0.12.11+, vLLM; also for streams); the
  `confidence` metric; a judge-confirmed F4 shows the token confidence as evidence;
  re-runs drop `logprobs` where a provider refuses them (Groq); `counterfix setup --test`
  reports which providers return logprobs. `counterfix.perplexity.perplexity()`: corpus
  perplexity with a local Hugging Face model (sliding window, as in the HF guide), with
  `bench/perplexity_check.py` against the guide's GPT-2 large numbers (measured 16.44,
  the guide's figure); `dtype="float16"` for small GPUs.
- Safety: new failure type F13 (followed instructions planted in retrieved text or tool
  output): `InjectionCheck` (judge, runs only on calls with untrusted text) and
  `CanaryCheck` (exact, for probes); verified fixes `delimit_untrusted`,
  `data_not_instructions`, `restate_task`; the `injection` benchmark (20 probes written for
  counterfix, 12 in passages and 8 in tool results, canary-checked). `pii` metric with
  built-in recognizers (email, phone, Luhn-checked cards, IBAN, US SSN, Verhoeff-checked
  Aadhaar, IPv4, API keys), only data not already in the prompt, masked in reports;
  optional Presidio. `toxicity` metric (Detoxify). Extra `safety`. `bench/injection_seed.jsonl`
  (20 labelled answers) and `bench/eval_injection.py` measure the judge check
  (precision 7/7, recall 7/8 with qwen3.8-27b as judge). Measured on the probes:
  gpt-oss-120b resisted 6/20 and `delimit_untrusted` verified on 9 of 10 misses;
  qwen2.5:3b resisted 13/20 with no prompt fix verified.
- Agents and tool calls (wave 5): new failure type F14 (tool misuse).
  `counterfix.agents` reads runs from OpenAI-style messages, LangChain/LangGraph,
  Anthropic blocks, OpenTelemetry GenAI spans or guarded responses
  (`agents.record(task, run)`); `ToolUseCheck` (live, free): unknown tool, loops, missing
  expected call (`expect_tools`), too many steps; `AgentTaskCheck` (judge): task completed,
  and if not, the cause. `tool_call` metric (expected call or `"no_call"`, `any_of`
  values, parallel calls) with causes; datasets take `tools` and `expect_tools` fields;
  `evaluate(app=...)` accepts agent runs. Verified fixes `tool_choice`, `name_the_tool`,
  `describe_arguments`, `no_repeat`. The `tools` benchmark (20 probes written for
  counterfix, BFCL-style). `bench/agent_seed.jsonl` (20 labelled runs) and
  `bench/eval_agents.py` measure the judge check. A tool that was not offered is now F14,
  not F2.
- Trust in the metrics (wave 6): `bench/metric_seed.jsonl` (26 labelled answers) and
  `bench/eval_metrics.py` measure answer_relevancy, answer_correctness, context_recall,
  context_precision and citations; `bench/spot_check.py` is a human spot check of the
  reference labels behind the newer sets (resumable, 5 per set); Measured results opens
  with one table of every judged check and where it was measured. Report lines: "every
  group passed" instead of a list of perfect "weakest groups".
  Measured: every RAG metric agreed with every label (relevancy 12/12, correctness,
  recall and precision 22/22, citations 4/4); the human spot check agreed with 33 of 35
  sampled labels (F6-004 relabelled F4 only by the reviewer).
- Developer walkthrough (counterfix installed from a wheel, used from a clean folder as the
  README and docs say; `bench/dev_walkthrough.py` repeats it on a real model): `counterfix
  eval` / `benchmark` exit 2 when every call failed (a wrong model name no longer passes a
  CI gate); "did you mean" for metric, benchmark and provider names; `guard()` refuses a
  class (`guard(OpenAI)`) or an object with no chat, responses or messages API instead of
  checking nothing; `litellm_callback()`, `langchain_callback()` and `counterfix mcp` say
  what to install; eval reports note when `fix` cannot run with `app=` and when no
  retrieved passage id ever matches a gold `relevant` id; `counterfix.confidence` reads a
  guarded response directly (`confidence.describe(r)`); fix reports tell a failure that
  stayed ("still") from one the change brought ("new"); docs show `default_threshold` for
  custom metrics and the eval exit codes.
- From the walkthrough on real models (qwen2.5:3b, gpt-oss-120b): F5 (false premise)
  now supersedes F9 (a request built on a false premise is not ambiguous; no measured
  number changed); retry mode raises a cut-off call's `max_tokens` 4x and to at least
  1,024 (5 -> 20 or 40 never helped; `fix()` and its measured order are unchanged); eval
  reports name an app by its function name and an unnamed dataset "dataset".
- Fixes from these runs: the F6 instruction check no longer reads instructions planted in
  retrieved text as the user's; a metric that names the root cause (F13, F12) now
  supersedes the live findings it explains; Ollama embedding models (all-minilm, ...) are
  no longer offered as judges or chat models.
- Code evaluation (wave 3): `counterfix.code` runs model-written code against tests in a
  sandbox (Docker with no network and resource caps, or an isolated subprocess; off
  until `--allow-code` / `COUNTERFIX_ALLOW_CODE=1`). The `tests` metric reads HumanEval,
  MBPP or plain assert tests; new failure type F12 (code fails its tests) with causes
  no code, syntax, name, signature, import, runtime, wrong output, timeout (cut-off code
  is F1); verified fixes `error_feedback`, `restate_signature`, `one_code_block`.
  `samples=N` / `--samples N` reports pass@k (unbiased estimator). `bench/code_parity.py`:
  counterfix's checker agrees with EvalPlus on 491 of 492 verdicts.
- Benchmarks: `counterfix benchmark NAME` / `counterfix.benchmarks.run_benchmark()` for
  MMLU, GSM8K, ARC-Challenge, TruthfulQA MC1, IFEval (official checks via lm-eval) and
  HumanEval+/MBPP+ (scored by EvalPlus; Linux and macOS), `humaneval` / `mbpp` (original
  tests, counterfix's checker, any OS), generative, temperature 0, versioned prompts,
  seeded subsets (`--per-group`, `--limit`), per-group accuracy, licences listed; data
  downloaded at run time. Extras `code` and `benchmarks`. `bench/lm_eval_parity.py`
  compares with lm-evaluation-harness: on qwen2.5:3b with the same prompts, GSM8K 0.81 vs
  0.79 (94/100 items scored the same) and MMLU abstract algebra 0.56 vs 0.56 (100/100).
- Code fixes after the first live runs: a failed assert now reports what the code
  returned (`... == 14 (got 12)`), also in `error_feedback`; new fix `check_examples`
  (work the examples by hand first). On qwen2.5:3b HumanEval (40 problems) 2 of 8
  repeatable failures got a verified fix, up from 0. Cut-off code says so instead of
  "invalid syntax". Benchmark answers get 4,096 tokens (reasoning models ran out at 1,536).
  `humaneval` / `mbpp` benchmarks score the original tests with counterfix's checker and
  run on Windows; EvalPlus's scorer (`humaneval+`, `mbpp+`) needs Linux or macOS.

## 0.1.3 (2026-11-09)

- Fix: `counterfix.otel.SpanChecker` crashed inside `span.end()` on newer OpenTelemetry
  SDKs (1.4x), which call a `_on_ending` hook on every span processor. A test now checks
  `SpanChecker` against every hook the installed SDK defines.
- README roadmap: the PyPI release moved to "Done".
- First release published by the Release workflow (trusted publishing), with a GitHub
  Release entry.

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
