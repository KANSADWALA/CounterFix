# counterfix

[![PyPI](https://img.shields.io/pypi/v/counterfix)](https://pypi.org/project/counterfix/)
[![Python](https://img.shields.io/pypi/pyversions/counterfix)](https://pypi.org/project/counterfix/)
[![CI](https://github.com/KANSADWALA/CounterFix/actions/workflows/ci.yml/badge.svg)](https://github.com/KANSADWALA/CounterFix/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://kansadwala.github.io/CounterFix/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/KANSADWALA/CounterFix/blob/main/LICENSE)

**Find out why an LLM call failed, and prove the fix works.**

One line around the client you already use. counterfix names the likely failure type
(F1 to F14) on every call. Ask it for a fix and it changes one thing at a time (raises
`max_tokens`, asks for bare JSON, adds the source text, tells the model to check the
question's premises, repeats a missed instruction) and re-runs the call. A change counts
only if it passes clearly more often than re-running the call unchanged (a Fisher exact
test against a plain retry). When none does, counterfix says so and lists what to try
next. We call this **counterfactual diagnosis**.

counterfix works alongside your evaluation and tracing tools (DeepEval, Ragas, Promptfoo,
Langfuse, MLflow, ...): they tell you *that* calls fail and how often; counterfix tells you
*why* one failed and which change verifiably fixes it.

Already have a results file from one of them? Point counterfix at it:

```bash
promptfoo eval -o results.json        # or a DeepEval test_run_*.json, or Ragas scores as CSV
counterfix diagnose results.json      # which cause (F1 to F14) behind each failed test case
counterfix fix results.json           # a change that verifiably fixes it
```

Promptfoo files are re-run exactly as sent, with your deterministic assertions as the pass
test; DeepEval and Ragas files need `--caller` to re-run. See
[DeepEval, Ragas and Promptfoo](https://kansadwala.github.io/CounterFix/eval-tools/).

> **Status: alpha** (latest version in the PyPI badge above). Everything below works today: live checks, judge checks,
> verified fixes, reports, regression tests, profiles, dataset and RAG evaluation, code
> evaluation, benchmarks, prompt-injection and personal-data checks, token confidence,
> and agent and tool-call checks. APIs may still change.
>
> **Known limits.** Verified fixes are common for cut-off answers and broken JSON, rare
> for invented facts (3 of 14 on our seed set) and none yet for over-refusals or false
> premises: there counterfix says so and lists what to try instead. For code that gives
> wrong results, a small model (qwen2.5:3b) got a verified fix on 2 of 8 repeatable
> failures. For prompt injection, one fix (`delimit_untrusted`) was verified on 9 of 10
> injected probes for gpt-oss-120b, none for qwen2.5:3b. The 20-probe `tools` benchmark
> is a smoke test (qwen2.5:3b passes all 20), not a model comparison. Shipped model
> profiles are small (54 probes; qwen2.5:3b only partly measured). The labels behind our accuracy
> numbers are reference labels; a human spot check agreed with 33 of 35 of them. See
> [`docs/results.md`](https://kansadwala.github.io/CounterFix/results/).

## Install

```bash
pip install counterfix
```

The core has 4 small dependencies and needs no API key. counterfix is an independent
open-source project, not affiliated with any model provider. If your project already has
the OpenAI or Anthropic SDK, nothing else is needed; otherwise add it with
`pip install "counterfix[openai]"` or `"counterfix[anthropic]"`.

## Quickstart

```python
import counterfix
from openai import OpenAI

client = counterfix.guard(OpenAI())  # 1. wrap your client (same API as before)
r = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
print(r.diagnosis.summary())  # 2. see why it failed (live checks, free)
print(counterfix.fix(r).explain())  # 3. get a fix that was re-run and verified
```

```text
found: F1 Truncation (95%)
plain retry: passed 0/6
tried raise_max_tokens: passed 0/2 vs retry 0/6, p=1.000 -> no (still: F1 Truncation)
tried raise_max_tokens_more: passed 5/5 vs retry 0/6, p=0.002 -> VERIFIED
fix: Raise max_tokens a lot (long answers, or reasoning that uses the budget)
  change: {'max_tokens': 4096}
13 re-run(s); stopped at the first verified fix
```

That is the whole loop. Step 2 costs nothing: live checks make no extra calls. Step 3
re-runs the call (here 13 times) and may use a judge model for checks code cannot do; the
judge comes from your own key (a free `GROQ_API_KEY` or `GEMINI_API_KEY` is enough; run
`counterfix setup` to see what it finds). "Verified" means the change beat asking again
*on this call*, not that it works on every prompt. How the test works:
[Verified fixes](https://kansadwala.github.io/CounterFix/fixes/).

**Try it with no key at all:**

```python
from counterfix import CallRecord, Message

record = CallRecord(
    messages=[Message(role="user", content="Answer in exactly one word: sky colour?")],
    response_text="The sky is usually blue on a clear day.",
)
print(counterfix.diagnose(record, judge=False).explain())
```

```text
F6 Dropped instruction (90%)
  why: instructions not met: exactly 1 word (has 9); a one-word answer (has 9 words)
  try: Repeat the broken instructions at the end of the prompt as a short checklist. ...
```

## Plug it into what you use

| You use | One line |
| --- | --- |
| OpenAI SDK, or any OpenAI-compatible API (Groq, Gemini, OpenRouter, Ollama, vLLM, ...) | `client = counterfix.guard(OpenAI(...))` |
| Anthropic SDK | `client = counterfix.guard(Anthropic())` |
| LiteLLM | `litellm.callbacks = [counterfix.litellm_callback()]` |
| LangChain | `llm.invoke(..., config={"callbacks": [counterfix.langchain_callback()]})` |
| Your own function | `@counterfix.watch(model="llama3.2")` on a function that takes a prompt and returns text |

The guarded client behaves like the original: sync or async, `stream=True`,
`chat.completions.parse()`, the Responses API, `with` blocks and `with_options()` all work
and stay checked, and your type checker sees the client you passed. Two things differ:
`isinstance(client, OpenAI)` is False (pass `client.unwrapped` where a library insists on
the real object), and type checkers do not know `response.diagnosis`, so typed code can use
`counterfix.integrations.get_diagnosis(response)`. Callbacks only observe; they never change a
reply. Details: [Plug it in](https://kansadwala.github.io/CounterFix/plugs/).

## In production

```python
client = counterfix.guard(
    OpenAI(),
    log="calls.jsonl",  # every call and its diagnosis, as JSON lines
    sample_rate=0.05,  # judge checks on every flagged call + 5% of the rest
    mode="retry",  # re-run a flagged call once with a known fix
)
```

Judge checks run on a background thread, capped per hour, so calls return as fast as
before. Retry mode is off by default; without it counterfix only reports. For anything
else, pass a `counterfix.Config` (the shortcuts above override it).

```bash
counterfix diagnose calls.jsonl --judge --html report.html   # a report you can send a teammate
counterfix fix calls.jsonl --judge --out fixes.jsonl          # search verified fixes in bulk
counterfix export-tests fixes.jsonl --out tests/test_prompt_regressions.py
```

Each verified fix becomes a pytest case that fails if the failure comes back after a prompt
edit or a model change; `counterfix.testing.assert_ok(response)` does the same in your own
tests. Guides: [In production](https://kansadwala.github.io/CounterFix/production/),
[Prompt regression tests](https://kansadwala.github.io/CounterFix/testing/).

## The same checks over a dataset

The checks and fixes above also run over a whole dataset: the misses grouped by cause, and
a verified fix searched for each. Install with `pip install "counterfix[evals]"`.

```bash
counterfix eval qa.jsonl --model groq:openai/gpt-oss-120b --metric exact_match --judge --fix 5
```

counterfix works alongside your evaluation tools rather than replacing them: it keeps only
the metrics its diagnosis and fixes rely on. For BLEU, ROUGE, BERTScore, judge-scored RAG
metrics, PII and toxicity, use DeepEval, Ragas or Promptfoo (and bring their results to
counterfix); for MMLU, GSM8K or HumanEval, use lm-evaluation-harness or EvalPlus.

| Area | What you get | Guide |
| --- | --- | --- |
| Datasets | Exact match, F1, number, choice, JSON, latency; misses by cause; fixes on up to N misses | [Evaluate a dataset](https://kansadwala.github.io/CounterFix/evals/) |
| RAG | Retrieval metrics from gold labels, and whether the retriever (F10) or the model (F11) caused each miss | [RAG evaluation](https://kansadwala.github.io/CounterFix/rag/) |
| Code | Tests run in a sandbox, failures named (F12), pass@k, fixes such as sending the error back | [Code evaluation](https://kansadwala.github.io/CounterFix/code/) |
| Prompt injection | Canary probes and a judge check (F13), with verified fixes | [Safety](https://kansadwala.github.io/CounterFix/safety/) |
| Your evaluation tool's results | DeepEval, Ragas and Promptfoo failures explained and fixed | [DeepEval, Ragas and Promptfoo](https://kansadwala.github.io/CounterFix/eval-tools/) |
| Agents | Tool misuse (F14) from OpenAI, LangGraph, Anthropic or OpenTelemetry runs | [Agents and tool calls](https://kansadwala.github.io/CounterFix/agents/) |
| Speed and confidence | Latency, TTFT and tokens/s percentiles; token confidence where logprobs exist | [Speed](https://kansadwala.github.io/CounterFix/speed/), [Confidence](https://kansadwala.github.io/CounterFix/confidence/) |

### More

| | |
| --- | --- |
| `with counterfix.context(source=..., reference=...)` | tell the checks what the model was given (RAG) or the right answer |
| `counterfix profile / compare / drift / route` | measure models, compare an upgrade on your own calls, alert on drift, pick the cheapest safe model per task |
| `counterfix.otel.export()`, `counterfix.otel.SpanChecker()` | send diagnoses to OpenTelemetry, or check the GenAI spans you already have |
| `counterfix lint prompts.jsonl` | catch prompt problems before you send them |
| `counterfix mcp` | an MCP server so coding agents can lint prompts and check calls |

Full docs: https://kansadwala.github.io/CounterFix/ (or build locally with `mkdocs serve`). Measured accuracy and fix rates:
[`docs/results.md`](https://kansadwala.github.io/CounterFix/results/). A Colab-ready notebook on a free model:
[`examples/quickstart.ipynb`](https://github.com/KANSADWALA/CounterFix/blob/main/examples/quickstart.ipynb).

## What it checks

Live checks make no extra LLM calls. They catch truncation (including reasoning models that
spend the whole budget thinking), broken JSON or schema mismatches, broken tool-call
arguments, refusals, empty answers, provider content filters, and broken instructions that
code can verify exactly: sentence, word and bullet counts, letter case, reply language
(by script), British or American spelling, banned words, "end with a question", "only the
number", "one word" and similar (in the spirit of IFEval). On agent runs they also catch
calls to tools that were not offered, repeated identical calls and missing expected calls;
with a canary planted in untrusted text, they catch a followed injection for free.

### Optional auto-retry: a fix that was re-run

```python
from counterfix import Config, guard

client = guard(OpenAI(base_url=GROQ_URL, api_key=key), config=Config(mode="retry"))
r = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[{"role": "user", "content": "List the planets of the solar system, one per line."}],
    max_tokens=20,
)
for f in r.diagnosis.retried_from:
    print("before:", f.failure_type.value, f.evidence[0].detail)
for attempt in r.diagnosis.retries:
    print("retry: ", attempt.changes, attempt.outcome)
print("after: ", r.diagnosis.summary())
```

```text
before: F1 the answer stopped at the max_tokens limit (20)
retry:  {'max_tokens': 1024} fixed
after:  no failure found by 5 checks
```

You get back the fixed response (all eight planets), and the record of what changed.
Retry keeps whichever response is better, records what it tried, and never raises if the
retry fails. Add your own fixes per model with `RuleStore` and `FixRule`.

### Deeper checks with a judge (diagnose mode)

Some failures need meaning to spot: invented facts, false premises, content instructions
("mention the price"), ambiguity, wrong arithmetic. These checks ask a judge model, so they
cost extra calls and never run in live mode; `counterfix.diagnose()` adds them when you
want the deeper look (on failures, on a sample of traffic, or in tests). `diagnose()` takes
a response from a guarded client or a `CallRecord`.

One mistake gets one cause: a wrong calculation is reported as F8, not also as an invented
fact (F4), and a cut-off answer as F1, not also as dropped instructions (F6). The findings
that were set aside are kept in `report.suppressed`.

The F8 check never trusts the judge's arithmetic: the judge writes one expression and
counterfix evaluates it with a small whitelist-only evaluator. A sampling check
(`SamplingCheck`) re-asks the model and flags answers that change between samples.
An optional Jev back-end (`counterfix[jev]`) plugs into the same judge interface.

### Bring your own key

counterfix never ships API keys. Live checks need none: they wrap your own client and make
no calls of their own. Judge checks use a judge you choose, in this order:

1. `counterfix.diagnose(response, judge="groq:qwen/qwen3.8-27b")`
2. `counterfix.configure(judge=...)`, once per process. A list of specs makes a voting
   panel, e.g. `["groq:qwen/qwen3.8-27b", "gemini:gemini-3.5-flash-lite", "ollama:llama3.2"]`
3. the `COUNTERFIX_JUDGE` environment variable (comma-separated for a panel)
4. a key it finds: `GROQ_API_KEY` or `GEMINI_API_KEY` (both have free tiers),
   `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or a model in a local Ollama. A different
   model family from the one being checked is preferred.
5. otherwise the checked model judges itself, with a `SelfJudgeWarning`, because judges are
   lenient with their own mistakes.

See what counterfix finds on your machine (key values are never printed):

```bash
counterfix setup          # keys, local models, and the judge that will be used
counterfix setup --test   # also asks the judge one tiny question
```

### Lint prompts before you send them

```python
from counterfix import lint

for issue in lint("Return ONLY JSON with keys name and age for: Priya, 29."):
    print(issue.rule, issue.title, "->", issue.suggestion)
```

Or from the command line, e.g. in CI:

```bash
counterfix lint prompts/*.txt --fail-on-warning
```

| Rule | Flags | Tends to cause |
| --- | --- | --- |
| L1 | A constraint buried in the middle of a long prompt | F6 |
| L3 | A factual question with no "I don't know" option | F4, F5 |
| L4 | JSON/YAML/CSV asked for in words only, no example or JSON mode | F2 |
| L7 | Pasted outside text with no delimiters | prompt injection |
| L2 | Instructions that conflict (needs `--judge`) | F6, F7 |
| L5 | System prompt and user message disagree (needs `--judge`) | F6, F3 |
| L6 | Request missing a key detail (needs `--judge`) | F9 |

Design rules:

- **Fail open.** If counterfix itself errors, your call still returns normally.
- **Your errors stay yours.** Exceptions from the provider reach your code unchanged.
- **Safe defaults.** Live mode only flags failures and makes no extra LLM calls.
- **Light core.** The core depends on 4 small libraries; providers and heavy tools are extras.

## Failure types

| Code | Failure |
| --- | --- |
| F1 | Truncation |
| F2 | Format break (JSON, schema) |
| F3 | Over-refusal |
| F4 | Hallucination from missing knowledge |
| F5 | False-premise acceptance |
| F6 | Dropped instruction |
| F7 | Prompt fragility |
| F8 | Computation or token-level error |
| F9 | Ambiguous request |
| F10 | Retrieval miss (RAG: the answer was not in the retrieved text) |
| F11 | Context ignored (RAG: the answer was there, the model missed it) |
| F12 | Code fails its tests |
| F13 | Followed instructions planted in retrieved text or tool output (prompt injection) |
| F14 | Tool misuse (wrong, unknown or missing tool call, loop, agent task not done) |

API errors, rate limits and provider content filters are tracked separately, so they are
never counted as the model's fault.

## Development

```bash
git clone https://github.com/KANSADWALA/CounterFix.git
cd CounterFix
pip install -e ".[dev]"
pytest
```

Optional extras: `openai`, `anthropic`, `litellm`, `langchain`, `llamaindex`, `otel`,
`eval`, `mcp`, `cache`, `cli`, `report`, `jev`; bundles `evals` (dataset runs, reports
and the command line) and `all`.

`python bench/dev_walkthrough.py SPEC` uses the library as a developer would against a real
model (about 40 small calls) and prints OK or the error for each step.

## Roadmap

Done:

- `guard()` for any OpenAI-compatible client, live checks, optional auto-retry ✅
- Prompt linter (`counterfix lint`) ✅
- Judge checks, `diagnose()`, bring your own key, `counterfix setup` ✅
- Reports: `log_to()`, `counterfix diagnose LOG` with terminal, JSON and HTML output ✅
- Background judge checks on a sample of traffic, with a cost cap ✅
- `@counterfix.watch` decorator for any function, sync or async ✅
- Anthropic SDK in `guard()`, LiteLLM and LangChain callbacks ✅
- Verified fixes: `counterfix.fix()` (counterfactual re-runs; a change must beat a plain retry) ✅
- Prompt regression tests: `counterfix export-tests` and a pytest plugin ✅
- Model profiles, `compare`, `drift`, `route`; fixes ordered by what worked per model ✅
- OpenTelemetry export and GenAI span reader; streams; RAG context; MCP server ✅
- Released on PyPI (`pip install counterfix`), docs site, releases by trusted publishing ✅
- Evaluation, wave 1: `counterfix eval`, reference metrics, TTFT and speed/cost, verified speed fixes ✅
- RAG evaluation: retrieval metrics, F10 retrieval miss vs F11 context ignored, verified RAG fixes ✅
- OpenAI Responses API in `guard()`; evaluate your own app or RAG pipeline (`--app`) ✅
- Code evaluation: sandboxed tests, F12 with its causes, verified code fixes, pass@k ✅
- Prompt injection (F13: probes and a judge check, verified fixes); token confidence ✅
- Agents and tool calls: trajectories from any framework, F14 checks, `tool_call` metric, verified tool-choice fixes ✅
- Trust in the metrics: a labelled set and measured accuracy for every judge-based check, and a human spot check of the labels ✅
- 0.4: works with your evaluation tools (your own test as the pass test, DeepEval, Ragas and Promptfoo results diagnosed and fixed); parts that duplicated them removed ✅

Next:

1. A public dataset of diagnosed failures with their verified fixes
2. Larger labelled sets and full profiles for more models
3. A BFCL loader, so the tool-use probe set can compare models

## License

MIT
