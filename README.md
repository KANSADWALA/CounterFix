# counterfix

[![PyPI](https://img.shields.io/pypi/v/counterfix)](https://pypi.org/project/counterfix/)
[![Python](https://img.shields.io/pypi/pyversions/counterfix)](https://pypi.org/project/counterfix/)
[![CI](https://github.com/KANSADWALA/CounterFix/actions/workflows/ci.yml/badge.svg)](https://github.com/KANSADWALA/CounterFix/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://kansadwala.github.io/CounterFix/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/KANSADWALA/CounterFix/blob/main/LICENSE)

**Find why an LLM call failed, and get a fix that was tested to work.**

Most LLM evaluation tools tell you *that* a model failed. counterfix is built to tell you
*why*, and to hand back a fix it has already checked. When a call fails, it changes one
thing at a time (raises `max_tokens`, asks for bare JSON, adds the source text, tells the
model to check the question's premises, repeats a missed instruction) and re-runs the
call. A change counts only if it works in 2 of 3 re-runs. The change that fixes the
failure is both the diagnosis and the prescription. We call this **counterfactual
diagnosis**.

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

That is all most people need: the core has 4 small dependencies and needs no API key.
counterfix is an independent open-source project, not affiliated with any model provider.

Optional extras add a provider's SDK only if you do not have it already:

| You use | Install |
| --- | --- |
| OpenAI's SDK, or any OpenAI-compatible API through it (Groq, Gemini, OpenRouter, Ollama, vLLM, ...) | `pip install "counterfix[openai]"` |
| Anthropic's SDK | `pip install "counterfix[anthropic]"` |
| LiteLLM or LangChain | `pip install "counterfix[litellm]"` or `"counterfix[langchain]"` |
| Benchmarks (MMLU, GSM8K, ARC, TruthfulQA, IFEval) | `pip install "counterfix[benchmarks]"` |
| HumanEval / MBPP and HumanEval+ / MBPP+ (EvalPlus) | `pip install "counterfix[code]"` |
| BLEU / chrF, or BERTScore, embeddings and perplexity | `"counterfix[metrics]"` or `"counterfix[semantic]"` |
| Toxicity and Presidio personal-data checks | `pip install "counterfix[safety]"` |
| Suite files in YAML, the MCP server, OpenTelemetry, LlamaIndex | `"counterfix[eval]"`, `"counterfix[mcp]"`, `"counterfix[otel]"`, `"counterfix[llamaindex]"` |

If that SDK is already in your project, plain `pip install counterfix` is enough.

## Quickstart

### 1. See it work in 10 seconds (no key)

```python
import counterfix
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
  fix: Repeat the broken instructions at the end of the prompt as a short checklist. ...
```

`judge=False` runs only the live checks: free, instant, no extra calls. More cases in
[`examples/quickstart.py`](https://github.com/KANSADWALA/CounterFix/blob/main/examples/quickstart.py).

### 2. Wrap your client (one line)

```python
from openai import OpenAI
import counterfix

client = counterfix.guard(OpenAI())  # same API as before
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Why did Einstein win his third Nobel Prize?"}],
)
print(response.choices[0].message.content)
print(response.diagnosis.summary())  # live checks, already run on this call
```

Any OpenAI-compatible endpoint works, for example a local Ollama server:

```python
client = counterfix.guard(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
```

The guarded client behaves like the original: sync or `AsyncOpenAI`, `stream=True`,
`chat.completions.parse()`, the Responses API (`responses.create` / `.parse`),
`with` / `async with`, and `with_options()` all work and stay checked. Your type checker sees the client you passed. Two things differ:
`isinstance(client, OpenAI)` is False (pass `client.unwrapped` where a library insists on
the real object), and type checkers do not know `response.diagnosis`, so typed code can use
`counterfix.get_diagnosis(response)`.

The Anthropic SDK works the same way (`messages.create`, sync or async; `tool_use`
arguments are checked against each tool's `input_schema`):

```python
from anthropic import Anthropic

client = counterfix.guard(Anthropic())
```

Already on LiteLLM or LangChain? Add a callback; it only observes, never changes a reply:

```python
litellm.callbacks = [counterfix.litellm_callback()]
llm.invoke("...", config={"callbacks": [counterfix.langchain_callback()]})
```

Not using any of these? Decorate the function that calls your model (sync or
async). It takes a prompt string or chat messages and returns text or a response:

```python
@counterfix.watch(model="llama3.2")
def ask(prompt: str) -> str:
    return my_local_model(prompt)


answer = ask("Answer in one word: sky colour?")  # returned unchanged
print(ask.last_diagnosis.summary())
```

### 3. Ask a judge for the deeper look

```python
report = counterfix.diagnose(response)  # adds judge checks: false premise, invented facts, ...
print(report.explain())
```

The judge comes from your own key (a free `GROQ_API_KEY` or `GEMINI_API_KEY` is enough;
see [Bring your own key](#bring-your-own-key)). Run `counterfix setup` to see what it
found. Full script: [`examples/diagnose_with_judge.py`](https://github.com/KANSADWALA/CounterFix/blob/main/examples/diagnose_with_judge.py).

### 4. Get a fix that was tested to work

```python
result = counterfix.fix(response)  # a guarded response, its diagnosis, or a CallRecord
print(result.explain())
```

```text
found: F1 Truncation (95%)
plain retry: passed 0/2
tried raise_max_tokens: passed 0/2 -> no (still: F1 Truncation)
tried raise_max_tokens_more: passed 2/3 -> VERIFIED
fix: Raise max_tokens a lot (long answers, or reasoning that uses the budget)
  change: {'max_tokens': 4096}
7 re-run(s); stopped at the first verified fix
```

For each failure, counterfix changes one thing (raise `max_tokens`, ask for bare JSON,
check the question's premises, repeat missed instructions, add the source text, ...),
re-runs the call, and re-checks it. A change counts as the fix only if the failure is
gone, with no new failure, in 2 of 3 re-runs, so luck does not count. It first re-runs
the call unchanged: if a plain retry already passes, no change is claimed. The verified
change is in `result.fix` (`changed` params, or full new `messages`) and on the finding's
prescription (`verified=True`). `max_runs` caps the re-runs (15); judge calls come on top.
When nothing passes, `result.escalation` lists what is left (retrieval, a larger model, a
code tool, a clarifying question).

### 5. Log calls, then report on them

```python
client = counterfix.guard(
    OpenAI(), config=counterfix.Config(on_diagnosis=counterfix.log_to("calls.jsonl"))
)
```

```bash
counterfix diagnose calls.jsonl                      # terminal summary, live checks only (free)
counterfix diagnose calls.jsonl --judge --html report.html --json results.jsonl
counterfix diagnose calls.jsonl --fail-on-failure    # exit code 1 if any call failed (CI)
```

The HTML report is one self-contained file you can send to a teammate: totals per failure
type, then each call with its prompt, answer, why it was flagged and the suggested fix.
The JSON lines (`{"record": ..., "diagnosis": ...}`) suit log pipelines, and can be read
back with `counterfix diagnose` or `counterfix.load_log()`.

### 6. Judge checks on production traffic, without slowing it down

```python
config = counterfix.Config(
    sample_rate=0.05,  # every flagged call + 5% of the rest get judge checks
    max_judged_per_hour=60,  # cost cap; calls over it are skipped, never delayed
    on_judged=counterfix.log_to("judged.jsonl"),
)
client = counterfix.guard(OpenAI(), config=config)
```

Judge checks run on one background thread, so your call returns as fast as before. If the
queue fills up, new calls are skipped rather than waited on; `counterfix.background.stats`
counts what was judged and skipped. Call `counterfix.background.flush()` before a short
script exits if you want the queued checks to finish.

### 7. Evaluate a dataset: the score, the misses by cause, the fixes

```bash
counterfix eval capitals.jsonl --model groq:openai/gpt-oss-120b --metric exact_match --judge --fix 5
```

Scores every answer (exact match, F1, ROUGE-L, BLEU, multiple choice, final number, JSON,
latency, ...), groups the misses by cause (F1 to F14), and searches verified fixes on up to
5 misses. Streamed calls also record TTFT; reports show latency, TTFT and tokens/s
percentiles per model. Guides: [Evaluate a dataset](https://kansadwala.github.io/CounterFix/evals/),
[Speed and cost](https://kansadwala.github.io/CounterFix/speed/).

### 8. RAG: was it the retriever or the model?

```bash
counterfix eval rag.jsonl --model groq:openai/gpt-oss-120b --metric recall@5 --metric faithfulness --fix 5
```

Retrieval metrics from gold labels (hit, recall, precision, MRR, nDCG at k), judge metrics
(faithfulness, context precision and recall, answer relevancy, citations), and the cause of
each miss: F10 retrieval miss or F11 context ignored, with verified fixes (the gold passage
proves the retriever was at fault; the key passage first fixes ignored context). To test
your whole pipeline, pass it as a function: `counterfix eval qa.jsonl --app myapp:answer`
(or `evaluate(..., app=answer)`; LangChain and LlamaIndex outputs are read as they are).
Guide: [RAG evaluation](https://kansadwala.github.io/CounterFix/rag/).

### 9. Code and benchmarks

```bash
counterfix eval coding.jsonl --model groq:openai/gpt-oss-120b --metric tests --allow-code --fix 5
counterfix benchmark gsm8k --model ollama:qwen2.5:3b --limit 200
counterfix benchmark humaneval+ --model groq:openai/gpt-oss-120b --allow-code
```

Code runs against its tests in a sandbox (Docker, or a subprocess for trusted
benchmarks; off until you allow it). A failure is named (syntax, wrong name or signature,
runtime error, wrong output, timeout: F12), with what the code returned, and fixes such as
sending the error back are verified on re-run (measured so far: 2 of 8 repeatable
failures fixed on qwen2.5:3b; gpt-oss-120b passed 18 of 20 HumanEval problems). `--samples N` reports pass@k. Benchmarks: MMLU, GSM8K, ARC-Challenge,
TruthfulQA, IFEval (official checks), HumanEval/MBPP (counterfix's checker, any OS) and
HumanEval+/MBPP+ (scored by EvalPlus, Linux and macOS), with seeded subsets that fit a free
tier. Scores match lm-evaluation-harness on the same prompts (GSM8K 0.81 vs 0.79, MMLU
abstract algebra 0.56 vs 0.56, on qwen2.5:3b). Guides: [Code evaluation](https://kansadwala.github.io/CounterFix/code/),
[Benchmarks](https://kansadwala.github.io/CounterFix/benchmarks/).

### 10. Prompt injection, personal data, token confidence

```bash
counterfix benchmark injection --model ollama:qwen2.5:3b --fix 10
counterfix eval support.jsonl --model groq:openai/gpt-oss-120b --metric pii --metric toxicity
counterfix eval qa.jsonl --model openai:gpt-4o-mini --param logprobs=true --metric confidence
```

20 injection probes (instructions planted in retrieved passages and tool results, checked
exactly with canaries) and a judge check for any call with untrusted text: F13, with
verified fixes such as marking untrusted text as data. PII in answers (built-in
recognizers with checksums, optional Presidio), toxicity (Detoxify), and token confidence
where the provider returns logprobs. Guides: [Safety](https://kansadwala.github.io/CounterFix/safety/),
[Confidence and perplexity](https://kansadwala.github.io/CounterFix/confidence/).

### 11. Agents and tool calls

```bash
counterfix benchmark tools --model ollama:qwen2.5:3b --fix 10
counterfix eval tool_tests.jsonl --model groq:openai/gpt-oss-120b --metric tool_call --fix 5
```

```python
rec = counterfix.agents.record(task, langgraph_result, tools=tools, expect_tools=["search"])
counterfix.diagnose(rec)  # F14: unknown tool, loop, missing call; judge: task done, and why
```

Runs from OpenAI-style messages, LangChain/LangGraph, Anthropic, OpenTelemetry spans or
guarded responses. Verified fixes for tool choice: `tool_choice`, naming the tool,
describing the arguments. Guide: [Agents and tool calls](https://kansadwala.github.io/CounterFix/agents/).

### 12. Keep fixed failures fixed (pytest)

```bash
counterfix fix calls.jsonl --judge --out fixes.jsonl
counterfix export-tests fixes.jsonl --out tests/test_prompt_regressions.py
pytest tests/test_prompt_regressions.py
```

Each verified fix becomes a test case that fails if the failure comes back after a prompt
edit or a model change. In your own tests, `counterfix.testing.assert_ok(response)` fails
with the explanation when counterfix finds a failure.

### More

| | |
| --- | --- |
| Streams (`stream=True`) | every chunk passes straight through; the whole answer is checked at the end |
| `with counterfix.context(source=..., reference=...)` | tell the checks what the model was given (RAG) or the right answer |
| `counterfix profile / compare / drift / route` | measure models, compare an upgrade on your own calls, alert on drift, pick the cheapest safe model per task |
| `counterfix.otel.export()`, `counterfix.otel.SpanChecker()` | send diagnoses to OpenTelemetry, or check the GenAI spans you already have |
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
`semantic`, `metrics`, `eval`, `code`, `benchmarks`, `safety`, `mcp`, `cache`, `cli`,
`report`, `jev`.

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
- Verified fixes: `counterfix.fix()` (counterfactual re-runs, 2 of 3 must pass) ✅
- Prompt regression tests: `counterfix export-tests` and a pytest plugin ✅
- Model profiles, `compare`, `drift`, `route`; fixes ordered by what worked per model ✅
- OpenTelemetry export and GenAI span reader; streams; RAG context; MCP server ✅
- Released on PyPI (`pip install counterfix`), docs site, releases by trusted publishing ✅
- Evaluation, wave 1: `counterfix eval`, reference metrics, TTFT and speed/cost, verified speed fixes ✅
- RAG evaluation: retrieval and judge metrics, F10 retrieval miss vs F11 context ignored, verified RAG fixes ✅
- OpenAI Responses API in `guard()`; evaluate your own app or RAG pipeline (`--app`) ✅
- Code evaluation: sandboxed tests, F12 with its causes, verified code fixes, pass@k ✅
- Benchmarks: MMLU, GSM8K, ARC, TruthfulQA, IFEval, HumanEval(+), MBPP(+), checked against lm-eval ✅
- Safety: prompt injection (F13, probes and judge check, verified fixes), PII, toxicity; token confidence and perplexity ✅
- Agents and tool calls: trajectories from any framework, F14 checks, `tool_call` metric, verified tool-choice fixes ✅
- Trust in the metrics: a labelled set and measured accuracy for every judge-based check, and a human spot check of the labels ✅

Next:

1. A public dataset of diagnosed failures with their verified fixes
2. Larger labelled sets and full profiles for more models
3. A BFCL loader, so the tool-use benchmark can compare models

## License

MIT
