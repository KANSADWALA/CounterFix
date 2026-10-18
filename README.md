# counterfix

**Find why an LLM call failed, and get a fix that was tested to work.**

Most LLM evaluation tools tell you *that* a model failed. counterfix is built to tell you
*why*, and to hand back a fix it has already checked. When a call fails, it changes one
thing at a time (adds context, removes a false premise, rewords the prompt, gives the
model a tool, raises `max_tokens`) and re-runs the call. The change that fixes the
failure is both the diagnosis and the prescription. We call this **counterfactual
diagnosis**.

> **Status: alpha (v0.1.0).** `guard()`, the live checks, auto-retry, `diagnose()` with
> judge checks and the prompt linter work today. The counterfactual engine (fixes that are
> re-run and verified) is next. APIs may still change.

## Install

```bash
pip install "counterfix[openai]"
```

The core needs no API key. The `openai` extra is only for wrapping OpenAI-compatible
clients (OpenAI, Groq, Gemini, OpenRouter, Ollama, vLLM, ...).

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
[`examples/quickstart.py`](examples/quickstart.py).

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

Not using an OpenAI-style client? Decorate the function that calls your model (sync or
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
found. Full script: [`examples/diagnose_with_judge.py`](examples/diagnose_with_judge.py).

### 4. Log calls, then report on them

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

### 5. Judge checks on production traffic, without slowing it down

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

## What it checks

Live checks make no extra LLM calls. They catch truncation (including reasoning models that
spend the whole budget thinking), broken JSON or schema mismatches, broken tool-call
arguments, refusals, empty answers, provider content filters, and broken instructions that
code can verify exactly: sentence, word and bullet counts, letter case, reply language
(by script), British or American spelling, banned words, "end with a question", "only the
number", "one word" and similar (in the spirit of IFEval).

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
retry:  {'max_tokens': 80} fixed
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

## Failure types (v1)

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
`semantic`, `cache`, `cli`, `report`, `jev`.

## Roadmap

Done:

- `guard()` for any OpenAI-compatible client, live checks, optional auto-retry ✅
- Prompt linter (`counterfix lint`) ✅
- Judge checks, `diagnose()`, bring your own key, `counterfix setup` ✅
- Reports: `log_to()`, `counterfix diagnose LOG` with terminal, JSON and HTML output ✅
- Background judge checks on a sample of traffic, with a cost cap ✅
- `@counterfix.watch` decorator for any function, sync or async ✅

Next, in order:

1. First release on PyPI (v0.1.0)
2. More plugs: Anthropic SDK, LiteLLM, LangChain
3. Verified fixes: try a fix, re-run the call, suggest only what worked
4. Prompt regression tests for CI (pytest plugin)
5. OpenTelemetry export

## License

MIT
