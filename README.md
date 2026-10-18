# counterfix

**Find why an LLM call failed, and get a fix that was tested to work.**

Most LLM evaluation tools tell you *that* a model failed. counterfix is built to tell you
*why*, and to hand back a fix it has already checked. When a call fails, it changes one
thing at a time (adds context, removes a false premise, rewords the prompt, gives the
model a tool, raises `max_tokens`) and re-runs the call. The change that fixes the
failure is both the diagnosis and the prescription. We call this **counterfactual
diagnosis**.

> **Status: pre-alpha.** The `guard()` wrapper, the live checks (F1 to F3, F6 in code,
> empty output, provider filter), optional auto-retry, the prompt linter, `diagnose()` with
> LLM-judge checks work today. The counterfactual engine is being built. APIs will change.

## What it plugs into

counterfix is middleware: it sits between your code and any text-in, text-out LLM.

```python
from openai import OpenAI
from counterfix import guard

client = guard(OpenAI())  # same API as before
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Why did Einstein win his third Nobel Prize?"}],
)
print(response.choices[0].message.content)
print(response.diagnosis.summary())  # what counterfix found
```

The same wrapper works with any OpenAI-compatible endpoint, for example Groq or a local
Ollama server:

```python
client = guard(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
```

Live checks make no extra LLM calls. They catch truncation (including reasoning models that
spend the whole budget thinking), broken JSON or schema mismatches, broken tool-call
arguments, refusals, empty answers, provider content filters, and broken instructions that
code can verify exactly: sentence, word and bullet counts, letter case, reply language
(by script), British or American spelling, banned words, "end with a question", "only the
number", "one word" and similar (in the spirit of IFEval).

### Optional auto-retry

```python
from counterfix import Config, guard

client = guard(OpenAI(), config=Config(mode="retry"))  # one retry with a known fix
r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=40)
for attempt in r.diagnosis.retries:
    print(attempt.changes, attempt.outcome)  # e.g. {'max_tokens': 160} fixed
```

Retry keeps whichever response is better, records what it tried, and never raises if the
retry fails. Add your own fixes per model with `RuleStore` and `FixRule`.

### Deeper checks with a judge (diagnose mode)

Some failures need meaning to spot: invented facts, false premises, content instructions
("mention the price"), ambiguity, wrong arithmetic. These checks ask a judge model, so they
cost extra calls and never run in live mode. Ask for them when you want the deeper look:

```python
import counterfix

report = counterfix.diagnose(response)  # a response from a guarded client
print(report.summary(), "| judged by", report.judge)
```

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

## Install (development)

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

Next, in order:

1. First release (v0.1.0) with a quickstart and examples
2. Readable reports: terminal summary, JSON for logs, shareable HTML
3. Production-safe diagnose: sample a share of traffic in the background, with a cost cap
4. More plugs: Anthropic SDK, LiteLLM, LangChain, a decorator for any function, async
5. Verified fixes: try a fix, re-run the call, suggest only what worked
6. Prompt regression tests for CI (pytest plugin)
7. OpenTelemetry export

## License

MIT
