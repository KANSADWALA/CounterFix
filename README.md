# counterfix

**Find why an LLM call failed, and get a fix that was tested to work.**

Most LLM evaluation tools tell you *that* a model failed. counterfix is built to tell you
*why*, and to hand back a fix it has already checked. When a call fails, it changes one
thing at a time (adds context, removes a false premise, rewords the prompt, gives the
model a tool, raises `max_tokens`) and re-runs the call. The change that fixes the
failure is both the diagnosis and the prescription. We call this **counterfactual
diagnosis**.

> **Status: pre-alpha.** The `guard()` wrapper, the live checks (F1 to F3, empty output,
> provider filter), optional auto-retry and the prompt linter work today. The LLM-judge
> checks and the counterfactual engine are being built. APIs will change.

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
arguments, refusals, empty answers and provider content filters.

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

1. Foundations and seed failure set ✅
2. Core records and the `guard()` plug ✅
3. Cheap live detectors (F1 to F3), auto-retry and the prompt linter ✅
4. LLM-judge and sampling detectors
5. Diagnose mode and HTML report
6. Counterfactual engine with verified fixes
7. Model failure profiles
8. More plugs: decorator, Anthropic, LiteLLM, LangChain, OpenTelemetry
9. Evaluation on public benchmarks
10. Release and paper

## License

MIT
