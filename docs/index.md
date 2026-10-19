# counterfix

**Find why an LLM call failed, and get a fix that was tested to work.**

Most LLM tools tell you *that* a call failed. counterfix tells you *why* (one of nine
failure types, F1 to F9) and hands back a fix it has already re-run and checked: it
changes one thing at a time, re-runs the call, and keeps only the change that made the
failure go away in 2 of 3 runs. The change that fixes the failure is both the diagnosis
and the prescription.

```bash
pip install "counterfix[openai]"
```

## Three steps

**1. Wrap your client.** Every call gets free live checks; your code does not change.

```python
from openai import OpenAI
import counterfix

client = counterfix.guard(OpenAI())
response = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
print(response.diagnosis.summary())
```

**2. Ask for the deeper look** when something seems off. A judge model (your own key;
free Groq or Gemini keys work) checks for invented facts, false premises and more.

```python
report = counterfix.diagnose(response)
print(report.explain())
```

**3. Get a verified fix.**

```python
result = counterfix.fix(response)
print(result.explain())  # what was tried, what passed, the change to make
```

## What it is built for

- **Engineers shipping LLM features**: one line to add, nothing to run or host.
- **Low cost**: live checks make no extra calls; judge checks and fixes run only when
  you ask, on a sample of traffic, or with an hourly cap.
- **Fixes you can trust**: nothing is called a fix unless it passed on re-run, and a
  plain retry is tried first so luck is never sold as a fix.
- **Your data stays yours**: no telemetry; the judge can be a local Ollama model.

It plugs into OpenAI-compatible clients (OpenAI, Groq, Gemini, OpenRouter, Ollama, vLLM),
the Anthropic SDK, LiteLLM, LangChain, OpenTelemetry GenAI spans, or any function via a
decorator. See [Plug it in](plugs.md).
