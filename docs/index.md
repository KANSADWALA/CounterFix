# counterfix

**LLM evaluation that diagnoses failures and re-runs fixes to see which ones work.**

Most LLM tools tell you *that* a call failed. counterfix also looks for *why* (one of
fourteen failure types, F1 to F14), then tests fixes: it changes one thing at a time,
re-runs the call, and keeps only a change that made the failure go away clearly more
often than re-running the call unchanged.
When no change passes, it says so and lists what to try next. A change that works is both
the diagnosis and the prescription.

```bash
pip install counterfix
```

Add `"counterfix[openai]"` or `"counterfix[anthropic]"` only if that SDK is not in your
project yet. counterfix is independent and not affiliated with any model provider.

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
- **Fixes you can check**: a change is called a fix only if it beats a plain retry on
  re-run, by a stated statistical test ([how](fixes.md#what-verified-means)).
- **Your data stays yours**: no telemetry; the judge can be a local Ollama model.

It plugs into OpenAI-compatible clients (OpenAI, Groq, Gemini, OpenRouter, Ollama, vLLM),
the Anthropic SDK, LiteLLM, LangChain, OpenTelemetry GenAI spans, or any function via a
decorator. See [Plug it in](plugs.md).

Beyond single calls: [evaluate a dataset](evals.md) or your own app, [RAG
evaluation](rag.md) (retriever or model at fault), [code evaluation](code.md) (sandboxed
tests, pass@k), [benchmarks](benchmarks.md) (MMLU, GSM8K, ARC, TruthfulQA, IFEval,
HumanEval, MBPP), [safety](safety.md) (prompt injection, personal data, toxicity), and [agents](agents.md)
(tool choice, loops, task success). Every miss gets a cause and, where one works, a verified fix.
