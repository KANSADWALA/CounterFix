# Launch posts (drafts; nothing has been posted)

Check the numbers against `docs/results.md` and add the links before posting.

## LinkedIn

I built counterfix, an open-source Python library that tells you *why* an LLM call
failed and gives you a fix that was tested to work.

Most LLM tools score answers. counterfix names the cause (nine failure types: truncation,
broken JSON, over-refusal, invented facts, false premises, dropped instructions, prompt
fragility, maths errors, ambiguous requests) and then does what an engineer would do:
change one thing, re-run the call, check again. A change counts as the fix only if it
works in 2 of 3 re-runs, and a plain retry is tried first, so luck is never sold as a fix.

On our labelled seed set, when the cause was right and a retry alone did not help,
counterfix returned a verified fix for 18 of 23 failures, at about 4 re-runs each.

One line to add: `client = counterfix.guard(OpenAI())`. Works with OpenAI-compatible
APIs, Anthropic, LiteLLM, LangChain and OpenTelemetry. Bring your own (free) key; no
telemetry.

pip install counterfix · (GitHub link)

## r/LocalLLaMA

**Title:** I made a library that finds why a local model's answer failed and tests fixes
until one actually works (2 of 3 re-runs)

Body:

- Wrap any OpenAI-compatible client (Ollama, vLLM, llama.cpp server):
  `counterfix.guard(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))`
- Free live checks on every call: truncation (incl. reasoning models that burn the whole
  budget thinking), broken JSON / tool args, refusals, exact instruction checks.
- `counterfix.diagnose(response)` adds judge checks; the judge can be another local model,
  so nothing leaves your machine.
- `counterfix.fix(response)` changes one thing at a time and keeps only what passes 2 of 3
  re-runs. It also tries a plain retry first and refuses to call that a fix.
- Honest results: on qwen2.5:3b, "check the premises first" fixed the Marie Curie false
  premise in only 1 of 3 runs, so it is reported as *not* fixed.
- Model profiles: `counterfix profile ollama:qwen2.5:3b` measures failure rates per task,
  `counterfix route` picks the cheapest model that does not fail a task.

MIT. Feedback on the failure taxonomy very welcome.

## Hugging Face (community post / model card style)

**counterfix: verified fixes for LLM failures.** A Python library that classifies a
failed LLM call into one of nine failure types, then re-runs the call with one targeted
change at a time and keeps only changes that pass 2 of 3 re-runs. Ships measured profiles
(54 probes each; so far openai/gpt-oss-120b and, partly, qwen2.5:3b: update this list
before posting), a prompt linter, a pytest plugin for prompt regression tests, and an MCP
server.

## Demo video (2 minutes): outline

1. 0:00 The problem: a dashboard says "hallucination 12%". Why? What do I change?
2. 0:20 `guard(OpenAI())`, a truncated answer, `response.diagnosis.summary()`.
3. 0:45 `counterfix.fix(response)`: plain retry 0/2, raise max_tokens 2/3, VERIFIED.
4. 1:10 The honest case: the false premise that "check premises" fixes only 1/3.
5. 1:30 `counterfix export-tests` turns the fix into a pytest case; CI catches a regression.
6. 1:50 `pip install counterfix`, link.
