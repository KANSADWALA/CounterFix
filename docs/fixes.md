# Verified fixes

```python
result = counterfix.fix(response)
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

## How it decides

1. **Find the cause** with the live checks and (unless `judge=False`) the judge checks.
2. **Try a plain retry first.** If the unchanged call already passes 2 of 3 times, the
   failure was luck, and no change is claimed as the fix.
3. **Change one thing at a time**, most confident cause first: the interventions for
   each cause are listed on its [failure type page](failures/F1.md).
4. **Re-run and re-check.** A change is *verified* only if the failure is gone, with no
   new failure, in 2 of 3 re-runs (`runs=3, need=2`). Runs stop as soon as the outcome is
   settled.
5. **Stop at the first verified fix**, or at `max_runs` (15 calls to your model; judge
   calls come on top).

When nothing passes, `result.escalation` lists what is left outside the prompt:
retrieval, a larger model, a code tool, a clarifying question.

## Using the fix

`result.fix.changed` holds changed parameters (e.g. `{"max_tokens": 4096}`) and
`result.fix.messages` the full new messages when the prompt changed. The finding's
prescription is marked `verified=True`, so reports show it as tested.

## Re-running

`fix()` re-runs the call through the guarded client that made it, or the provider named
on the record (`groq`, `gemini`, `ollama`, `openai`, `openrouter`), or any `caller=` you
pass (anything with `complete(messages, **params)`).

## Learning per model

With a [model profile](profiles.md), `fix()` tries first what worked before on that model
and leaves never-working changes for last (`profile="auto"`, the default, loads the stored
profile when there is one).
