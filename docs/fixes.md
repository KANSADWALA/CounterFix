# Verified fixes

```python
result = counterfix.fix(response)
print(result.explain())
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

## How it decides

1. **Find the cause** with the live checks and (unless `judge=False`) the judge checks.
2. **Try a plain retry first**, up to `runs` times (6). If it passes so often that no
   change could be shown to do better, the failure was not repeatable and no change is
   claimed as the fix.
3. **Change one thing at a time**, most confident cause first: the interventions for
   each cause are listed on its [failure type page](failures/F1.md).
4. **Re-run and compare.** Each change is re-run up to `runs` times (6). A run passes
   when the failure is gone and no new failure appears. The change is *verified* only if
   it passes clearly more often than the plain retry (below). Runs stop as soon as the
   outcome is settled either way.
5. **Stop at the first verified fix**, or at `max_runs` (30 calls to your model; judge
   calls come on top).

## What "verified" means

A one-sided Fisher exact test compares the change's pass count with the plain retry's.
The change is verified when the p-value is at or below `alpha` (0.05) divided by the
number of changes the search may try, so trying many changes does not make a lucky one
more likely to be called a fix. `result.explain()` prints both counts and the p-value;
`result.level` is the bar each change had to reach.

With 6 runs each and three changes to try (bar 0.0167):

| Plain retry | Change | Verified? |
| --- | --- | --- |
| 0/6 | 6/6 or 5/6 | yes |
| 0/6 | 4/6 | no (p = 0.03) |
| 1/6 | 6/6 | yes |
| 2/6 or more | any | no: a retry passes too often to tell, so no change is claimed |

On a simulated model that fails at random whatever the request says, so that no change
can help, this rule called a change verified in under 1% of searches at every failure
rate from 20% to 90%. The earlier rule (2 of 3 re-runs, before 0.2.2) did so in up to
36%.

What it does not tell you: whether the change works on *other* prompts, or on this one
every time. Keep that question for your tests: [`export-tests`](testing.md) turns a
verified fix into a case you re-run in CI.

`fix(..., check_retry=False)` skips the plain retry and assumes it never passes. Use it
only for failures you know repeat every time, such as a fixed token limit.

When nothing passes, `result.escalation` lists what is left outside the prompt:
retrieval, a larger model, a code tool, a clarifying question. When the model cannot be
called at all (endpoint down, bad key), `fix()` stops after the first failed call and
says so in `result.unreachable` instead of giving advice it never tested.

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

## Your own test as the pass test

Already testing answers with your own function, DeepEval or Ragas? Pass it as
`fix(response, check=...)` and it decides which re-runs pass: see
[Fix with your own test](your-check.md).
