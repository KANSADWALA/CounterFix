# Measured results

All numbers come from the labelled seed set: 54 prompts, 6 per failure type, answered by
two models (Groq `openai/gpt-oss-120b` and local `qwen2.5:3b`), so 108 labelled answers.
With this few positives per type, read them as rough. They are re-run before every release
as a regression gate (`bench/eval_judges.py`, `bench/fix_seed.py`).

## Detection

Live checks plus a panel of three judges (qwen3.8-27b, gemini-3.5-flash-lite,
gemma-4-31b-it), judge run 3:

| Type | Labelled failures | Precision | Recall |
| --- | --- | --- | --- |
| F1 Truncation | 12 | 1.00 | 1.00 |
| F2 Format break | 1 | 1.00 | 1.00 |
| F3 Over-refusal | 3 | 0.67 | 0.67 |
| F4 Hallucination | 15 | 1.00 | 0.93 |
| F5 False premise | 1 | 0.50 | 1.00 |
| F6 Dropped instruction | 2 | 1.00 | 1.00 |
| F8 Computation | 5 | 1.00 | 1.00 |
| F9 Ambiguous request | 1 | 0.33 | 1.00 |

Weak spots: F9 (2 false alarms), F5 and F3 (1 each). Full tables, per-judge numbers and
every misjudged item: `bench/results/judges.md` and `judges.json`.

## Verified fixes

The engine on all 38 labelled failures, re-run on the model that produced them, judged
by a two-judge panel:

| Measure | Result |
| --- | --- |
| Top cause matches the label | 28 of 38 (74%) |
| A plain retry already passed (no fix claimed) | 7 of 38 (18%) |
| **Verified fix, when the cause was right and a retry did not pass** | **18 of 23 (78%)** |
| Re-runs of the target model | 3.7 per failure |

Fixes that verified: `raise_max_tokens_more` 11, `allow_unknown` 3, `work_it_out` 2,
`json_only_prompt` 1, `add_detail` 1. Most misses were detection misses (the two-judge
panel did not re-find 8 hallucinations), not fixes that failed. Item by item:
`bench/results/fixes.md`.

## Model profiles

Failure rate per task on the 54-prompt probe suite (6 per task; 5 for F5, where one probe
is a control). Shipped with counterfix in `counterfix/data/profiles/`.

| Task | gpt-oss-120b (Groq) | qwen2.5:3b (Ollama) |
| --- | --- | --- |
| F1 long answers under a token limit | 6/6 | 6/6 |
| F2 JSON and structured output | 0/6 | 1/6 |
| F3 sensitive but harmless requests | 1/6 | 2/6 |
| F4 facts it may not know | 4/6 | not yet |
| F5 questions with a false premise | 0/5 | not yet |
| F6 exact formatting instructions | 0/6 | 3/6 |
| F7 reworded prompts | 1/6 | 3/6 |
| F8 maths and counting | 0/6 | not yet |
| F9 vague requests | 0/6 | not yet |

gpt-oss-120b was judged by gemini-3.5-flash-lite with no skipped checks. The qwen2.5:3b
run lost its judge to Gemini's free daily quota partway through, so its judge-only rows are
left out until they are re-measured, and its F3, F6 and F7 rows used live checks and the
cheap comparison only. Profiles for gemini-3.5-flash-lite and qwen3.8-27b are next; both
were blocked by free-tier daily limits on 2026-10-19.

Which fixes verified per model (from the seed run) is in each profile's `fixes`; for
example, on both models the first `max_tokens` raise never verified (0/6) on the seed
truncations, while the bigger raise did (5/6 and 6/6).

### Does a profile save re-runs?

Six new truncation prompts (not in the seed set the profile learned from) on local
qwen2.5:3b, live checks only, each failure fixed twice
(`bench/profile_ordering.py`, `bench/results/profile_ordering.md`):

| | Re-runs per failure | Verified fixes |
| --- | --- | --- |
| Default order | 6.0 | 6/6 |
| Profile order | 4.0 | 6/6 |

The profile knew the first `max_tokens` raise never worked on this model, so it went
straight to the bigger one: a third fewer calls for the same result.
