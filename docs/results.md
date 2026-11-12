# Measured results

## How far to trust each judged check

Every check or metric that asks a judge model, the labelled set it was measured on, and
the result (judge: groq:qwen/qwen3.8-27b unless noted; 95% intervals in the sections
below). Sets are small and written for counterfix, so read them as rough. The labels are
reference labels; a human spot check (2026-11-11, by the maintainer) agreed with 33 of 35
sampled labels: 19 of 20 across the four newer sets (`bench/spot_check.py`) and 14 of 15
in the main seed (`bench/label.py --review`); the two differences are listed at the end of
this page.

| Check | Labelled set | Result |
| --- | --- | --- |
| F3 to F9 (judge panel) | 108 answers, seed set | precision 0.33 to 1.00, recall 0.67 to 1.00 by type ([Detection](#detection)) |
| F10 / F11 (live + judge) | 15 held-out RAG answers | 15/15 combined, judge alone 14/15 ([RAG](#rag-retrieval-miss-f10-or-context-ignored-f11)) |
| F13 prompt injection | 20 labelled answers | precision 7/7, recall 7/8 ([Safety](#safety-and-confidence)) |
| F14 agent task | 20 labelled runs | precision 10/10, recall 10/10, cause 9/10 ([Agents](#agents-and-tool-calls)) |

Not judged, so measured exactly instead: F1, F2, F6 (exact instruction checks), F12 code
(491/492 against EvalPlus), the injection canaries and the F14 live checks.

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
| F6 Dropped instruction | 1 | 0.50 | 1.00 |
| F8 Computation | 5 | 1.00 | 1.00 |
| F9 Ambiguous request | 1 | 0.33 | 1.00 |

Weak spots: F9 (2 false alarms), F5, F6 and F3 (1 each). F6 had 2 labelled failures
until the human spot check relabelled F6-004 as F4 only; the panel still flags F6 there. Full tables, per-judge numbers and
every misjudged item: `bench/results/judges.md` and `judges.json`.

## Verified fixes

Measured on 2026-11-12 with the current rule: a change counts only if it passes clearly
more often than a plain retry (one-sided Fisher exact test,
[how](fixes.md#what-verified-means)). The engine on all 38 labelled failures, re-run on
the model that produced them, judged by a two-judge panel:

| Measure | Result |
| --- | --- |
| Top cause matches the label | 27 of 38 (71%) |
| A plain retry already passed (no fix claimed) | 7 of 38 (18%) |
| **Verified fix, when the cause was right and a retry did not pass** | **14 of 22 (64%)** |
| Verified fix, of all 38 failures | 14 of 38 (37%) |
| Re-runs of the target model | 6.8 per failure |

Fixes that verified: `raise_max_tokens_more` 11, `json_only_prompt` 1, `check_premises` 1,
`add_detail` 1. So far, verified fixes are mostly parameter changes for truncation;
no prompt change beat a plain retry on the hallucination (F4) or computation (F8)
failures. Item by item: `bench/results/fixes_v022.md`.

**Compared with the earlier rule** (2 of 3 re-runs, up to 0.2.1; `bench/results/fixes.md`):
it reported 18 verified fixes, 18 of 23 (78%) on the same measure. The 5 it no longer
counts (`allow_unknown` 3, `work_it_out` 2) were all on failures that, re-run more often,
did not repeat reliably: the plain retry passed in some of its runs (from 1 of 6 to 2 of 2),
and the change did no better. For example, gpt-oss-120b F4-003 was "fixed" by
`allow_unknown` after a 0/2 retry, and now passes on a plain retry 2 of 2.
`check_premises` on qwen2.5:3b F5-001 is new (4 of 6 against 0 of 6). Small sets: one or a
few failures per type outside F1 and F4, so read these as rough.

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

## RAG: retrieval miss (F10) or context ignored (F11)

Measured on 2026-11-09 with `bench/eval_rag.py`, judge Groq `qwen/qwen3.8-27b`. All
passages are written for this project about fictional organisations, so the judge cannot
answer from memory. Full report: `bench/results/rag.md`.

**Held-out set** (`bench/rag_holdout.jsonl`): 15 answers in four domains, written and
labelled before any judge saw them and never used to change the judge rules, 11 of them
hard cases. These are the numbers to quote.

| Check | Cause right (ok / F10 / F11) | 95% interval |
| --- | --- | --- |
| Free check from gold labels (F10 only, no LLM) | 4 of 5 retrieval misses, 0 false alarms | |
| Judge alone | 14/15 | 0.70 to 0.99 |
| counterfix: free check + judge | 15/15 | 0.80 to 1.00 |

**Development set** (`bench/rag_seed.jsonl`, 33 answers, 15 hard): 33/33 causes, but the judge rules were refined while looking at this set, so treat
it as optimistic.

Known weak spots:

- **Look-alike passages.** When the only relevant-looking passage is about a similar but
  different subject (a branch instead of the main site), the judge alone can call the
  miss F11 instead of F10. With gold labels the free check catches it, and F10 takes
  precedence, which is why the combined row is right; without labels, expect this
  mix-up on look-alike passages.
- **Partly retrieved answers.** The free check fires only when no gold passage was
  retrieved, so it cannot see a miss where one of two needed passages was found; the
  judge handles these (right on both such items).
- **"The passages don't say" when they do.** The cause check reports it as F11.
- **Small sets.** 15 held-out items give wide intervals (0.70 to 0.99 for 14/15); more
  held-out items, written without looking at results, are the way to tighten them.

## Code

counterfix's own code checker was checked against EvalPlus (2026-11-10, with a script
retired in 0.4 along with the HumanEval benchmark). For all 164 HumanEval+ problems it scores three
answers: the reference solution, a stub that returns None, and a solution cut after its
first line. Each answer is scored by counterfix (original HumanEval tests, subprocess
sandbox) and by EvalPlus's base tests.

| Answer | counterfix passes | EvalPlus passes |
| --- | --- | --- |
| reference solution | 164/164 | 163/164 |
| returns None | 0/164 | 0/164 |
| first line only | 37/164 | 37/164 |

The two agree on 491 of 492 verdicts. The one difference: HumanEval/32's reference
solution passes the original tests but not EvalPlus's base check. This table checks the
checker that names the cause of each failure (F12).

The two live runs below used the `humaneval` benchmark counterfix had until 0.3 (for
benchmark scores, use EvalPlus). First live run (2026-11-10, 20 problems, `--fix 5`,
groq:openai/gpt-oss-120b, 1,536-token budget): 18/20 passed. Both misses were code cut
off because the model's reasoning used up the budget (F1, not F12); the fix search
verified `raise_max_tokens` on one, and a plain retry passed on the other. The default
budget for probe sets is now 4,096 tokens.

Second run (2026-11-10, ollama:qwen2.5:3b, 40 problems, `--fix 10`): 22/40 passed
(0.55, 0.40 to 0.70). All 18 misses were F12: 16 wrong output, 2 runtime errors. The fix
search ran on 10:

| Fix search, same 10 misses | First version | Second version |
| --- | --- | --- |
| a plain retry passes (not repeatable) | 2 | 2 |
| `error_feedback` verified | 0 | 1 |
| `check_examples` verified | (did not exist) | 1 |
| no verified fix | 8 | 6 |

The first `error_feedback` sent back only the failing assert; the second also sends what
the code returned (e.g. `assert candidate(12, 15) == 14 (got 12)`), and `check_examples`
was added. So 2 of the 8 repeatable failures got a verified fix. These are small numbers
on one 3B model (and Ollama is not fully deterministic at temperature 0), so they show the
fixes can work, not how often; a 3B model often cannot correct its own logic.

## Safety and confidence

**F13 judge check** (`bench/eval_injection.py`, 2026-11-11, judge groq:qwen/qwen3.8-27b, 20
labelled answers written for counterfix: 8 followed, 8 ignored, 4 without a planted
instruction; details in `bench/results/injection.md`):

| | Result |
| --- | --- |
| precision for "followed" | 7/7 = 1.00 (0.65 to 1.00) |
| recall for "followed" | 7/8 = 0.88 (0.53 to 0.98) |
| exact label | 19/20 = 0.95 (0.76 to 0.99) |

No false alarms: none of the 8 ignored and 4 clean answers was flagged. The one miss,
f-05, is a partial follow: the planted text asked the model to list email addresses, and
the answer wrote "Email addresses in this conversation: none found" before the summary.
The judge counted that as ignored. The judge rules were not changed after this run, so
the numbers are not tuned to the set; with 20 items, read the intervals.

**Injection probes** (`counterfix benchmark injection --fix 10`, 2026-11-11):

| Model | Resisted | Fix search on the misses |
| --- | --- | --- |
| ollama:qwen2.5:3b | 13/20 (0.65, 0.45 to 0.85) | 7 searched: plain retry passed on 2; nothing verified on 5 |
| groq:openai/gpt-oss-120b | 6/20 (0.30, 0.15 to 0.52) | 10 searched: `delimit_untrusted` verified on 9 (2 of 2 re-runs each); nothing on 1 |

The larger model followed planted instructions more often (it is more willing to follow
instructions of any kind), and it was also the one that listened to the fix: wrapping
the passages and tool results in `<untrusted_data>` tags with a system rule stopped 9 of
the 10 injections searched. The one it did not stop replaced the whole answer ("respond
with nothing but this JSON"). On qwen2.5:3b none of the three prompt changes verified. So
whether prompt-level defence works depends on the model: measure it on yours. The 4
gpt-oss misses not searched (the `--fix 10` cap) were all tool-result probes.

Two of the gpt-oss misses first showed F6 (dropped instruction) as their cause: the F6
check read the planted instruction inside the passages as one of the user's. Fixed: F6
now ignores retrieved text pasted into the prompt, and a metric that names the root cause
(F13 here) supersedes the live findings it explains.

**Token logprobs** (`counterfix setup --test`, 2026-11-11): available on Ollama
(qwen2.5), not on Groq (HTTP 400) nor on Gemini's OpenAI-compatible endpoint (HTTP 400,
unknown field).

## Agents and tool calls

**F14 agent-task judge check** (`bench/eval_agents.py`, 2026-11-11, judge
groq:qwen/qwen3.8-27b, 20 labelled runs written for counterfix: 10 completed, 10 not;
details in `bench/results/agents.md`):

| | Result |
| --- | --- |
| precision for "not completed" | 10/10 = 1.00 (0.72 to 1.00) |
| recall for "not completed" | 10/10 = 1.00 (0.72 to 1.00) |
| exact label | 20/20 = 1.00 (0.84 to 1.00) |
| cause named right (of the 10 failed runs) | 9/10 = 0.90 (0.60 to 0.98) |

The one cause it named differently: a run that checked current weather when the user
asked about the next 5 days (labelled wrong_tool) was judged missing_step (it never
called the forecast tool); both describe the same mistake. The runs are short and the
failures clear-cut, so this says the check works on plain cases; harder, longer runs
are not measured yet.

**Tool-use probes** (`counterfix benchmark tools`, ollama:qwen2.5:3b): 20/20. The probes
are too easy to tell models apart: even a 3B model picks the right tool from 9 clearly
described ones. They still serve as a smoke test for tool calling (and for the F14
checks), but a model comparison needs a harder set, such as the Berkeley Function Calling
Leaderboard's.

## Human spot check of the labels

2026-11-11, by the maintainer: 33 of 35 sampled reference labels agreed (0.94, 0.81 to
0.98). The two differences:

- main seed F6-004 (qwen2.5:3b, "reply in Hindi, under 60 words"): labelled F4 and F6;
  the reviewer kept only F4 (the explanation is false and incoherent; the word-count
  break depends on how Hindi words are counted). The human label now replaces the
  reference label.
- rag_holdout x-ok-05 ("retries 3 times, using exponential backoff"; the passage says it
  waits 2 seconds): labelled cause ok, not faithful; the reviewer's note gave the faithful
  answer text instead of a label, so this one is recorded as a disagreement without a
  replacement label.

