# Judge accuracy against hand labels

Generated 2026-10-18 19:19 UTC by `bench/eval_judges.py` on 108 labelled responses.

Labels: 36 auto, 71 claude-reference, 1 human. No human spot-check of the Claude labels yet (run bench/label.py --review 15).

Labelled failures per type: F1 12, F2 1, F3 3, F4 8, F5 1, F6 2, F8 5, F9 1.
Precision = of the responses flagged, how many truly failed. Recall = of the true
failures, how many were flagged. With few positives, treat numbers as rough.

## live checks (no LLM)

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 12 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F2 | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F3 | 3 | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| F4 | 8 | 0 | 0 | 8 | - | 0.00 | 0.00 |
| F5 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F6 | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F8 | 5 | 0 | 0 | 5 | - | 0.00 | 0.00 |
| F9 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |

## judge groq:qwen/qwen3.8-27b

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 5 | 10 | 3 | 0.33 | 0.62 | 0.43 |
| F5 | 1 | 1 | 3 | 0 | 0.25 | 1.00 | 0.40 |
| F6 | 2 | 1 | 1 | 1 | 0.50 | 0.50 | 0.50 |
| F8 | 5 | 5 | 6 | 0 | 0.45 | 1.00 | 0.62 |
| F9 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |

## judge gemini:gemini-3.5-flash-lite

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 6 | 2 | 2 | 0.75 | 0.75 | 0.75 |
| F5 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| F6 | 2 | 1 | 0 | 1 | 1.00 | 0.50 | 0.67 |
| F8 | 5 | 5 | 2 | 0 | 0.71 | 1.00 | 0.83 |
| F9 | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |

## judge gemini:gemma-4-31b-it

Judge failures (skipped checks): 109 (InternalServerError: 107, APITimeoutError: 2). Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 3 | 9 | 5 | 0.25 | 0.38 | 0.30 |
| F5 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| F6 | 2 | 1 | 1 | 1 | 0.50 | 0.50 | 0.50 |
| F8 | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F9 | 1 | 0 | 2 | 1 | 0.00 | 0.00 | 0.00 |

## panel of all judges

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 6 | 6 | 2 | 0.50 | 0.75 | 0.60 |
| F5 | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |
| F6 | 2 | 1 | 0 | 1 | 1.00 | 0.50 | 0.67 |
| F8 | 5 | 5 | 4 | 0 | 0.56 | 1.00 | 0.71 |
| F9 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |

## counterfix: live checks + panel

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 12 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F2 | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F3 | 3 | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| F4 | 8 | 6 | 6 | 2 | 0.50 | 0.75 | 0.60 |
| F5 | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |
| F6 | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F8 | 5 | 5 | 4 | 0 | 0.56 | 1.00 | 0.71 |
| F9 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
