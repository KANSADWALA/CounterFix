# Judge accuracy against hand labels

Generated 2026-10-15 17:52 UTC by `bench/eval_judges.py` on 108 labelled responses.

Labelled failures per type: F1 12, F2 1, F3 3, F4 8, F5 1, F6 1, F8 5, F9 1.
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
| F6 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F8 | 5 | 0 | 0 | 5 | - | 0.00 | 0.00 |
| F9 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |

## judge groq:openai/gpt-oss-20b

Judge failures (skipped checks): 147. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 1 | 0 | 2 | 1.00 | 0.33 | 0.50 |
| F4 | 8 | 2 | 14 | 6 | 0.12 | 0.25 | 0.17 |
| F5 | 1 | 0 | 4 | 1 | 0.00 | 0.00 | 0.00 |
| F6 | 1 | 0 | 3 | 1 | 0.00 | 0.00 | 0.00 |
| F8 | 5 | 5 | 2 | 0 | 0.71 | 1.00 | 0.83 |
| F9 | 1 | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 |

## judge groq:qwen/qwen3.8-27b

Judge failures (skipped checks): 8. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 8 | 25 | 0 | 0.24 | 1.00 | 0.39 |
| F5 | 1 | 1 | 3 | 0 | 0.25 | 1.00 | 0.40 |
| F6 | 1 | 0 | 4 | 1 | 0.00 | 0.00 | 0.00 |
| F8 | 5 | 4 | 2 | 1 | 0.67 | 0.80 | 0.73 |
| F9 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |

## judge gemini:gemini-3.5-flash-lite

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 7 | 17 | 1 | 0.29 | 0.88 | 0.44 |
| F5 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| F6 | 1 | 0 | 11 | 1 | 0.00 | 0.00 | 0.00 |
| F8 | 5 | 4 | 0 | 1 | 1.00 | 0.80 | 0.89 |
| F9 | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |

## panel of all judges

Judge failures (skipped checks): 0. Internal errors: 0.

| Type | Positives | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 12 | 0 | 0 | 12 | - | 0.00 | 0.00 |
| F2 | 1 | 0 | 0 | 1 | - | 0.00 | 0.00 |
| F3 | 3 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| F4 | 8 | 8 | 21 | 0 | 0.28 | 1.00 | 0.43 |
| F5 | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| F6 | 1 | 0 | 7 | 1 | 0.00 | 0.00 | 0.00 |
| F8 | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| F9 | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |
