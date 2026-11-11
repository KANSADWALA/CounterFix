# RAG checks against the labelled RAG sets

Generated 2026-11-09 13:53 UTC by `bench/eval_rag.py`. All passages are written for this project about fictional organisations, so a judge cannot answer from memory; labels are reference labels following the rules in the judge task. With this many items, read the 95% intervals, not the point estimates.

## held-out set: never used to change the judge rules: quote these numbers

15 items: 5 ok, 5 F10 retrieval miss, 5 F11 context ignored; 4 clear, 11 hard (denies_present 1, distractor 2, embellished 1, hedged 1, multi_passage 1, parametric 1, partial_answer 1, partial_retrieval 1, stale_value 1, wrong_synthesis 1).

Free check from gold labels (F10 only, no LLM): caught 4 of 5 retrieval misses; false alarms: 0. Not caught: x-f10-05 (partial_retrieval) (the free check fires only when no gold passage was retrieved; partial retrieval needs the judge).

### judge groq:qwen/qwen3.8-27b

| Items | Cause accuracy | 95% interval | Not judged |
| --- | --- | --- | --- |
| all (15) | 14/15 | 0.70 to 0.99 | 0 |
| clear (4) | 4/4 | 0.51 to 1.00 | 0 |
| hard (11) | 10/11 | 0.62 to 0.98 | 0 |
| counterfix: free check + judge (15) | 15/15 | 0.80 to 1.00 | 0 |

Faithfulness (every claim supported, or not) agrees with the label on 14 of 15 items (0.70 to 0.99); 0 items have no faithfulness label and 0 could not be judged.

| Label \ judged | ok | F10 | F11 | error |
| --- | --- | --- | --- | --- |
| ok | 5 | 0 | 0 | 0 |
| F10 | 0 | 4 | 1 | 0 |
| F11 | 0 | 0 | 5 | 0 |

| Cause | Precision | Recall |
| --- | --- | --- |
| F10 | 1.00 | 0.80 |
| F11 | 0.83 | 1.00 |

Disagreements with the labels:

- x-f10-03 (distractor): label F10, judged F11
- x-f11-02 (denies_present): faithfulness label True, judged False: 0/1 claims supported; unsupported: The passages don't mention night buses

## development set: the judge rules were refined on it: these numbers are optimistic

33 items: 11 ok, 10 F10 retrieval miss, 12 F11 context ignored; 18 clear, 15 hard (distractor 3, embellished 1, hedged 1, multi_passage 2, nuance 1, other_year 1, parametric 1, partial_answer 1, partial_retrieval 1, refusal 1, stale_value 1, wrong_synthesis 1).

Free check from gold labels (F10 only, no LLM): caught 9 of 10 retrieval misses; false alarms: 0. Not caught: h-f10-02 (partial_retrieval) (the free check fires only when no gold passage was retrieved; partial retrieval needs the judge).

### judge groq:qwen/qwen3.8-27b

| Items | Cause accuracy | 95% interval | Not judged |
| --- | --- | --- | --- |
| all (33) | 33/33 | 0.90 to 1.00 | 0 |
| clear (18) | 18/18 | 0.82 to 1.00 | 0 |
| hard (15) | 15/15 | 0.80 to 1.00 | 0 |
| counterfix: free check + judge (33) | 33/33 | 0.90 to 1.00 | 0 |

Faithfulness (every claim supported, or not) agrees with the label on 32 of 32 items (0.89 to 1.00); 1 items have no faithfulness label and 0 could not be judged.

| Label \ judged | ok | F10 | F11 | error |
| --- | --- | --- | --- | --- |
| ok | 11 | 0 | 0 | 0 |
| F10 | 0 | 10 | 0 | 0 |
| F11 | 0 | 0 | 12 | 0 |

| Cause | Precision | Recall |
| --- | --- | --- |
| F10 | 1.00 | 1.00 |
| F11 | 1.00 | 1.00 |
