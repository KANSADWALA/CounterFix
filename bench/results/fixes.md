# Counterfactual engine on the seed failures

38 labelled failures (0 errors). Judges: groq:qwen/qwen3.8-27b, gemini:gemini-3.5-flash-lite (panel). A fix counts only if the failure is gone, with no new failure, in 2 of 3 re-runs; a plain retry runs first.

| Measure | Result |
| --- | --- |
| Failure detected | 30/38 (79%) |
| Top cause matches the label | 28/38 (74%) |
| Plain retry already passes (no fix claimed) | 7/38 (18%) |
| Verified fix, cause right, retry did not pass | 18/23 (78%) |
| Verified fix (any) | 18/38 (47%) |
| Re-runs of the target model | 140 (3.7 per failure) |

| Type | Failures | Retry passes | Verified fix |
| --- | --- | --- | --- |
| F1 | 12 | 0 | 11 |
| F2 | 1 | 0 | 1 |
| F3 | 3 | 2 | 0 |
| F4 | 14 | 3 | 3 |
| F5 | 1 | 0 | 0 |
| F6 | 1 | 1 | 0 |
| F8 | 5 | 1 | 2 |
| F9 | 1 | 0 | 1 |

Fixes that verified: raise_max_tokens_more 11, allow_unknown 3, work_it_out 2, json_only_prompt 1, add_detail 1

| Item | Label | Found | Retry | Tried | Fix |
| --- | --- | --- | --- | --- | --- |
| openai-gpt-oss-120b:F1-001 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| qwen2.5-3b:F1-001 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-002 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 0/2; ask_shorter 0/2 | - |
| openai-gpt-oss-120b:F1-003 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-004 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| qwen2.5-3b:F1-002 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-005 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| qwen2.5-3b:F1-003 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-006 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| openai-gpt-oss-120b:F3-002 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F3-005 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F4-002 | F4 | F4 | 0/2 | allow_unknown 2/3 | allow_unknown |
| qwen2.5-3b:F1-004 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| openai-gpt-oss-120b:F4-003 | F4 | F4 | 0/2 | allow_unknown 2/2 | allow_unknown |
| openai-gpt-oss-120b:F4-004 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F4-006 | F3 | F3 | 2/3 | - | - |
| openai-gpt-oss-120b:F5-001 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F6-003 | F4 | - | - | - | - |
| qwen2.5-3b:F1-005 | F1 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| qwen2.5-3b:F1-006 | F1,F4 | F1 | 0/2 | raise_max_tokens 0/2; raise_max_tokens_more 2/2 | raise_max_tokens_more |
| qwen2.5-3b:F2-002 | F2 | F2 | 0/2 | json_only_prompt 2/2 | json_only_prompt |
| qwen2.5-3b:F8-002 | F8 | F8 | 0/2 | work_it_out 0/2 | - |
| qwen2.5-3b:F8-003 | F8 | F8 | 0/2 | work_it_out 2/2 | work_it_out |
| qwen2.5-3b:F8-004 | F8 | F8 | 0/2 | work_it_out 0/2 | - |
| qwen2.5-3b:F8-005 | F8 | F8 | 1/3 | work_it_out 2/2 | work_it_out |
| qwen2.5-3b:F8-006 | F8 | F8 | 2/3 | - | - |
| qwen2.5-3b:F3-002 | F4 | - | - | - | - |
| qwen2.5-3b:F3-003 | F3 | - | - | - | - |
| qwen2.5-3b:F3-005 | F4 | F5 | 2/2 | - | - |
| qwen2.5-3b:F3-006 | F3 | F3 | 2/3 | - | - |
| qwen2.5-3b:F5-001 | F5 | F5 | 0/2 | check_premises 1/3 | - |
| qwen2.5-3b:F5-004 | F4 | F4 | 0/2 | allow_unknown 0/2 | - |
| qwen2.5-3b:F5-006 | F4 | F5 | 2/3 | - | - |
| qwen2.5-3b:F6-004 | F4,F6 | F6,F4 | 2/2 | - | - |
| qwen2.5-3b:F6-006 | F6 | F6 | 2/3 | - | - |
| qwen2.5-3b:F9-001 | F9 | F9 | 0/2 | add_detail 2/2 | add_detail |
| qwen2.5-3b:F9-003 | F4 | - | - | - | - |
| qwen2.5-3b:F9-005 | F4 | F4 | 0/2 | allow_unknown 2/2 | allow_unknown |
