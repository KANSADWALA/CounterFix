# Counterfactual engine on the seed failures

38 labelled failures (0 errors). Judges: groq:qwen/qwen3.8-27b, gemini:gemini-3.5-flash-lite (panel). A run passes when the failure is gone with no new failure; a change counts as a fix only if it passes clearly more often than a plain retry (one-sided Fisher exact test, see counterfix.verify).

| Measure | Result |
| --- | --- |
| Failure detected | 30/38 (79%) |
| Top cause matches the label | 27/38 (71%) |
| Plain retry already passes (no fix claimed) | 7/38 (18%) |
| Verified fix, cause right, retry did not pass | 14/22 (64%) |
| Verified fix (any) | 14/38 (37%) |
| Re-runs of the target model | 259 (6.8 per failure) |

| Type | Failures | Retry passes | Verified fix |
| --- | --- | --- | --- |
| F1 | 12 | 0 | 11 |
| F2 | 1 | 0 | 1 |
| F3 | 3 | 1 | 0 |
| F4 | 14 | 4 | 0 |
| F5 | 1 | 0 | 1 |
| F6 | 1 | 1 | 0 |
| F8 | 5 | 1 | 0 |
| F9 | 1 | 0 | 1 |

Fixes that verified: raise_max_tokens_more 11, json_only_prompt 1, check_premises 1, add_detail 1

| Item | Label | Found | Retry | Tried | Fix |
| --- | --- | --- | --- | --- | --- |
| openai-gpt-oss-120b:F1-001 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| qwen2.5-3b:F1-001 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-002 | F1 | F1 | 0/6 | raise_max_tokens_more 3/5; ask_shorter 0/2; raise_max_tokens 0/2 | - |
| openai-gpt-oss-120b:F1-003 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| qwen2.5-3b:F1-002 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| qwen2.5-3b:F1-003 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-004 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| qwen2.5-3b:F1-004 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-005 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| openai-gpt-oss-120b:F1-006 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| openai-gpt-oss-120b:F3-002 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F3-005 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F4-002 | F4 | F4 | 1/6 | allow_unknown 0/1 | - |
| openai-gpt-oss-120b:F4-003 | F4 | F4 | 2/2 | - | - |
| openai-gpt-oss-120b:F4-004 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F4-006 | F3 | F3 | 3/3 | - | - |
| openai-gpt-oss-120b:F5-001 | F4 | - | - | - | - |
| openai-gpt-oss-120b:F6-003 | F4 | - | - | - | - |
| qwen2.5-3b:F1-005 | F1 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| qwen2.5-3b:F1-006 | F1,F4 | F1 | 0/6 | raise_max_tokens_more 5/5 | raise_max_tokens_more |
| qwen2.5-3b:F2-002 | F2 | F2 | 0/6 | json_only_prompt 5/5 | json_only_prompt |
| qwen2.5-3b:F8-002 | F8 | F8 | 0/6 | work_it_out 0/3 | - |
| qwen2.5-3b:F8-003 | F8 | F8 | 2/6 | work_it_out 1/2 | - |
| qwen2.5-3b:F8-004 | F8 | F8 | 0/6 | work_it_out 0/3 | - |
| qwen2.5-3b:F8-005 | F8 | F8 | 3/6 | - | - |
| qwen2.5-3b:F8-006 | F8 | F8 | 2/6 | work_it_out 0/1 | - |
| qwen2.5-3b:F5-006 | F4 | F5 | 3/6 | - | - |
| qwen2.5-3b:F5-004 | F4 | F4 | 2/5 | - | - |
| qwen2.5-3b:F3-003 | F3 | - | - | - | - |
| qwen2.5-3b:F3-002 | F4 | - | - | - | - |
| qwen2.5-3b:F6-004 | F4 | F6,F4 | 2/2 | - | - |
| qwen2.5-3b:F3-005 | F4 | F5 | 2/6 | check_premises 0/1 | - |
| qwen2.5-3b:F3-006 | F3 | F3 | 2/6 | state_purpose 0/1 | - |
| qwen2.5-3b:F5-001 | F5 | F5 | 0/6 | check_premises 4/6 | check_premises |
| qwen2.5-3b:F6-006 | F6 | F6 | 3/4 | - | - |
| qwen2.5-3b:F9-001 | F9 | F9 | 0/6 | add_detail 5/5 | add_detail |
| qwen2.5-3b:F9-003 | F4 | - | - | - | - |
| qwen2.5-3b:F9-005 | F4 | F4 | 1/6 | allow_unknown 0/1 | - |
