# Profile ordering on held-out failures

6 truncated answers from `ollama:qwen2.5:3b` (max_tokens 64), prompts not in the seed set. Live checks only. Same failures fixed twice: default intervention order, then the order from the model's shipped profile.

| | Re-runs per failure | Verified fixes |
| --- | --- | --- |
| Default order | 6.0 | 6/6 |
| Profile order | 4.0 | 6/6 |

| Prompt | Default | Profile | Fix (profile order) |
| --- | --- | --- | --- |
| Describe the water cycle in detail, covering every | 6 | 4 | raise_max_tokens_more |
| Write a step-by-step beginner's guide to baking so | 6 | 4 | raise_max_tokens_more |
| Explain how a bill becomes a law in India, from dr | 6 | 4 | raise_max_tokens_more |
| List twenty common Python built-in functions, each | 6 | 4 | raise_max_tokens_more |
| Write a short story of about 500 words about a lig | 6 | 4 | raise_max_tokens_more |
| Compare five popular databases (PostgreSQL, MySQL, | 6 | 4 | raise_max_tokens_more |
