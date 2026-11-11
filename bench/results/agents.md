# F14 agent-task check against labelled runs

Generated 2026-11-11 09:47 UTC by `bench/eval_agents.py` on bench/agent_seed.jsonl (20 runs written for counterfix: 10 completed, 10 not). Positive class: not completed. 20 runs is a small set: read the intervals.

## groq:qwen/qwen3.8-27b

- precision (said not completed, was not): 10/10 = 1.00 (0.72 to 1.00)
- recall (not completed, and caught): 10/10 = 1.00 (0.72 to 1.00)
- exact label: 20/20 = 1.00 (0.84 to 1.00)
- cause named right, of the failed runs caught: 9/10 = 0.90 (0.60 to 0.98)
- not judged (errors): 0

Disagreements (label -> judged): n-01 (not_completed/wrong_tool -> not_completed/missing_step)
