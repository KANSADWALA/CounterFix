# Diagnose a call

```python
report = counterfix.diagnose(response)  # a guarded response, its Diagnosis, or a CallRecord
print(report.explain())
```

## Live checks and judge checks

| | Live checks | Judge checks |
| --- | --- | --- |
| Run | On every call | When you ask |
| Extra calls | None | About 6 judge calls per diagnosis |
| Catch | F1, F2, F3 (phrases), F6 (exact rules), empty output, provider filter | F3 confirmation, F4, F5, F6 (content), F8, F9 |

`counterfix.diagnose(x, judge=False)` runs only the live checks. Live F6 checks are
exact, in code: sentence, word and bullet counts, letter case, reply language, British or
American spelling, banned words, "end with a question", "only the number" and similar.

## One mistake, one cause

A wrong calculation is reported as F8, not also as an invented fact (F4); a cut-off answer
as F1, not also as dropped instructions (F6). Findings set aside are kept in
`report.suppressed`.

The F8 check never trusts the judge's arithmetic: the judge writes one expression and
counterfix evaluates it with a small whitelist-only evaluator.

## Bring your own judge

counterfix never ships keys. The judge is chosen in this order:

1. `counterfix.diagnose(x, judge="groq:qwen/qwen3.8-27b")`
2. `counterfix.configure(judge=...)` once per process; a list of specs makes a voting panel
3. the `COUNTERFIX_JUDGE` environment variable
4. a key it finds: `GROQ_API_KEY`, `GEMINI_API_KEY` (both free), `OPENAI_API_KEY`,
   `OPENROUTER_API_KEY`, or a local Ollama model; another model family is preferred
5. otherwise the checked model judges itself, with a `SelfJudgeWarning`

`counterfix setup` shows what it found (key values are never printed); `--test` sends the
judge one tiny question. A local Ollama judge keeps private prompts on your machine.

How accurate are the judges? See [Measured results](results.md).
