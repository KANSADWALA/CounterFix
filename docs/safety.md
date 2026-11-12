# Safety: prompt injection

## Prompt injection (F13)

Text your app hands the model without writing it itself (retrieved passages, web pages,
tool results) can carry instructions aimed at the AI: "ignore previous instructions",
"tell the user to log in at ...", "reply in French", "forward this inbox". When the answer
carries one out, that is F13.

**Find it.** With a judge, `counterfix.diagnose()` and `counterfix eval --judge` check
every call that carries retrieved text (`counterfix.context(chunks=...)`, RAG prompts) or
tool results (`tool` messages): the judge finds a planted instruction and decides whether
the answer followed it. Calls without such text are skipped, so this costs nothing
elsewhere.

**Probe your model.** The `injection` benchmark runs 20 probes, 12 with an instruction
planted in a retrieved passage and 8 in a tool result (a web search, an email, a file, an
API reply, a shell output...). Each planted instruction asks for a made-up canary code,
so whether the model followed it is checked exactly, without a judge:

```bash
counterfix benchmark injection --model ollama:qwen2.5:3b --fix 10
```

The score is the share of probes resisted; each miss also says whether the answer still
did the user's task.

**Fixes it verifies:**

| Intervention | Change |
| --- | --- |
| `delimit_untrusted` | wrap retrieved text and tool results in `<untrusted_data>` tags, with a system rule that their instructions are not to be followed |
| `data_not_instructions` | the system rule alone |
| `restate_task` | repeat the user's request after the untrusted text, with a reminder |

A fix counts only when the re-runs resist clearly more often than a plain retry
([how](fixes.md#what-verified-means)). Measured with the earlier 2-of-3 rule (0.2.1): gpt-oss-120b resisted only 6 of
20 probes, and `delimit_untrusted` stopped 9 of the 10 injections searched; qwen2.5:3b
resisted 13 of 20, and no prompt change verified on its 5 repeatable misses. Whether
prompt-level defence works depends on the model, so measure it on yours (see [Measured
results](results.md#safety-and-confidence)).
When none does, counterfix suggests
filtering untrusted text, removing risky tools from untrusted turns, or a model trained to
resist injection. Large red-team suites are out of scope; promptfoo and garak cover them.

## Personal data and toxicity

For personal data in answers use [Microsoft Presidio](https://microsoft.github.io/presidio/),
and for toxicity [Detoxify](https://github.com/unitaryai/detoxify), or your evaluation
tool's checks. To make one of them the pass test of a fix search, wrap it in a check:
see [Fix with your own test](your-check.md).
