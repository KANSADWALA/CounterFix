# Safety: prompt injection, personal data, toxicity

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

## Personal data in answers

The `pii` metric counts personal data in the answer that was not already in the prompt
(an answer that repeats the user's own email back is not a leak):

```bash
counterfix eval support.jsonl --model groq:openai/gpt-oss-120b --metric pii
```

Built in, with nothing to install: email addresses, phone numbers, payment card numbers
(Luhn-checked), IBANs (checksum), US social security numbers, Aadhaar numbers (Verhoeff
checksum), IPv4 addresses and API keys (OpenAI, Anthropic, Groq, Google, AWS, GitHub,
Hugging Face). Reports mask what they found. With `pip install "counterfix[safety]"`
(Python 3.11+), the option `engine="presidio"` adds Microsoft Presidio (names, places and
more ID formats). In Python: `from counterfix.evals.safety import find_pii`, then `find_pii(text)`.

## Toxicity

The `toxicity` metric is the highest Detoxify score of the answer (toxicity, insult,
threat, obscenity, identity attack), 0 to 1; it fails at 0.5 or above by default
(`toxicity:0.3` is stricter). Needs `pip install "counterfix[safety]"`.
