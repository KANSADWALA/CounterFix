# Confidence and perplexity

## Token confidence from the API

When a provider returns token log probabilities, counterfix keeps a summary on each
record (`record.metadata["logprobs"]`: token count, mean and lowest log probability, the
least likely tokens). Ask for them with `logprobs=True`:

```bash
counterfix eval qa.jsonl --model openai:gpt-4o-mini --param logprobs=true --metric exact_match --metric confidence
```

On a guarded call, read it from the response:

```python
import counterfix
from counterfix import confidence
from openai import OpenAI

client = counterfix.guard(OpenAI())
r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, logprobs=True)
confidence.mean_probability(r)  # 0.87, or None when the provider sent no logprobs
confidence.describe(r)  # 'token confidence 0.87 over 12 tokens; least likely: ...'
```

- The `confidence` metric is the answer's geometric-mean token probability, exp(mean
  logprob), from 0 to 1 (`confidence:0.5` fails answers below 0.5; with the option
  `use="min"` it is the least likely token's probability).
- When a judge confirms invented facts (F4), the token confidence is added to the
  evidence ("token confidence 0.41 over 23 tokens; least likely: '1889' 0.03").

Low confidence is a signal, not a verdict: models are often unsure and right, and sure
and wrong. counterfix never reports a failure from confidence alone.

| Provider | Logprobs |
| --- | --- |
| OpenAI Chat Completions | yes |
| Ollama 0.12.11 or later | yes (checked with qwen2.5) |
| vLLM, most OpenAI-compatible servers | usually |
| Groq | no (HTTP 400; counterfix drops the field and the metric is "not scored") |
| Gemini through its OpenAI-compatible endpoint | no (HTTP 400; Vertex AI's own API has them) |
| OpenAI Responses API, Anthropic | no |

`counterfix setup --test` asks each configured provider once (a 5-token request) and
prints "available" or "not available", so you know before a run.

## Corpus perplexity with a local model

Chat APIs do not give probabilities for the text you send, so perplexity of a corpus
needs the model's weights (`pip install "counterfix[semantic]"`):

```python
from counterfix.perplexity import perplexity

perplexity(texts, model="gpt2-large", stride=512).value
```

It uses the sliding-window method of the
[Hugging Face perplexity guide](https://huggingface.co/docs/transformers/perplexity):
windows of the model's context length move `stride` tokens at a time, and each window
scores only tokens no earlier window scored. `bench/perplexity_check.py` reproduces the
guide's GPT-2 large result on WikiText-2: measured 16.44 with stride 512, the guide's own
figure. On a small GPU, `dtype="float16"` halves the memory (that run used it).
