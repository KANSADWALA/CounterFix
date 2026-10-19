# Plug it in

Every plug runs the same free live checks and gives you a `Diagnosis`. Pick the one that
matches how your code calls the model.

## OpenAI-compatible clients

```python
client = counterfix.guard(OpenAI())  # OpenAI
client = counterfix.guard(OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key))
client = counterfix.guard(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
```

`response.diagnosis` holds the result (or `counterfix.get_diagnosis(response)` for
objects that refuse new attributes). Async clients work the same way.

## Anthropic SDK

```python
from anthropic import Anthropic

client = counterfix.guard(Anthropic())
message = client.messages.create(model="claude-haiku-5-5", max_tokens=1024, messages=[...])
```

`tool_use` arguments are checked against each tool's `input_schema`; thinking blocks are
kept as reasoning, so "the model spent its whole budget thinking" is caught as F1. A
`stop_reason` of `refusal` is reported as a provider filter, not as over-refusal (F3).

## Streams

`stream=True` works with both: every chunk reaches you the moment it arrives, and the
whole answer is checked when the stream ends (`stream.diagnosis`). Streams are never
retried, because the tokens are already with the user.

## LiteLLM and LangChain

```python
litellm.callbacks = [counterfix.litellm_callback()]
llm.invoke("...", config={"callbacks": [counterfix.langchain_callback()]})
```

Callbacks only observe; the latest result is on `handler.last_diagnosis`, and every result
goes to `Config.on_diagnosis`.

## Any function

```python
@counterfix.watch(model="llama3.2")
def ask(prompt: str) -> str:
    return my_local_model(prompt)
```

The first argument (or `prompt=` / `messages=`) is read as the request, the return value
(text, or an OpenAI-shaped response) as the answer. Sync and async both work.

## OpenTelemetry GenAI spans

Already instrumented (for example with OpenLLMetry)? Check the spans you have:

```python
provider.add_span_processor(counterfix.otel.SpanChecker())
```

## Telling counterfix what the model was given

```python
with counterfix.context(source=retrieved_text, reference="21 days"):
    response = client.chat.completions.create(...)
```

`source` lets the hallucination check tell "the source was missing" from "the source was
ignored", and lets `fix()` try the call with the source in the prompt. `reference` is a
known right answer. It works with every plug above.

## Settings

`Config` is shared by all plugs: `mode="retry"` (one automatic retry with a known fix),
`on_diagnosis=` (a hook per call, e.g. `counterfix.log_to("calls.jsonl")`), background
judge checks (see [In production](production.md)), and `enabled=False` to switch it off.
counterfix fails open: if it errors, your call still returns normally, and provider
errors reach your code unchanged.
