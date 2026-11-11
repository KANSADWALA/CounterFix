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

## OpenAI Responses API

OpenAI recommends the Responses API for new projects; a guarded client checks it too:

```python
client = counterfix.guard(OpenAI())
r = client.responses.create(model="gpt-5-mini", instructions="Be brief.", input="...")
r.diagnosis.summary()
```

`responses.create` and `responses.parse`, sync or async, streamed or not, are checked. The
call is recorded in chat terms (`instructions` becomes the system message,
`max_output_tokens` the token limit, `text.format` the JSON schema, function tools and
their outputs become tool calls), so every check, judge and `fix()` works as for a chat
call. A response with `status="incomplete"` and reason `max_output_tokens` is F1. Retry
mode changes Responses fields (`max_output_tokens`, `text.format`). Known limit: with
`previous_response_id` or `conversation` the earlier turns live on OpenAI's servers, so
the record sees only the new input (it is marked `metadata["server_state"]`), and `fix()`
re-runs through Chat Completions.

Newer OpenAI models refuse `max_tokens` and a non-default `temperature` on Chat
Completions. Re-runs and judges notice the refusal once and resend with
`max_completion_tokens` (and without `temperature`), and fixes raise whichever limit
field your call used.

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

`stream=True` works with all three (chat, Responses, Anthropic): every chunk reaches you the moment it arrives, and the
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
