# Speed and cost

Every guarded call records its latency and token counts. Streamed calls also record when
the pieces arrived, on `record.timing`:

| Field | Meaning |
| --- | --- |
| `ttft_ms` | time to the first chunk of visible answer text (what a user sees first) |
| `first_reasoning_ms` | time to the first chunk of reasoning, for models that stream it |
| `last_chunk_ms`, `chunks` | time to the last answer chunk, and how many chunks carried text |
| `mean_chunk_gap_ms` | average time between answer chunks (inter-chunk latency) |

Nothing is delayed: chunks reach your code the moment they arrive.

## Percentiles per model

`counterfix diagnose calls.jsonl` adds a speed table per model to the report (and the HTML
report gets the same table):

```text
[groq:openai/gpt-oss-120b] 120 call(s), 1 error(s); slow tail in the last two columns
                       p50       p95       p99
  latency              640      1910      2805  ms (n=119)
  TTFT                 210       480       702  ms (n=40)
  output speed         412       233       201  tokens/s (n=119, columns p50/p5/p1)
```

For latency and TTFT the slow tail is the high percentiles; for output speed it is the low
ones (p5, p1), so every row's last two columns show the slowest calls. (Numbers above are
an illustration of the layout.)

From Python: `counterfix.perf.speed_summary(records)`; group with `perf.by_model(records)`.
Output speed counts from the first chunk on streamed calls, so time spent waiting for the
model to start is not counted. It needs token counts: OpenAI-compatible streams send them
only when asked, with `stream_options={"include_usage": True}`.

## Cost

counterfix ships no prices: they change and differ by account. Tell it what you pay per
million tokens, then reports and summaries include cost:

```python
counterfix.perf.set_prices({"groq:openai/gpt-oss-120b": (0.15, 0.60)})  # (input, output)
```

Calls of models without a price are counted as unpriced, never guessed.

## OpenTelemetry metrics

```python
config = counterfix.Config(
    on_diagnosis=counterfix.hooks(counterfix.log_to("calls.jsonl"), counterfix.otel.metrics())
)
```

records `gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk`
(streamed calls) and `gen_ai.client.token.usage`, in seconds and tokens, with
`gen_ai.request.model` and `gen_ai.provider.name` attributes, following the OpenTelemetry
GenAI semantic conventions. `counterfix.hooks(...)` runs several hooks in order.

## Verified speed fixes

```python
result = counterfix.perf.speed_fix(record, target_ms=3000, alternatives=[faster_caller])
print(result.explain())
```

```text
plain retry: under 3000 ms in 0/2, median 7950 ms -> no
tried lower_reasoning_effort: under 3000 ms in 2/2, median 2100 ms -> VERIFIED
fix: Lower the reasoning effort (reasoning_effort='low')
  change: {'reasoning_effort': 'low'}
4 re-run(s); stopped at the first verified fix
```

The same rules as the fix engine: a plain retry first (a slow call that is fast on retry
was a one-off), then one change at a time: lower reasoning effort (reasoning models), a
concise-answer instruction, then each model in `alternatives`. A change counts only when
the call is under the target in 2 of 3 re-runs **and** the live checks find no new failure,
so a fast answer that is cut off is not a fix.

In an eval, gate on speed with the `latency:MS` and `ttft:MS` metrics.
