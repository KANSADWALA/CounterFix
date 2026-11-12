# In production

The common settings are keyword shortcuts on `guard` (and `watch`), so most setups need
no `Config` object:

```python
client = counterfix.guard(
    OpenAI(),
    log="calls.jsonl",  # every call and its diagnosis
    sample_rate=0.05,  # background judge checks (see below)
    judge="groq:qwen/qwen3.8-27b",  # optional; otherwise picked from your keys
    mode="retry",  # re-run a flagged call once with a known fix
)
```

For the other settings (hourly cap, hooks, rules) pass a `counterfix.Config`; shortcuts
given next to it override the matching fields.

## Log every call

```python
client = counterfix.guard(OpenAI(), log="calls.jsonl")
# same as: guard(OpenAI(), config=counterfix.Config(on_diagnosis=counterfix.log_to("calls.jsonl")))
```

Each line is `{"record": ..., "diagnosis": ...}`: the call and what the live checks found.

## Reports

```bash
counterfix diagnose calls.jsonl                      # terminal summary (free)
counterfix diagnose calls.jsonl --judge --html report.html --json results.jsonl
counterfix diagnose calls.jsonl --fail-on-failure    # exit code 1 if any call failed
```

The HTML report is one self-contained file to send to a teammate: totals per failure
type, then each call with its prompt, answer, why it was flagged and what to try.

## Judge checks on live traffic, with a cost cap

```python
config = counterfix.Config(
    sample_rate=0.05,  # every flagged call + 5% of the rest
    max_judged_per_hour=60,  # calls over the cap are skipped, never delayed
    judge="groq:qwen/qwen3.8-27b",
    on_judged=counterfix.log_to("judged.jsonl"),
)
```

Judge checks run on one background thread: your call returns as fast as before. A full
queue drops new calls rather than waiting; `counterfix.background.stats` counts what was
judged and skipped. Call `counterfix.background.flush()` before a short script exits.

## Automatic retry

`Config(mode="retry")` re-runs a failed call once with a known parameter fix (for example
`max_tokens` raised 4x and to at least 1,024: a limit is a ceiling, so a short answer costs
no more), keeps whichever response is better, and records the attempt in
`diagnosis.retries`. It never raises if the retry fails. Add your own per-model fixes with
`RuleStore` and `FixRule`.

## OpenTelemetry

```python
config = counterfix.Config(on_diagnosis=counterfix.otel.export())
```

Each diagnosis becomes a short `counterfix.diagnosis` span under your request span, with
`gen_ai.*` attributes and `counterfix.failures`, `counterfix.top`,
`counterfix.top.confidence` and `counterfix.top.fix`. Needs `pip install "counterfix[otel]"`.

## Privacy

No telemetry. Logs stay where you write them. Judge checks send the prompt and answer to
the judge you chose; use a local Ollama judge to keep them on your machine.
