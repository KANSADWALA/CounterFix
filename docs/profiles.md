# Model profiles

A profile says how often a model fails each kind of request, and which fixes worked on
it. counterfix ships profiles for the models it was measured on, and you can make your own.

```bash
counterfix profile groq:qwen/qwen3.8-27b --judge          # run the 54-prompt probe suite
counterfix profile groq:qwen/qwen3.8-27b --judge --fix    # also learn which fixes work
```

Profiles are saved under `~/.counterfix/profiles` (or `$COUNTERFIX_HOME`). `fix()` loads
the profile for the model it is fixing and tries first what worked before, which saves
re-runs on known models.

## Compare two models, or a model upgrade

```bash
counterfix compare openai/gpt-oss-120b qwen/qwen3.8-27b        # two profiles
counterfix compare calls.jsonl --model groq:qwen/qwen3.8-27b   # re-run your real calls
```

The second form re-runs every logged call on the new model and reports each as *fixed*,
*still failing*, *new failure* or ok: a model upgrade report from your own traffic.

## Drift

```bash
counterfix drift groq:qwen/qwen3.8-27b --judge     # exit code 1 if a task moved 15+ points
```

It re-runs the probes, compares with the stored profile, and keeps the previous one as
`.prev.json`. Run it weekly in CI; `examples/drift-workflow.yml` is a GitHub Actions
workflow to copy (add your keys as repository secrets).

## Route

```bash
counterfix route cheap-model expensive-model --max-failure 0.2
```

For each task type, the cheapest model whose measured failure rate is at most 20%.
Models are taken in the order given, or by `cost_per_mtok` when the profiles have it
(`counterfix profile ... --cost 0.15`).

## Probe suite

54 short prompts, 6 per failure type, written for counterfix (no outside data). The F7
probes come with rewordings, and count as failed when a rewording flips the answer. Pass
`--probes your.jsonl` (same format as `counterfix/data/probes.jsonl`) to measure on your
own prompts instead; that is what you want before trusting a route.
