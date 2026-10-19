---
license: mit
language: [en]
task_categories: [text-classification]
pretty_name: counterfix diagnosed LLM failures
size_categories: [n<1K]
---

# counterfix diagnosed LLM failures (draft card)

*Draft for the planned public dataset (future direction 1). Not uploaded. Before release:
re-check every source (all prompts below are handwritten for counterfix, so no third-party
licence applies yet), and get the owner's go-ahead.*

## What it is

Each row is one LLM call that failed, with:

- the prompt (chat messages and parameters) and the model's answer,
- the failure type it was labelled with (F1 to F9) and how the label was made (automatic
  for objective cases, a reference label otherwise, adjudicated when judges disagreed),
- what counterfix found, the interventions it tried, how many re-runs each passed, and
  the verified fix when there was one.

## Sources

- Prompts: the 54-prompt counterfix seed set (`bench/seed_failures.jsonl`), written by the
  authors. No text is copied from other datasets.
- Answers: `openai/gpt-oss-120b` (Groq) and `qwen2.5:3b` (Ollama), October 2026. Check each
  provider's terms on publishing model outputs before release.
- Labels: `bench/labels/`, with the history of corrected labels in `bench/labels/history/`.
- Fix runs: `bench/results/fixes.jsonl`.

## Fields

`key`, `model`, `messages`, `params`, `answer`, `labels`, `label_source`, `found`,
`retry` (passed, runs), `attempts` (name, passed, runs), `fix`, `calls`.

## Known limits

Small (108 labelled answers, 38 failures); English only; two target models; labels are
partly model-made (reference labels), with 9 corrected after review. Treat rates as rough.
