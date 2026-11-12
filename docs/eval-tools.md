# DeepEval, Ragas and Promptfoo

Already testing with one of these? Keep doing so. Point counterfix at the results file
your tool saves, and it explains each failure (which of F1 to F14 caused it) and searches
for a fix that verifiably works:

```bash
counterfix diagnose results.json          # why each failed test case failed
counterfix fix results.json               # find and prove a fix for each
```

The tool is detected from the file (or say it: `--from promptfoo`, `--from deepeval`,
`--from ragas`). Only the test cases your tool failed are looked at.

## What each file allows

The files hold different things, and that decides what counterfix can do with them:

| | Promptfoo | DeepEval | Ragas |
| --- | --- | --- | --- |
| Diagnose the failures | yes | yes | yes, with `--threshold` |
| The exact prompt that was sent | yes | no: only the test input | no: only the user input |
| Which model served it | yes (provider id) | if you recorded it | no |
| Re-run for a fix | yes, as it was sent | with `--caller` (and `--system`) | with `--caller` (and `--system`) |
| Your tool's verdict on re-runs | deterministic assertions re-checked | no: use `--check` | no: use `--check` |

**Promptfoo** saves the rendered prompt, the provider and its settings (temperature,
`max_tokens`, ...) and the finish reason, so the call is re-run exactly as it was sent.
Its deterministic assertions (`equals`, `contains`, `icontains`, `contains-any`,
`contains-all`, `starts-with`, `regex`, `is-json`, and their `not-` forms) are re-checked
on every re-run, so your assertions decide which re-runs pass. Model-graded assertions
(`llm-rubric`, `factuality`, ...), `javascript` and `python` need Promptfoo itself: for
those test cases counterfix says so and judges re-runs with its own checks, or with your
`--check`.

**DeepEval and Ragas** save the test input, the answer, the retrieved context and the
scores, but not the full prompt your app built or the model it called. counterfix can
diagnose these as they are (the retrieved context and expected output are used by the
judge checks). To re-run one, say which model with `--caller`, and pass your app's system
prompt with `--system` (text or a file) if it has one. Without that, the re-run is the
bare question, not the call your app makes, and a fix found that way may not carry over.
Their metrics are Python objects that are not in the file, so to keep them as the pass
test, wrap one in a check (see [Fix with your own test](your-check.md)) and pass `--check`.

When counterfix finds no cause for a failure, it says so and tries nothing: a targeted
change needs a cause. Add `--judge` to check facts, premises and context use too.

## Promptfoo

```bash
promptfoo eval -o results.json
counterfix diagnose results.json
counterfix fix results.json --out fixes.jsonl
```

Provider ids map to counterfix models: `openai:...`, `groq:...`, `ollama:...`,
`openrouter:...`, `anthropic:...` and `google:...` (Gemini). A provider with an
`apiBaseUrl` is re-run at that address. Other providers (Azure, Bedrock, `file://`,
`python:`, HTTP) need `--caller`, e.g. `--caller groq:llama-3.1-8b-instant` or
`--caller MODEL@http://host/v1`.

## DeepEval

Save the run with a results folder, either with the environment variable or in code:

```bash
DEEPEVAL_RESULTS_FOLDER=./deepeval-results deepeval test run test_app.py
```

(PowerShell: `$env:DEEPEVAL_RESULTS_FOLDER="deepeval-results"; deepeval test run test_app.py`.)

```python
from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig

evaluate(
    test_cases,
    metrics,
    hyperparameters={"model": "llama-3.1-8b-instant"},
    display_config=DisplayConfig(results_folder="deepeval-results"),
)
```

Then:

```bash
counterfix diagnose deepeval-results/test_run_20261112_174940.json --judge
counterfix fix deepeval-results/test_run_20261112_174940.json \
    --caller groq:llama-3.1-8b-instant --system prompts/system.txt
```

A `.deepeval/.latest_test_run.json` file, when you have one, is read too. The model
named in `hyperparameters={"model": ...}` or in a test case's `metadata={"model": ...}` is
shown in the report; re-running still needs `--caller`, because the file does not say
which provider served it.

## Ragas

Ragas saves scores without pass/fail, so give a pass mark for each score you want judged:
`NAME=0.8` means 0.8 or more passes, `NAME<=0.3` means 0.3 or less passes (for scores
where lower is better).

```python
result = evaluate(dataset, metrics=[...])
result.to_pandas().to_csv("ragas.csv", index=False)  # or .to_json(..., orient="records")
```

```bash
counterfix diagnose ragas.csv --threshold faithfulness=0.8 --threshold answer_relevancy=0.7 --judge
counterfix fix ragas.csv --threshold faithfulness=0.8 --caller groq:llama-3.1-8b-instant
```

A score Ragas could not compute (empty cell) is reported as not scored, not as a failure.
Older column names (`question`, `answer`, `contexts`, `ground_truth`) are read too.

## In Python

```python
from counterfix.integrations.eval_results import read_results

for case in read_results("results.json"):
    if case.failed:
        print(case.name, case.reason)
        result = counterfix.fix(case.record, caller=..., check=case.check)
```

Each case has the `record` (a `CallRecord`), the tool's verdict (`failed`, `reason`),
`rerun` (a model spec when the file names one), `check` (the re-checkable assertions, or
None) and `notes` (what could not be carried over).
