# Fix with your own test

You may already test your LLM's answers: a function you wrote, a DeepEval metric, a Ragas
score. Pass that test to `fix()` and counterfix uses it to judge every re-run:

```python
import counterfix


def mentions_refund_window(record):
    return "30 days" in record.answer_text


result = counterfix.fix(response, check=mentions_refund_window)
print(result.explain())
```

## Who decides what

| Question | Who answers it |
| --- | --- |
| Did this call fail? | **Your check** |
| Why did it fail? | counterfix's diagnosis (F1 to F14) |
| Which change to try? | counterfix, from the cause it found |
| Did a re-run pass? | **Your check**, and no new failure (a cut-off answer, broken JSON...) may appear |
| Is the change really better than retrying? | counterfix's statistics: it must beat a plain retry |

So your tool keeps deciding what "good" means, and counterfix adds what your tool does not
do: it finds the cause, tries one change at a time, and proves the change works.

## What happens in each case

- **Your check passes on the call:** nothing is re-run; `stopped` says "your check passes
  on this call: nothing to fix".
- **Your check fails and counterfix finds a cause:** the normal search runs, with your
  check as the pass test. `explain()` shows `pass test: your check <name>`.
- **Your check fails but counterfix finds no cause:** nothing is re-run, and `stopped`
  says so. A targeted change needs a cause; counterfix will not guess one. Without a
  judge, only the live checks ran: pass `judge=...` (or set `GROQ_API_KEY` or
  `GEMINI_API_KEY`) so facts and premises are checked too.
- **Your check raises on the original call:** you get a `CheckError` with your error
  attached, because that is a bug in the check. If it raises on a re-run, that run counts
  as a fail and the error is shown in `explain()`.

You do not need a judge to confirm a fix when you pass a check: your check is the judge.

## What a check returns

A check takes the `CallRecord` of an answer (`record.answer_text`, `record.prompt_text`,
`record.messages`, ...) and returns one of:

- `True` or `False`;
- a pair `(passed, reason)`, and the reason is shown when the check fails;
- an object with a boolean `.passed`;
- or pass a counterfix `Metric` object itself, with a threshold and no reference needed.

A score is not a verdict: return `score >= 0.8`, not `0.8`. counterfix refuses a bare
number, so a check that forgets the threshold never counts as a pass.

## With DeepEval

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

relevancy = AnswerRelevancyMetric(threshold=0.7)


def relevant(record):
    case = LLMTestCase(input=record.prompt_text, actual_output=record.answer_text)
    relevancy.measure(case)
    return relevancy.is_successful(), relevancy.reason


result = counterfix.fix(response, check=relevant)
```

## With Ragas

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

faithfulness = Faithfulness(llm=my_ragas_llm)


def faithful_to(chunks):
    def check(record):
        sample = SingleTurnSample(
            user_input=record.prompt_text,
            response=record.answer_text,
            retrieved_contexts=chunks,
        )
        score = faithfulness.single_turn_score(sample)
        return score >= 0.8, f"faithfulness {score:.2f}"

    return check


result = counterfix.fix(response, check=faithful_to(retrieved_chunks))
```

Both run an LLM to score, so every re-run costs one scoring call too. Keep `max_runs`
(30 by default) in mind.

## From the command line

Put the check in a file and name it with `--check`:

```python
# checks.py
def mentions_refund_window(record):
    return "30 days" in record.answer_text
```

```bash
counterfix fix calls.jsonl --check checks.py:mentions_refund_window --caller groq:qwen/qwen3.8-27b
```

With `--check`, your check picks the failed calls in the log (calls it passes are
skipped), and the run reports which of them got a verified fix.

For a whole dataset, use `counterfix.evaluate(..., metrics=[YourMetric()], fix=N)`: see
[Evaluate a dataset](evals.md) for writing a `Metric`.
