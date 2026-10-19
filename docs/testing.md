# Prompt regression tests

Keep fixed failures fixed: every verified fix can become a pytest case, so a prompt edit
or a model upgrade cannot quietly bring the failure back.

## In your own tests

```python
from counterfix.testing import assert_ok


def test_summary_prompt():
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    assert_ok(response)  # fails with the explanation if F1-F9 is found
    assert_ok(response, judge=True)  # also judge checks
```

## From verified fixes

```bash
counterfix fix calls.jsonl --judge --out fixes.jsonl
counterfix export-tests fixes.jsonl --out tests/test_prompt_regressions.py
pytest tests/test_prompt_regressions.py
```

`export-tests` writes the test file and a `.json` file of cases beside it. Each case is a
call that once failed, with its verified fix applied. It passes while the failure stays
gone in 2 of 3 runs.

| Option | Effect |
| --- | --- |
| `--counterfix-judge [SPEC]` | add judge checks (live checks only by default) |
| `--counterfix-runs N` | runs per case (default 3; a majority must pass) |
| `--counterfix-caller SPEC` | the model to run cases on (or set `COUNTERFIX_CALLER`) |
| `--counterfix-strict` | fail cases that cannot run (on by default when `CI` is set) |

Cases re-run on the provider named in the case, with keys from the environment or a
`.env` file. For another model, or your own server, name it:

```bash
pytest tests/test_prompt_regressions.py --counterfix-caller groq:qwen/qwen3.8-27b
pytest tests/test_prompt_regressions.py --counterfix-caller my-model@http://localhost:8000/v1
```

`openai:MODEL` also works and honours `OPENAI_BASE_URL`; the `MODEL@URL` form uses
`OPENAI_API_KEY` if it is set. A case that cannot run (no key, or it came from a custom
server and no caller was named) is skipped on your machine, but **fails in CI**, so a
green build means the cases really ran.

To run them through your own client object instead, override the fixture:

```python
# conftest.py
import pytest
from counterfix.core.caller import OpenAICompatibleCaller


@pytest.fixture
def counterfix_caller():
    return OpenAICompatibleCaller(my_client, "gpt-4o-mini")
```

The plugin registers itself when counterfix is installed; it only adds these options and
the `counterfix_caller`, `counterfix_judge`, `counterfix_runs` and `counterfix_strict`
fixtures.
