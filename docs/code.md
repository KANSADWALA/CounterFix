# Code evaluation

Score code a model wrote against tests, see why it failed (F12), and re-run candidate
fixes to find one that makes the tests pass.

```bash
counterfix eval coding.jsonl --model groq:openai/gpt-oss-120b --metric tests --allow-code --fix 5
```

Each row holds the prompt and the tests, in any of these forms (the `reference` field):

- HumanEval style: `{"prompt": "def add(a, b):\n    ...", "entry_point": "add", "test": "def check(candidate): ..."}`
- MBPP style: `{"test_list": ["assert add(1, 2) == 3", ...]}`, optional `test_setup_code`
- a list of `assert` lines, or test code as a string

## Running model-written code

Code a model wrote is untrusted, so nothing runs until you turn it on: `--allow-code`,
`COUNTERFIX_ALLOW_CODE=1`, or `allow=True` in Python. Then it runs in one of two sandboxes
(`--sandbox`, or `COUNTERFIX_SANDBOX`):

| Sandbox | What it does | Use it for |
| --- | --- | --- |
| `docker` | throw-away container: no network, 1 GB memory, 1 CPU, 128 processes, read-only file system, the program mounted read-only | anything you would not run on your own machine |
| `subprocess` | a fresh isolated Python in an empty folder, killed at the time limit, no keys or working proxies in its environment; on Linux and macOS also memory, CPU and file-size caps | benchmarks you trust (as HumanEval's own harness does); **not** a security boundary |
| `auto` (default) | Docker when it answers, else the subprocess | |

On Windows, Docker Desktop (or WSL) gives you the `docker` sandbox.

## Why it failed: F12 and its causes

The checks run cheapest first, so most failures are named without running anything:

| Cause | Found by |
| --- | --- |
| no code | no code block and nothing that looks like code |
| syntax | `ast.parse` fails (when the answer hit its token limit this is F1, cut off) |
| name / signature | the function the tests call is missing, or takes other arguments than the prompt's stub |
| import | a module outside the standard library and your `allowed_imports` |
| runtime | an exception that is not a failed `assert`, or the program dies before the tests |
| wrong output | a failed `assert`; the report names the line and what the code returned |
| timeout | over the time limit (default 10 s) |

Each test runs on its own, so MBPP-style items report "2/3 tests passed" and the first
failing assert. The report groups failed code by cause.

## Fixes it verifies

| Intervention | Change |
| --- | --- |
| `error_feedback` | send the error or failed assert back once (with what the code returned, e.g. "got 3"), with the model's own code, and ask for a fix |
| `check_examples` | ask the model to work the task's examples by hand before writing the code (wrong output, runtime errors) |
| `restate_signature` | restate the exact function name and signature the tests call |
| `one_code_block` | ask for the whole program in one code block |

Cut-off code gets the F1 fixes (raise the token limit). As everywhere in counterfix, a fix
counts only when the re-runs pass the tests clearly more often than a plain retry
([how](fixes.md#what-verified-means)).

## pass@k

`--samples N` (or `samples=N`) asks each question N times and reports pass@k for k in 1,
5, 10 and 100 (up to N), with the unbiased estimator from the HumanEval paper. Set a
temperature above 0 for distinct samples, e.g. `--param temperature=0.8`.

```python
from counterfix.code import pass_at_k, check_code

pass_at_k(n=20, c=3, k=10)
check_code(answer, {"test_list": ["assert add(1, 2) == 3"]}, allow=True).why
```

## HumanEval+ and MBPP+

The [benchmarks](benchmarks.md) `humaneval+` and `mbpp+` are scored by EvalPlus itself
(`pip install "counterfix[code]"`), so their pass@1 is EvalPlus's number; counterfix adds
the cause of each failure and the fix search. EvalPlus runs code in its own guarded child
process, not in counterfix's Docker sandbox.

## Accuracy

`bench/code_parity.py` scores the reference solution, a stub that returns None, and a
solution cut after one line for all 164 HumanEval+ problems, once with counterfix's
checker (original HumanEval tests) and once with EvalPlus's base tests. They agree on
491 of 492 verdicts; the one difference is HumanEval/32, whose reference solution
passes the original tests but not EvalPlus's base check. Details in
[Measured results](results.md#code).
