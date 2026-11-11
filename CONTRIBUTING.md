# Contributing to counterfix

Thanks for helping. Bug reports, failure cases a check gets wrong, new fixes and docs
improvements are all welcome.

## Set up

```bash
git clone https://github.com/KANSADWALA/CounterFix.git
cd CounterFix
pip install -e ".[dev]"
pytest
```

Python 3.10 or newer. The tests make no live API calls and need no key.

## Before you open a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest
```

CI runs the same three on Python 3.10 to 3.14, plus a strict docs build
(`pip install -e ".[docs]"`, then `mkdocs build --strict`).

- **Add a test** for every bug fix or new check. A check that flags something should also
  have a test showing what it must *not* flag.
- **Keep the core light.** The core depends on 4 small libraries. A provider SDK or a
  heavy tool goes in an optional extra in `pyproject.toml`, imported only when used.
- **Fail open.** Nothing in `guard()` may break the user's call: internal errors are
  logged and recorded on the diagnosis, and provider errors reach the user unchanged.
- **Never commit keys.** Put them in `.env` (git-ignored); `.env.example` lists the names.
- Add a line to `CHANGELOG.md` under the next version.

## Reporting a wrong result

The most useful report is a single call: the messages, the parameters, the model's
answer, what counterfix said, and what you expected. `counterfix.log_to("calls.jsonl")`
writes calls in exactly that form (secret-looking headers are masked).

## Judge accuracy

Changes to judge prompts or checks should not make the labelled sets worse. Each judged
check has its own labelled set and script:

```bash
python bench/eval_judges.py --offline   # F3 to F9 on the main seed, from cached verdicts
python bench/eval_rag.py                # faithfulness, F10 and F11 (rag_seed, rag_holdout)
python bench/eval_injection.py          # F13
python bench/eval_agents.py             # F14 task success
python bench/eval_metrics.py            # answer_relevancy, answer_correctness, context metrics
python bench/code_parity.py             # the code checker against EvalPlus (Linux, macOS)
```

Live runs need free `GROQ_API_KEY` / `GEMINI_API_KEY` keys or Ollama; see `bench/` and
`docs/results.md`. `python bench/dev_walkthrough.py SPEC` checks the library end to end on
a real model, the way a developer would use it.
