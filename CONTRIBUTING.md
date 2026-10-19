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

CI runs the same three on Python 3.10 to 3.13.

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

Changes to judge prompts or checks should not make the labelled seed set worse:

```bash
python bench/eval_judges.py --offline   # from cached verdicts, no API calls
```

A full run needs free `GROQ_API_KEY` / `GEMINI_API_KEY` keys; see `bench/` and
`docs/results.md`.
