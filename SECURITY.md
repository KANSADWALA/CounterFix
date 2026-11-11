# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting instead:
**Security -> Report a vulnerability** on this repository. You will get an answer within
7 days.

## What counts

For example: a way for model output or a log file to run code (the F8 check evaluates
judge-written expressions with a whitelist-only evaluator, never `eval`), a way for
model-written code to run when code execution is off, or to escape the sandbox when it is
on (off by default; `--allow-code` or `COUNTERFIX_ALLOW_CODE=1` turns it on, and
`--sandbox docker` runs it with no network and resource caps), a way for API keys to end
up in logs or reports, text in a judged answer or retrieved passage that takes over the
judge, or HTML reports that execute content from a call.

## Supported versions

Fixes go into the latest release on PyPI.
