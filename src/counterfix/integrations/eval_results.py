"""Read the results files of DeepEval, Ragas and Promptfoo, so counterfix can explain
their failures and search for verified fixes.

>>> from counterfix.integrations.eval_results import read_results     # doctest: +SKIP
>>> cases = read_results("results.json")          # the tool is detected from the file
>>> [c for c in cases if c.failed]

Each :class:`ResultCase` holds a :class:`CallRecord` built from the test case and the
tool's own verdict. What the files hold differs, and so does what counterfix can do:

- **Promptfoo** (``promptfoo eval -o results.json``) records the exact prompt sent, the
  provider, its settings and the finish reason: the call can be re-run as it was, and
  deterministic assertions (``contains``, ``equals``, ``regex``, ``is-json`` ...) are
  re-checked on every re-run, so your assertions decide which re-runs pass.
- **DeepEval** (the ``test_run_*.json`` written with a results folder, or
  ``.deepeval/.latest_test_run.json``) and **Ragas** (``result.to_pandas()`` saved as CSV
  or JSON) record the test input, the answer, the retrieved context and the scores, but
  not the model or any system prompt your app added. They can be diagnosed as they are;
  re-running them needs a model (``--caller``) and, if your app adds one, the system
  prompt (``--system``), or the re-run is not the call your app makes. Their metrics are
  Python objects that are not in the file, so re-runs are judged by counterfix's checks,
  or by your own check (``--check``).
- Ragas saves scores without pass/fail: give thresholds, e.g. ``{"faithfulness": 0.8}``
  (higher is better) or ``{"noise_sensitivity": "<=0.3"}`` (lower is better).
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from counterfix.core.record import CallRecord, Message, Usage
from counterfix.judges.tasks import CONTEXT_KEY, REFERENCE_KEY

TOOLS = ("promptfoo", "deepeval", "ragas")

# request settings kept from a Promptfoo provider config (others, like keys, are not)
_PARAMS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "response_format",
)
# Promptfoo provider prefix -> counterfix's name for it (None: no preset, use --caller)
_PROVIDERS = {
    "openai": "openai",
    "groq": "groq",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "google": "gemini",
    "anthropic": "anthropic",
}
_SUBTYPES = {"chat", "completion", "messages", "responses"}


@dataclass
class ResultCase:
    """One test case from an evaluation tool's results."""

    name: str
    record: CallRecord
    failed: bool
    """The tool's verdict: True when the test case failed (or errored)."""
    tool: str = ""
    """"promptfoo", "deepeval" or "ragas"."""
    reason: str = ""
    """Why the tool failed it, in its own words (scores, assertion messages)."""
    rerun: str | None = None
    """A model spec that re-runs the call (e.g. ``groq:llama-3.1-8b-instant``), when the
    file says which model served it; None when only ``--caller`` can say."""
    check: Callable[[CallRecord], tuple[bool, str]] | None = None
    """Re-checks an answer the way the tool did, when that can be done without the tool
    (Promptfoo's deterministic assertions); None otherwise."""
    notes: list[str] = field(default_factory=list)
    """What could not be carried over, said plainly."""
    errored: bool = False
    """The call itself errored in the tool's run (no answer to diagnose)."""


def read_results(
    path: str | Path,
    tool: str = "auto",
    *,
    thresholds: Mapping[str, float | str] | None = None,
) -> list[ResultCase]:
    """The test cases in a results file of DeepEval, Ragas or Promptfoo. ``tool`` is
    "auto" (detected from the file), "promptfoo", "deepeval" or "ragas"; ``thresholds``
    turn Ragas scores into pass/fail and are required for Ragas."""
    path = Path(path)
    data = _load(path)
    tool = _detect(data, path) if tool == "auto" else tool
    readers = {"promptfoo": _promptfoo, "deepeval": _deepeval}
    if tool == "ragas":
        cases = _ragas(data, thresholds)
    elif tool in readers:
        cases = readers[tool](data)
    else:
        raise ValueError(f"unknown tool {tool!r}: use one of {', '.join(TOOLS)}")
    for case in cases:
        case.tool = tool
    return cases


# ---------------------------------------------------------------- loading and detection


def detect_tool(path: str | Path) -> str | None:
    """ "promptfoo", "deepeval" or "ragas" when the file is one of their results; else None."""
    try:
        path = Path(path)
        return _detect(_load(path), path)
    except (OSError, ValueError):
        return None


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        text = text.replace("\r\n", "\n")  # a CSV saved on Windows
        return list(csv.DictReader(io.StringIO(text, newline="")))  # keeps multi-line cells
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows  # JSON lines (e.g. a pandas frame saved with lines=True)


def _detect(data: Any, path: Path) -> str:
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, dict) and isinstance(results.get("results"), list):
            return "promptfoo"
        if "testCases" in data or "testRunData" in data:
            return "deepeval"
    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = set(data[0])
        if keys & {"user_input", "question"} and keys & {"response", "answer"}:
            return "ragas"
    raise ValueError(
        f"{path}: not a results file counterfix can read. Expected Promptfoo's "
        "`promptfoo eval -o results.json`, DeepEval's test_run_*.json, or Ragas's "
        "result.to_pandas() saved as CSV or JSON"
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _contexts(value: Any) -> list[str]:
    """Retrieved passages, from a list or a CSV cell holding a Python or JSON list."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _attach(record: CallRecord, contexts: list[str], reference: Any) -> None:
    if contexts:
        from counterfix.detect.retrieval import CHUNKS_KEY, joined, to_chunks

        chunks = to_chunks(contexts)
        record.metadata[CHUNKS_KEY] = [c.model_dump() for c in chunks]
        record.metadata[CONTEXT_KEY] = joined(chunks)
    if reference not in (None, "") and not (isinstance(reference, float) and math.isnan(reference)):
        record.metadata[REFERENCE_KEY] = _text(reference)


# --------------------------------------------------------------------------- Promptfoo


def _promptfoo(data: dict[str, Any]) -> list[ResultCase]:
    configs = [p for p in (data.get("config") or {}).get("providers") or [] if isinstance(p, dict)]
    cases = []
    for n, row in enumerate(data["results"]["results"]):
        provider = row.get("provider") or {}
        pid = provider.get("id", "") if isinstance(provider, dict) else str(provider)
        label = provider.get("label", "") if isinstance(provider, dict) else ""
        config = next(
            (
                c.get("config") or {}
                for c in configs
                if c.get("id") == pid and (c.get("label") or "") == (label or "")
            ),
            next((c.get("config") or {} for c in configs if c.get("id") == pid), {}),
        )
        response = row.get("response") or {}
        tokens = response.get("tokenUsage") or {}
        prompt = (row.get("prompt") or {}).get("raw", "")
        name, model, spec = _provider(pid, config)
        record = CallRecord(
            model=model,
            provider=name,
            messages=_messages(prompt),
            params={k: config[k] for k in _PARAMS if k in config},
            response_text=None if response.get("output") is None else _text(response.get("output")),
            finish_reason=response.get("finishReason"),
            usage=Usage(
                prompt_tokens=tokens.get("prompt"),
                completion_tokens=tokens.get("completion"),
                total_tokens=tokens.get("total"),
            )
            if tokens
            else None,
            latency_ms=row.get("latencyMs"),
            error=None if row.get("failureReason") != 2 else _text(row.get("error") or "error"),
        )
        grading = row.get("gradingResult") or {}
        parts = [p for p in grading.get("componentResults") or [] if isinstance(p, dict)]
        reason = "; ".join(
            str(p.get("reason")) for p in parts if not p.get("pass") and p.get("reason")
        ) or str(grading.get("reason") or row.get("error") or "")
        check, notes = _assertion_check(parts, row)
        if spec is None:
            notes.append(f"no preset for the provider {pid!r}: re-run it with --caller")
        cases.append(
            ResultCase(
                name=_case_name(row, n),
                record=record,
                failed=not row.get("success", False),
                reason=reason,
                rerun=spec,
                check=check,
                notes=notes,
                errored=row.get("failureReason") == 2,
            )
        )
    return cases


def _case_name(row: dict[str, Any], n: int) -> str:
    desc = (row.get("testCase") or {}).get("description")
    where = f"test {row.get('testIdx', n)}, prompt {row.get('promptIdx', 0)}"
    return f"{desc} ({where})" if desc else where


def _provider(pid: str, config: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    """(counterfix provider, model, re-run spec) from a Promptfoo provider id."""
    head, _, rest = pid.partition(":")
    sub, _, after = rest.partition(":")
    model = after if sub in _SUBTYPES and after else rest
    if not rest or pid.startswith(("file:", "python:", "exec:", "http", "ws")):
        return None, pid or None, None
    base = config.get("apiBaseUrl") or config.get("apiHost")
    if base and head in ("openai", "ollama", "groq", "openrouter"):
        url = str(base) if "://" in str(base) else f"https://{base}"
        return head, model, f"{model}@{url}"
    name = _PROVIDERS.get(head)
    return name, model, None if name is None else f"{name}:{model}"


def _messages(prompt: str) -> list[Message]:
    """Promptfoo's rendered prompt: a JSON list of chat messages, or plain text."""
    text = prompt.strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
            if items and all(isinstance(m, dict) and "role" in m for m in items):
                return [Message.model_validate(m) for m in items]
        except (json.JSONDecodeError, ValueError):
            pass
    return [Message(role="user", content=prompt)]


_VAR = re.compile(r"{{\s*([A-Za-z_][\w.]*)\s*}}")


def _render(value: Any, part: Mapping[str, Any], vars_: Mapping[str, Any]) -> Any:
    rendered = (part.get("metadata") or {}).get("renderedAssertionValue")
    if rendered is not None:
        return rendered
    if isinstance(value, str) and "{{" in value:
        if "{%" in value or "|" in value:
            raise ValueError("templated value")
        return _VAR.sub(lambda m: _text(vars_.get(m.group(1), m.group(0))), value)
    return value


def _one(kind: str, value: Any, out: str) -> bool:
    values = value if isinstance(value, list) else [value]
    if kind == "equals":
        return out == _text(value) if not isinstance(value, (dict, list)) else _json_eq(out, value)
    if kind == "contains":
        return _text(value) in out
    if kind == "icontains":
        return _text(value).lower() in out.lower()
    if kind == "contains-any":
        return any(_text(v) in out for v in values)
    if kind == "contains-all":
        return all(_text(v) in out for v in values)
    if kind == "icontains-any":
        return any(_text(v).lower() in out.lower() for v in values)
    if kind == "icontains-all":
        return all(_text(v).lower() in out.lower() for v in values)
    if kind == "starts-with":
        return out.startswith(_text(value))
    if kind == "regex":
        return re.search(_text(value), out) is not None
    if kind == "is-json":
        try:
            json.loads(out)
        except json.JSONDecodeError:
            return False
        return True
    raise KeyError(kind)


def _json_eq(out: str, value: Any) -> bool:
    try:
        return bool(json.loads(out) == value)
    except json.JSONDecodeError:
        return False


SUPPORTED_ASSERTIONS = (
    "equals",
    "contains",
    "icontains",
    "contains-any",
    "contains-all",
    "icontains-any",
    "icontains-all",
    "starts-with",
    "regex",
    "is-json",
)


def _assertion_check(
    parts: list[dict[str, Any]], row: dict[str, Any]
) -> tuple[Callable[[CallRecord], tuple[bool, str]] | None, list[str]]:
    """Promptfoo's assertions as a check, when every one is deterministic and the test
    passes only if all pass. Otherwise None and a note saying why."""
    test = row.get("testCase") or {}
    vars_ = row.get("vars") or test.get("vars") or {}
    if not parts:
        return None, ["no assertions in the file to re-check"]
    if test.get("threshold") is not None:
        return None, ["the test uses a score threshold, which counterfix does not re-check"]
    rules: list[tuple[str, bool, Any]] = []
    for part in parts:
        assertion = part.get("assertion") or {}
        kind = str(assertion.get("type", ""))
        negate = kind.startswith("not-")
        base = kind[4:] if negate else kind
        if base not in SUPPORTED_ASSERTIONS or assertion.get("transform"):
            return None, [
                f"the {kind or 'unnamed'} assertion cannot be re-checked without Promptfoo, so"
                " re-runs are judged by counterfix's checks (or pass --check)"
            ]
        try:
            value = _render(assertion.get("value"), part, vars_)
        except ValueError:
            return None, [f"the {kind} assertion's value is a template counterfix cannot render"]
        rules.append((base, negate, value))
    transform = (test.get("options") or {}).get("transform")
    if transform:
        return None, ["the test transforms the output first, which counterfix cannot repeat"]

    def check(record: CallRecord) -> tuple[bool, str]:
        out = record.response_text or ""
        for base, negate, value in rules:
            if _one(base, value, out) == negate:
                return False, f"{'not-' if negate else ''}{base} {_text(value)[:60]} failed"
        return True, ""

    check.__name__ = "(the Promptfoo assertions)"
    return check, []


# ---------------------------------------------------------------------------- DeepEval


def _deepeval(data: dict[str, Any]) -> list[ResultCase]:
    run = data.get("testRunData", data)
    hyper = run.get("hyperparameters") or {}
    cases = []
    for n, tc in enumerate(run.get("testCases") or []):
        meta = tc.get("metadata") or {}
        model = meta.get("model") or hyper.get("model")
        record = CallRecord(
            model=None if model is None else str(model),
            messages=[Message(role="user", content=_text(tc.get("input")))],
            response_text=_text(tc.get("actualOutput")),
        )
        _attach(
            record,
            _contexts(tc.get("retrievalContext")) or _contexts(tc.get("context")),
            tc.get("expectedOutput"),
        )
        metrics = [m for m in tc.get("metricsData") or [] if isinstance(m, dict)]
        reason = "; ".join(
            f"{m.get('name')} {_score(m.get('score'))} (threshold {m.get('threshold')}): "
            f"{m.get('error') or m.get('reason') or ''}".rstrip(": ")
            for m in metrics
            if not m.get("success")
        )
        cases.append(
            ResultCase(
                name=str(tc.get("name") or f"test_case_{n}"),
                record=record,
                failed=not tc.get("success", False),
                reason=reason,
                notes=_bare_notes("DeepEval"),
            )
        )
    return cases


def _score(value: Any) -> str:
    return "no score" if value is None else f"{float(value):.2f}"


def _bare_notes(tool: str) -> list[str]:
    return [
        f"{tool} files hold the test input, not the full prompt or the model: re-runs send "
        "the input as the user message (add --system if your app uses a system prompt)"
    ]


# ------------------------------------------------------------------------------- Ragas

_RAGAS_FIELDS = {
    "user_input",
    "question",
    "response",
    "answer",
    "retrieved_contexts",
    "contexts",
    "reference",
    "ground_truth",
    "ground_truths",
    "reference_contexts",
    "multi_responses",
    "rubrics",
}


def _ragas(
    rows: list[dict[str, Any]], thresholds: Mapping[str, float | str] | None
) -> list[ResultCase]:
    scores = [k for k in rows[0] if k not in _RAGAS_FIELDS] if rows else []
    if not thresholds:
        raise ValueError(
            "Ragas saves scores, not pass/fail: give a threshold for each metric to judge, "
            f"e.g. --threshold {scores[0] if scores else 'faithfulness'}=0.8 "
            f"(score columns in this file: {', '.join(scores) or 'none'})"
        )
    rules = {name: _rule(name, spec) for name, spec in thresholds.items()}
    missing = [name for name in rules if name not in scores]
    if missing:
        raise ValueError(
            f"no column {', '.join(missing)} in this file; its score columns are "
            f"{', '.join(scores) or 'none'}"
        )
    cases = []
    for n, row in enumerate(rows):
        record = CallRecord(
            messages=[
                Message(role="user", content=_text(row.get("user_input", row.get("question"))))
            ],
            response_text=_text(row.get("response", row.get("answer"))),
        )
        _attach(
            record,
            _contexts(row.get("retrieved_contexts", row.get("contexts"))),
            row.get("reference", row.get("ground_truth")),
        )
        misses, unscored = [], []
        for name, (lower, limit) in rules.items():
            value = _number(row.get(name))
            if value is None:
                unscored.append(name)
            elif (value > limit) if lower else (value < limit):
                misses.append(f"{name} {value:.2f} ({'<=' if lower else '>='} {limit} needed)")
        notes = _bare_notes("Ragas")
        if unscored:
            notes.append(f"not scored by Ragas: {', '.join(unscored)}")
        cases.append(
            ResultCase(
                name=f"row {n}",
                record=record,
                failed=bool(misses),
                reason="; ".join(misses),
                notes=notes,
            )
        )
    return cases


def _rule(name: str, spec: float | str) -> tuple[bool, float]:
    """(lower is better, limit) from 0.8, "0.8", ">=0.8" or "<=0.3"."""
    text = str(spec).strip()
    lower = text.startswith("<=")
    text = text.removeprefix("<=").removeprefix(">=").strip()
    try:
        return lower, float(text)
    except ValueError:
        raise ValueError(
            f"threshold for {name} must be a number, >=N or <=N, not {spec!r}"
        ) from None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number
