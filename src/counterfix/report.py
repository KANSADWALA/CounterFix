"""Logs and reports: write calls to a JSONL file, read them back, and show what was found.

>>> client = guard(OpenAI(), config=Config(on_diagnosis=log_to("calls.jsonl")))  # doctest: +SKIP

Then ``counterfix diagnose calls.jsonl --html report.html`` turns the log into a report.
Each line of a log holds ``{"record": ..., "diagnosis": ...}``; a line holding a bare
CallRecord is read too.
"""

from __future__ import annotations

import html
import json
import re
import threading
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from counterfix.core.config import DiagnosisHook
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord

_SECRET = re.compile(r"auth|key|token|secret|password|cookie|signature", re.IGNORECASE)
_HEADER_PARAMS = ("extra_headers", "headers", "default_headers")


def _redacted(params: dict[str, Any]) -> dict[str, Any]:
    """Request parameters safe to write to disk: secret-looking headers and keys masked."""
    out = dict(params)
    for name in _HEADER_PARAMS:
        if isinstance(out.get(name), dict):
            out[name] = {k: "***" if _SECRET.search(str(k)) else v for k, v in out[name].items()}
    for name in list(out):
        if name not in _HEADER_PARAMS and _SECRET.search(name) and isinstance(out[name], str):
            out[name] = "***"
    return out


def log_line(record: CallRecord, diagnosis: Diagnosis) -> str:
    """One JSON line holding the call and what counterfix found. Secret-looking headers
    are masked, and values JSON cannot hold (SDK objects) are written as text, so a line
    is never lost to an odd parameter."""
    data = record.model_dump(mode="python")
    data["params"] = _redacted(data.get("params") or {})
    return json.dumps(
        {"record": data, "diagnosis": diagnosis.model_dump(mode="json")},
        ensure_ascii=False,
        default=_as_text,
    )


def _as_text(value: Any) -> Any:
    """json.dumps fallback: pydantic models as dicts, everything else as its repr."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    if isinstance(value, type):
        return value.__name__  # e.g. response_format=MyModel from parse()
    return repr(value)


def log_to(path: str | Path) -> DiagnosisHook:
    """A ``Config.on_diagnosis`` hook that appends every guarded call to a JSONL file."""
    target = Path(path)
    lock = threading.Lock()

    def hook(record: CallRecord, diagnosis: Diagnosis) -> None:
        line = log_line(record, diagnosis)
        with lock, target.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return hook


def load_log(path: str | Path) -> list[CallRecord]:
    """The calls in a JSONL log written by :func:`log_to` (or one CallRecord per line)."""
    records = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            record = CallRecord.model_validate(obj.get("record", obj))
            for name in _HEADER_PARAMS:  # masked secrets: let the client send its own
                headers = record.params.get(name)
                if isinstance(headers, dict):
                    record.params[name] = {k: v for k, v in headers.items() if v != "***"}
            records.append(record)
        except Exception as exc:
            raise ValueError(f"{path}:{n}: not a call record ({exc})") from exc
    return records


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def counts(diagnoses: Iterable[Diagnosis]) -> Counter[str]:
    """How many calls had each failure type, e.g. {"F1 Truncation": 3}."""
    found: Counter[str] = Counter()
    for d in diagnoses:
        found.update({f"{f.failure_type.value} {f.failure_type.label}" for f in d.findings})
    return found


def text_report(diagnoses: list[Diagnosis], *, failures_only: bool = True) -> str:
    """A terminal summary: totals per failure type, then each flagged call explained."""
    flagged = [d for d in diagnoses if not d.ok]
    lines = [f"{len(diagnoses)} call(s) checked, {len(flagged)} with a failure."]
    for name, n in sorted(counts(diagnoses).items()):
        lines.append(f"  {name:<40} {n}")
    for d in diagnoses if not failures_only else flagged:
        record = d.record
        prompt = _short(record.prompt_text) if record is not None else d.record_id
        model = f" [{record.model}]" if record is not None and record.model else ""
        lines += ["", f"- {prompt}{model}"]
        lines += ["    " + line for line in d.explain().splitlines()]
    return "\n".join(lines)


_CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --card:#f5f5f7; --line:#d2d2d7;
        --bad:#b42318; --ok:#067647; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141416; --fg:#f5f5f7; --muted:#a1a1a6; --card:#1f1f23; --line:#3a3a3f;
          --bad:#f97066; --ok:#47cd89; } }
body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:900px; padding:24px 16px;
       font:15px/1.5 system-ui, sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } .muted { color:var(--muted); }
table { border-collapse:collapse; margin:16px 0; }
td, th { padding:4px 12px 4px 0; text-align:left; }
details { background:var(--card); border:1px solid var(--line); border-radius:8px;
          margin:8px 0; padding:8px 12px; }
summary { cursor:pointer; } .bad { color:var(--bad); font-weight:600; } .ok { color:var(--ok); }
pre { white-space:pre-wrap; word-break:break-word; margin:4px 0 8px; font-size:13px; }
"""


def html_report(diagnoses: list[Diagnosis], *, title: str = "counterfix report") -> str:
    """A single self-contained HTML file: totals, then one expandable card per call."""
    e = html.escape
    flagged = sum(not d.ok for d in diagnoses)
    rows = "".join(
        f"<tr><td>{e(name)}</td><td>{n}</td></tr>" for name, n in sorted(counts(diagnoses).items())
    )
    cards = []
    for d in sorted(diagnoses, key=lambda d: d.ok):  # failures first
        record = d.record
        prompt = record.prompt_text if record is not None else ""
        answer = record.answer_text if record is not None else ""
        status = (
            '<span class="ok">ok</span>' if d.ok else f'<span class="bad">{e(d.summary())}</span>'
        )
        body = "".join(
            f"<p><b>{e(f.failure_type.value)} {e(f.failure_type.label)}</b> "
            f"({f.confidence:.0%})</p>"
            + "".join(f"<pre>why: {e(ev.detail)}</pre>" for ev in f.evidence)
            + (f"<pre>fix: {e(f.prescription.description)}</pre>" if f.prescription else "")
            for f in d.findings
        )
        model = f' <span class="muted">{e(record.model)}</span>' if record and record.model else ""
        cards.append(
            f"<details{'' if d.ok else ' open'}><summary>{status} · {e(_short(prompt, 80))}"
            f"{model}</summary>{body}<p class='muted'>prompt</p><pre>{e(prompt)}</pre>"
            f"<p class='muted'>answer</p><pre>{e(answer) or '(empty)'}</pre></details>"
        )
    judge = next((d.judge for d in diagnoses if d.judge), None)
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{e(title)}</h1><p class='muted'>{len(diagnoses)} call(s) checked, {flagged} with "
        f"a failure{' · judged by ' + e(judge) if judge else ' · live checks only'}</p>"
        f"<table><tr><th>Failure</th><th>Calls</th></tr>{rows}</table>{''.join(cards)}"
        f"</body></html>"
    )
