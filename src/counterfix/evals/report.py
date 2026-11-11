"""Eval reports: the score, the misses by cause, and the verified fixes."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from counterfix.core.types import FailureType

if TYPE_CHECKING:
    from counterfix.evals.runner import EvalResult, FixTry, ItemResult, MetricSummary


def _label(code: str) -> str:
    try:
        return f"{code} {FailureType(code).label}"
    except ValueError:
        return code


def _num(x: float | None, as_ms: bool) -> str:
    if x is None:
        return "-"
    return f"{x:.0f}" if as_ms else f"{x:.3f}"


def _metric_row(m: MetricSummary) -> tuple[str, str, str, str]:
    as_ms = not m.higher_is_better  # latency-like metrics are in ms
    name = f"{m.name} (ms)" if as_ms else m.name
    ci = f"{_num(m.ci_low, as_ms)} to {_num(m.ci_high, as_ms)}" if m.ci_low is not None else "-"
    judged = m.passed + m.failed
    passes = f"{m.passed}/{judged}" if judged else "-"
    if m.not_scored:
        passes += f" ({m.not_scored} not scored)"
    return name, _num(m.mean, as_ms), ci, passes


def _question(item: ItemResult) -> str:
    """The question asked: the bare RAG question when counterfix built the prompt."""
    q = item.record.metadata.get("rag_question")
    return q if isinstance(q, str) else item.record.prompt_text


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fix_line(item: ItemResult) -> str:
    f = item.fix
    if f is None:
        return ""
    if f.fix is not None:
        return f"verified fix: {f.fix.description} ({f.fix.passed}/{f.fix.runs})"
    tries = [t for t in f.tried if t.runs]
    if not tries:
        return f"fix search: {f.note}"

    def closeness(t: FixTry) -> tuple[int, int]:
        # fewest runs where the failure itself stayed (or a new one came): a try that
        # removed the failure but missed a gate is closer than one that changed nothing
        broken = sum(r.startswith(("still ", "new ", "call failed")) for r in t.why_not)
        return t.passed, -broken

    best = max(tries, key=closeness)
    why = f": {best.reasons()}" if best.why_not else ""
    return f"fix search: {f.note}; closest was {best.name}, passed {best.passed}/{best.runs}{why}"


def text_report(result: EvalResult) -> str:
    n, failed, flagged = len(result.items), result.failed, result.flagged
    head = f"eval {result.name}"
    head += f" · {result.model}" if result.model else " · recorded answers"
    head += f" · {n} item(s), {len(failed)} missed"
    if flagged:
        head += f", {len(flagged)} flagged"
    if result.judge:
        head += f" · judged by {result.judge}"
    lines = [head]
    info = {k: v for k, v in result.info.items() if k != "by_group" and v not in ("", None)}
    if info:
        lines.append("  " + " · ".join(f"{k} {v}" for k, v in info.items()))
    for note in result.notes:
        lines.append(f"  note: {note}")
    lines.append("")
    lines.append(f"  {'metric':<24} {'mean':>9}  {'95% CI':<20} pass")
    for m in result.metrics:
        name, mean, ci, passes = _metric_row(m)
        lines.append(f"  {name:<24} {mean:>9}  {ci:<20} {passes}")
    by_group = result.info.get("by_group") or {}
    if by_group:

        def rate(v: str) -> float:
            done, total = (int(x) for x in v.split("/"))
            return done / total if total else 0.0

        weak = [
            kv for kv in sorted(by_group.items(), key=lambda kv: rate(kv[1])) if rate(kv[1]) < 1
        ]
        if weak:
            shown = ", ".join(f"{g} {v}" for g, v in weak[:5])
            lines.append(f"  weakest groups: {shown} ({len(by_group)} groups in all)")
        else:
            lines.append(f"  every group passed in full ({len(by_group)} groups)")
    for name, values in result.pass_at_k.items():
        shown = ", ".join(f"pass@{k} {v:.3f}" for k, v in sorted(values.items()))
        lines.append(f"  {name}: {shown} ({result.samples} samples per question)")
    if failed:
        lines += ["", f"Misses by cause ({len(failed)})"]
        for code, count in result.by_cause.items():
            lines.append(f"  {_label(code):<44} {count}")
        if result.code_causes:
            from counterfix.code import CAUSE_TEXT

            lines.append("  code that failed its tests, by what went wrong:")
            for cause, count in result.code_causes.items():
                lines.append(f"    {CAUSE_TEXT.get(cause, cause):<42} {count}")
        if result.judge_errors:
            lines.append(
                f"  ({len(result.judge_errors)} miss(es) could not be judged, so their cause may "
                f"be missing; first: {result.judge_errors[0]})"
            )
    if flagged:
        lines += ["", f"Flagged by live checks, though the metrics passed ({len(flagged)})"]
        for code, count in result.flagged_by_cause.items():
            lines.append(f"  {_label(code):<44} {count}")
        for item in flagged[:8]:
            line = f"  - {item.id}: {_short(_question(item), 60)}"
            if fl := _fix_line(item):
                line += f"\n      {fl}"
            lines.append(line)
        if len(flagged) > 8:
            lines.append(f"  ... and {len(flagged) - 8} more")
        if not result.fix_tried:
            lines.append("  (--fix N searches verified fixes; --fail-on-flagged fails CI on these)")
    if result.fix_tried:
        lines += ["", f"Verified fixes (searched on {result.fix_tried} missed or flagged item(s))"]
        for name, count in result.fixes.items():
            lines.append(f"  {name:<44} {count}")
    if result.speed.calls:
        lines += ["", "Speed and cost", result.speed.table()]
    if failed:
        lines += ["", "Missed items"]
        for item in failed[:20]:
            why = "; ".join(r.reason for r in item.metrics if r.passed is False) or (
                item.record.error or ""
            )
            cause = f" [{_label(item.top_cause)}]" if item.top_cause else ""
            lines.append(f"- {item.id}: {_short(_question(item), 60)}{cause}")
            lines.append(f"    {_short(why, 100)}")
            if fl := _fix_line(item):
                lines.append(f"    {fl}")
        if len(failed) > 20:
            lines.append(f"  ... and {len(failed) - 20} more")
    if result.stopped:
        lines += ["", result.stopped]
    return "\n".join(lines)


def html_report(result: EvalResult) -> str:
    from counterfix.report import _CSS

    e = html.escape
    rows = "".join(
        "<tr>" + "".join(f"<td>{e(c)}</td>" for c in _metric_row(m)) + "</tr>"
        for m in result.metrics
    )
    causes = "".join(
        f"<tr><td>{e(_label(c))}</td><td>{k}</td></tr>" for c, k in result.by_cause.items()
    )
    fixes = "".join(f"<tr><td>{e(name)}</td><td>{k}</td></tr>" for name, k in result.fixes.items())
    flagged = "".join(
        f"<tr><td>{e(_label(c))}</td><td>{k}</td></tr>" for c, k in result.flagged_by_cause.items()
    )
    from counterfix.code import CAUSE_TEXT

    causes += "".join(
        f"<tr><td>&nbsp;&nbsp;code: {e(CAUSE_TEXT.get(c, c))}</td><td>{k}</td></tr>"
        for c, k in result.code_causes.items()
    )
    passk = "".join(
        f"<p>{e(name)}: "
        + ", ".join(f"pass@{k} {v:.3f}" for k, v in sorted(values.items()))
        + f" <span class='muted'>({result.samples} samples per question)</span></p>"
        for name, values in result.pass_at_k.items()
    )
    cards = []
    for item in result.failed:
        reasons = "".join(
            f"<pre>{e(r.name)}: {e(r.reason)}</pre>" for r in item.metrics if r.passed is False
        )
        if item.record.error:
            reasons += f"<pre>error: {e(item.record.error)}</pre>"
        fix = _fix_line(item)
        cause = f" · {e(_label(item.top_cause))}" if item.top_cause else ""
        cards.append(
            f"<details><summary><span class='bad'>{e(item.id)}</span>{cause} · "
            f"{e(_short(_question(item)))}</summary>{reasons}"
            + (f"<pre>{e(fix)}</pre>" if fix else "")
            + f"<p class='muted'>prompt</p><pre>{e(item.record.prompt_text)}</pre>"
            f"<p class='muted'>answer</p><pre>{e(item.record.answer_text) or '(empty)'}</pre>"
            f"<p class='muted'>reference</p><pre>{e(str(item.reference))}</pre></details>"
        )
    speed = (
        f"<h2>Speed and cost</h2><pre>{e(result.speed.table())}</pre>" if result.speed.calls else ""
    )
    sub = f"{len(result.items)} item(s), {len(result.failed)} missed"
    sub += f", {len(result.flagged)} flagged" if result.flagged else ""
    sub += f" · {e(result.model)}" if result.model else " · recorded answers"
    sub += f" · judged by {e(result.judge)}" if result.judge else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(result.name)}</title><style>{_CSS}</style></head><body>"
        f"<h1>eval {e(result.name)}</h1><p class='muted'>{sub} · {e(result.created)}</p>"
        "<table><tr><th>Metric</th><th>Mean</th><th>95% CI</th><th>Pass</th></tr>"
        f"{rows}</table>{passk}"
        + (f"<h2>Misses by cause</h2><table>{causes}</table>" if causes else "")
        + (
            f"<p class='muted'>{len(result.judge_errors)} miss(es) could not be judged: "
            f"{e(result.judge_errors[0])}</p>"
            if result.judge_errors
            else ""
        )
        + (
            "<h2>Flagged by live checks</h2><p class='muted'>these items passed their metrics, "
            f"but a live check found a model failure</p><table>{flagged}</table>"
            if flagged
            else ""
        )
        + (
            f"<h2>Verified fixes</h2><p class='muted'>searched on {result.fix_tried} missed or "
            f"flagged item(s)</p><table>{fixes}</table>"
            if result.fix_tried
            else ""
        )
        + speed
        + ("<h2>Missed items</h2>" + "".join(cards) if cards else "")
        + "".join(f"<p class='muted'>Note: {e(n)}</p>" for n in result.notes)
        + (f"<p class='muted'>{e(result.stopped)}</p>" if result.stopped else "")
        + "</body></html>"
    )
