"""An MCP server, so coding agents can lint prompts and check LLM calls in code they write.

    pip install "counterfix[mcp]"
    counterfix mcp                     # stdio server

For example, in Claude Code: ``claude mcp add counterfix -- counterfix mcp``.

Tools: ``lint_prompt``, ``diagnose_call``, ``fix_call``, ``compare_profiles``,
``route_models``. Judge checks and fixes use your own keys, as everywhere else in
counterfix (a ``.env`` in the folder the server starts in is read).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from counterfix.core.record import CallRecord, Message


def _record(
    prompt: str,
    answer: str,
    system: str,
    model: str | None,
    max_tokens: int | None,
    finish_reason: str | None,
    source: str | None,
    reference: str | None,
) -> CallRecord:
    from counterfix.judges.tasks import CONTEXT_KEY, REFERENCE_KEY

    messages = [Message(role="system", content=system)] if system else []
    messages.append(Message(role="user", content=prompt))
    record = CallRecord(
        model=model,
        provider=model.split(":", 1)[0] if model and ":" in model else None,
        messages=messages,
        params={"max_tokens": max_tokens} if max_tokens else {},
        response_text=answer,
        finish_reason=finish_reason,
    )
    if model and ":" in model:
        record.model = model.split(":", 1)[1]
    if source:
        record.metadata[CONTEXT_KEY] = source
    if reference:
        record.metadata[REFERENCE_KEY] = reference
    return record


def _findings(report: Any) -> list[dict[str, Any]]:
    return [
        {
            "code": f.failure_type.value,
            "name": f.failure_type.label,
            "confidence": round(f.confidence, 2),
            "why": [e.detail for e in f.evidence],
            "fix": f.prescription.description if f.prescription else None,
            "fix_verified": bool(f.prescription and f.prescription.verified),
        }
        for f in report.findings
    ]


def lint_prompt(prompt: str, system: str = "", judge: bool = False) -> list[dict[str, Any]]:
    """Check a prompt for patterns that make LLM calls fail (rules L1-L7). No model call
    unless judge is true (then rules L2, L5, L6 also run, using your judge key)."""
    from counterfix.lint import lint

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    chosen = None
    if judge:
        from counterfix.judges.auto import resolve_judge

        chosen = resolve_judge()[0]
    return [
        {
            "rule": i.rule,
            "title": i.title,
            "severity": i.severity,
            "message": i.message,
            "suggestion": i.suggestion,
            "risk": [t.value for t in i.failure_types],
        }
        for i in lint(messages, judge=chosen)
    ]


def diagnose_call(
    prompt: str,
    answer: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int | None = None,
    finish_reason: str | None = None,
    judge: bool = False,
    source: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    """Find why an LLM answer failed (taxonomy F1-F9). Live checks are free; judge=true
    adds judge checks (invented facts, false premises, ...) with your judge key. source is
    retrieved text the model was given; reference is a known correct answer."""
    from counterfix.diagnose import diagnose

    record = _record(prompt, answer, system, model, max_tokens, finish_reason, source, reference)
    report = diagnose(record, judge=None if judge else False)
    return {
        "ok": report.ok,
        "summary": report.summary(),
        "explain": report.explain(),
        "findings": _findings(report),
        "judge": report.judge,
    }


def fix_call(
    prompt: str,
    answer: str,
    model: str,
    system: str = "",
    max_tokens: int | None = None,
    finish_reason: str | None = None,
    judge: bool = True,
    max_runs: int = 9,
) -> dict[str, Any]:
    """Find a fix that was tested to work: re-run the call (model as provider:model, e.g.
    groq:openai/gpt-oss-120b) with one change at a time; a fix counts only if it passes
    2 of 3 re-runs. Makes real model calls with your keys."""
    from counterfix.engine import fix
    from counterfix.presets import load_env_file

    load_env_file(Path.cwd() / ".env")
    record = _record(prompt, answer, system, model, max_tokens, finish_reason, None, None)
    result = fix(record, judge=None if judge else False, max_runs=max_runs)
    best = result.fix
    return {
        "explain": result.explain(),
        "verified": best is not None,
        "change": None if best is None else best.changed,
        "messages": None
        if best is None or best.messages is None
        else [m.model_dump(exclude_none=True) for m in best.messages],
        "escalation": result.escalation,
        "re_runs": result.calls,
    }


def compare_profiles(old: str, new: str) -> str:
    """Failure rates per task for two models (profile files or model names)."""
    from counterfix import profiles

    a, b = profiles.load(old), profiles.load(new)
    if a is None or b is None:
        return "Profile not found: " + ", ".join(n for n, p in ((old, a), (new, b)) if p is None)
    return profiles.compare(a, b)


def route_models(models: list[str], max_failure: float = 0.2) -> dict[str, str | None]:
    """For each task type, the cheapest of these models (cheapest first, or by profile
    cost) whose measured failure rate is at most max_failure."""
    from counterfix import profiles

    found = [p for p in (profiles.load(m) for m in models) if p is not None]
    return profiles.route(found, max_failure)


TOOLS = [lint_prompt, diagnose_call, fix_call, compare_profiles, route_models]


def server() -> Any:
    from mcp.server.mcpserver import MCPServer

    app = MCPServer(
        name="counterfix",
        instructions="Find why an LLM call failed and get a fix that was tested to work.",
    )
    for tool in TOOLS:
        app.tool()(cast(Any, tool))
    return app


def main() -> None:
    server().run("stdio")
