"""Model profiles: how often a model fails each kind of request, and which fixes work on it.

    counterfix profile groq:qwen/qwen3.8-27b [--judge] [--fix]   # run the probe suite
    counterfix compare old.json new.json                          # what changed between two
    counterfix drift groq:qwen/qwen3.8-27b                        # re-run, alert on shifts
    counterfix route a.json b.json c.json                         # cheapest safe model per task

A profile is built by sending the probe suite (54 short prompts, 6 per failure type,
shipped with counterfix; pass ``probes=`` for your own) to the model and checking each
answer. With ``fix=True`` the engine also runs on each failure, and the profile records
which interventions were verified. ``fix()`` reads a model's profile (``profile="auto"``)
to try first what worked before on that model, so known models need fewer re-runs.

Profiles are JSON files under ``~/.counterfix/profiles`` (or ``$COUNTERFIX_HOME``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from counterfix.core.caller import DailyLimitError, ModelCaller
from counterfix.core.record import CallRecord, Message
from counterfix.fixes._profile_store import (
    PROBES as PROBES,
    SHIPPED as SHIPPED,
    TASKS as TASKS,
    Profile as Profile,
    home as home,
    load as load,
    profile_path as profile_path,
    save as save,
)


def load_probes(path: str | Path | None = None) -> list[dict[str, Any]]:
    rows = Path(path or PROBES).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in rows if line.strip()]


def probe_record(probe: dict[str, Any], record: CallRecord) -> CallRecord:
    """Attach what a probe knows (schema, right answer) so the checks can use it."""
    from counterfix.detect.fmt import SCHEMA_METADATA_KEY
    from counterfix.judges.tasks import REFERENCE_KEY

    if probe.get("schema"):
        record.metadata[SCHEMA_METADATA_KEY] = probe["schema"]
    if probe.get("gold"):
        record.metadata[REFERENCE_KEY] = probe["gold"]
    if probe.get("variants"):
        record.metadata["variants"] = probe["variants"]
    return record


def build_profile(
    caller: ModelCaller,
    *,
    judge: Any = False,
    probes: str | Path | None = None,
    fix: bool = False,
    max_runs: int = 30,
    progress: Any = None,
) -> Profile:
    """Run the probe suite on one model and measure it (``fix=True`` also learns fixes)."""
    from counterfix.diagnose import diagnose
    from counterfix.fixes.engine import fix as run_fix
    from counterfix.judges.auto import resolve_judge

    inner = getattr(caller, "caller", caller)  # unwrap RateLimitedCaller
    model = getattr(inner, "model", None) or caller.name.split(":", 1)[-1]
    chosen = None if judge is False else resolve_judge(None if judge is True else judge)[0]
    profile = Profile(model=str(model), judge=None if chosen is None else chosen.name)
    items = [p for p in load_probes(probes) if not p.get("control")]
    for n, probe in enumerate(items, 1):
        if progress:
            progress(n, len(items), probe["id"])
        messages = [Message.model_validate(m) for m in probe["messages"]]
        params = probe.get("params") or {}
        try:
            record = probe_record(probe, caller.complete(messages, **params))
        except DailyLimitError:
            raise  # every later probe would fail too: stop with the provider's message
        except Exception:
            profile.errors += 1  # provider errors are not the model's failures
            continue
        report = diagnose(record, judge=chosen if chosen is not None else False)
        profile.judge_skips += sum(
            why.startswith("judge failed") for why in report.skipped.values()
        )
        if probe["failure_type"] == "F7":  # fragility needs re-asks, so only on its probes
            from counterfix.detect import FragilityCheck

            extra = diagnose(record, detectors=[FragilityCheck(caller, judge=chosen)])
            report.findings += extra.findings
        row = profile.categories.setdefault(probe["failure_type"], [0, 0])
        row[1] += 1
        if report.model_failures:
            row[0] += 1
            if fix:
                result = run_fix(
                    report,
                    judge=chosen if chosen is not None else False,
                    caller=caller,
                    max_runs=max_runs,
                )
                profile.learn(result)
    return profile


def table(profile: Profile) -> str:
    """Failure rate per task, and the fixes that worked, for one profile."""
    lines = [f"{profile.model}: failure rate per task"]
    if profile.errors or profile.judge_skips:
        lines.append(
            f"  ({profile.errors} probe errors, {profile.judge_skips} judge checks skipped)"
        )
    for cat in sorted(profile.categories):
        failed, n = profile.categories[cat]
        lines.append(f"  {cat} {TASKS.get(cat, ''):<36} {failed}/{n}")
    for kind, row in sorted(profile.fixes.items()):
        best = ", ".join(f"{name} {v}/{t}" for name, (v, t) in row.items())
        lines.append(f"  fixes for {kind}: {best}")
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def compare(old: Profile, new: Profile) -> str:
    """A table of failure rates per task, old against new."""
    lines = [
        f"{'task':<36} {old.model[:18]:>18} {new.model[:18]:>18}",
        "-" * 74,
    ]
    for cat in sorted(set(old.categories) | set(new.categories)):
        a, b = old.failure_rate(cat), new.failure_rate(cat)
        mark = ""
        if a is not None and b is not None and abs(b - a) >= 0.15:
            mark = "  better" if b < a else "  WORSE"
        lines.append(f"{cat} {TASKS.get(cat, ''):<33} {_pct(a):>18} {_pct(b):>18}{mark}")
    return "\n".join(lines)


def drift(old: Profile, new: Profile, threshold: float = 0.15) -> list[str]:
    """Tasks whose failure rate moved by ``threshold`` or more."""
    out = []
    for cat in sorted(set(old.categories) & set(new.categories)):
        a, b = old.failure_rate(cat), new.failure_rate(cat)
        if a is not None and b is not None and abs(b - a) >= threshold:
            out.append(f"{cat} {TASKS.get(cat, '')}: {a:.0%} -> {b:.0%}")
    return out


def route(profiles: list[Profile], max_failure: float = 0.2) -> dict[str, str | None]:
    """For each task, the cheapest model whose failure rate is at most ``max_failure``.
    Models are taken cheapest first by ``cost_per_mtok``, else in the order given."""
    ordered = sorted(
        enumerate(profiles),
        key=lambda p: (p[1].cost_per_mtok is None, p[1].cost_per_mtok or 0, p[0]),
    )
    choice: dict[str, str | None] = {}
    for cat in sorted({c for p in profiles for c in p.categories}):
        choice[cat] = next(
            (
                p.model
                for _, p in ordered
                if (rate := p.failure_rate(cat)) is not None and rate <= max_failure
            ),
            None,
        )
    return choice
