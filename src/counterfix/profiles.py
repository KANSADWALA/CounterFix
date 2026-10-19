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
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.caller import DailyLimitError, ModelCaller
from counterfix.core.record import CallRecord, Message

PROBES = Path(__file__).with_name("data") / "probes.jsonl"
SHIPPED = Path(__file__).with_name("data") / "profiles"  # measured profiles, see bench/

TASKS = {
    "F1": "long answers under a token limit",
    "F2": "JSON and structured output",
    "F3": "sensitive but harmless requests",
    "F4": "facts it may not know",
    "F5": "questions with a false premise",
    "F6": "exact formatting instructions",
    "F7": "reworded prompts",
    "F8": "maths and counting",
    "F9": "vague requests",
}


class Profile(BaseModel):
    """One model's measured behaviour on the probe suite, plus what fixed it."""

    model: str
    """The model id as the provider reports it (what ``fix()`` looks up)."""
    spec: str | None = None
    """``provider:model`` it was measured through, e.g. ``groq:openai/gpt-oss-120b``."""
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    judge: str | None = None
    cost_per_mtok: float | None = None
    """Blended price per million tokens, for ``route`` (set it yourself; optional)."""
    categories: dict[str, list[int]] = Field(default_factory=dict)
    """Probe category -> [failed, probes]."""
    fixes: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    """Failure type -> intervention -> [verified, tried]."""
    errors: int = 0
    """Probes that got a provider error (not counted in ``categories``)."""
    judge_skips: int = 0
    """Judge checks that could not run (judge errors, limits); failure rates may be low."""

    def failure_rate(self, category: str) -> float | None:
        failed, n = self.categories.get(category, [0, 0])
        return failed / n if n else None

    def fix_order(self, failure_type: str, names: list[str]) -> list[str]:
        """``names`` reordered: best verified rate first; never-verified after 2 tries last;
        untried ones keep their default place in between."""
        stats = self.fixes.get(failure_type, {})

        def key(item: tuple[int, str]) -> tuple[int, float, int]:
            i, name = item
            verified, tried = stats.get(name, [0, 0])
            if not tried:
                return (1, 0.0, i)
            if verified == 0 and tried >= 2:
                return (2, 0.0, i)
            return (0, -verified / tried, i)

        return [name for _, name in sorted(enumerate(names), key=key)]

    def learn(self, result: Any) -> None:
        """Add one ``FixResult``'s attempts to the fix statistics."""
        for attempt in result.attempts:
            if attempt.failure_type is None:
                continue
            row = self.fixes.setdefault(attempt.failure_type.value, {})
            verified, tried = row.get(attempt.name, [0, 0])
            row[attempt.name] = [verified + attempt.verified, tried + 1]


def home() -> Path:
    return Path(os.environ.get("COUNTERFIX_HOME") or Path.home() / ".counterfix")


def profile_path(model: str) -> Path:
    return home() / "profiles" / (re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".json")


def save(profile: Profile, path: str | Path | None = None) -> Path:
    target = Path(path) if path else profile_path(profile.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return target


def load(model_or_path: str | Path) -> Profile | None:
    """A profile from a file path, or for a model: yours (``~/.counterfix/profiles``) first,
    then the ones shipped with counterfix. ``provider:model`` and bare model ids both work.
    None if there is none."""
    path = Path(model_or_path)
    if path.suffix == ".json" and path.is_file():
        return Profile.model_validate_json(path.read_text(encoding="utf-8"))
    name = str(model_or_path)
    names = [name, name.split(":", 1)[1]] if ":" in name else [name]
    for model in names:
        for candidate in (profile_path(model), SHIPPED / profile_path(model).name):
            if candidate.is_file():
                return Profile.model_validate_json(candidate.read_text(encoding="utf-8"))
    return None


def load_probes(path: str | Path | None = None) -> list[dict[str, Any]]:
    rows = Path(path or PROBES).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in rows if line.strip()]


def probe_record(probe: dict[str, Any], record: CallRecord) -> CallRecord:
    """Attach what a probe knows (schema, right answer) so the checks can use it."""
    from counterfix.detectors.fmt import SCHEMA_METADATA_KEY
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
    max_runs: int = 9,
    progress: Any = None,
) -> Profile:
    """Run the probe suite on one model and measure it (``fix=True`` also learns fixes)."""
    from counterfix.diagnose import diagnose
    from counterfix.engine import fix as run_fix
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
            from counterfix.detectors import FragilityCheck

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
