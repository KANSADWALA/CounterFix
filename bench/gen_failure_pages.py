"""Writes docs/failures/F1.md ... F9.md from the engine's interventions and escalations.

    python bench/gen_failure_pages.py

tests/test_profiles.py checks every intervention is listed, so rerun this after adding one.
"""

from pathlib import Path

from counterfix.core.types import FailureType
from counterfix.engine import ESCALATION, INTERVENTIONS

OUT = Path(__file__).resolve().parents[1] / "docs" / "failures"
OUT.mkdir(parents=True, exist_ok=True)

TEXT = {
    "F1": (
        "The answer stopped because it hit `max_tokens`, including reasoning models that "
        "spend the whole budget thinking and return an empty answer.",
        "Live, free: `finish_reason` of `length` / `max_tokens`; when the provider gives no "
        "stop reason, inferred from tokens used and an answer that ends mid-sentence "
        "(marked `inferred`, lower confidence).",
    ),
    "F2": (
        "The answer was meant to be JSON (or match a schema, or call a tool with valid "
        "arguments) and is not.",
        "Live, free: parses the answer (code fences and `<think>` blocks removed), validates "
        "it against your schema, and checks tool-call arguments against the tool's schema.",
    ),
    "F3": (
        "The model refused a harmless request (a school play villain, killing a hung process).",
        "Live: refusal phrases near the start of the answer. Judge: confirms the request was "
        "actually harmless, because phrases alone misfire.",
    ),
    "F4": (
        "The answer states facts that are false or unsupported, usually because the model "
        "did not know.",
        "Judge: lists the factual claims and marks false or unsupported ones; with a "
        "`source` (see `counterfix.context`) it checks the answer against that text. "
        "`SamplingCheck` can also re-ask the model and flag answers that change.",
    ),
    "F5": (
        "The question assumed something false and the model went along with it "
        '("Why did Einstein win three Nobel Prizes?").',
        "Judge: finds a false premise in the question and checks whether the answer accepted it.",
    ),
    "F6": (
        "The answer ignored an instruction: a word or sentence count, a format, a banned "
        'word, "mention the price".',
        "Live, free and exact: instructions that code can check. Judge: content "
        "instructions code cannot check.",
    ),
    "F7": (
        "Small rewordings of the same prompt give different answers.",
        "Not live (it needs re-asks): `FragilityCheck` asks the same thing in 3 other "
        'wordings (yours, from `metadata["variants"]`, or written by the model) and '
        "flags the call when a short answer changes meaning. Pass it to "
        "`diagnose(detectors=[...])`; profiles and `fix()` use it for F7. The linter's L2 "
        "flags conflicting constraints, a common cause.",
    ),
    "F8": (
        "Arithmetic, counting or date maths went wrong.",
        "Judge writes one expression for the claimed result; counterfix evaluates it with "
        "a whitelist-only evaluator and compares, so the judge's own arithmetic is never "
        "trusted.",
    ),
    "F9": (
        "The request had several readings and the model silently picked one.",
        "Judge: lists the readings and checks whether the answer named or asked about its "
        "choice. The linter's L6 flags requests missing who, what or which.",
    ),
}

for ft in [t for t in FailureType if t.is_model_failure]:
    what, how = TEXT[ft.value]
    ivs = [i for i in INTERVENTIONS if i.failure_type is ft]
    lines = [f"# {ft.value} {ft.label}", "", what, "", "## How counterfix finds it", "", how, ""]
    lines += ["## Fixes it tries", ""]
    if ivs:
        lines += [
            "In this order (a [model profile](../profiles.md) can reorder them). Each one is",
            "kept only if it passes 2 of 3 re-runs.",
            "",
            "| Intervention | Change |",
            "| --- | --- |",
        ]
        lines += [f"| `{i.name}` | {i.description} |" for i in ivs]
    else:
        lines.append("None yet: no prompt-level change is verified for this type in v0.1.")
    esc = ESCALATION.get(ft, [])
    if esc:
        lines += ["", "## When no fix verifies", "", *[f"- {e[0].upper()}{e[1:]}" for e in esc]]
    (OUT / f"{ft.value}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ft.value)
