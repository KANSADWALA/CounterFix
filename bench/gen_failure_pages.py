"""Writes docs/failures/F1.md ... F14.md from the engine's interventions and escalations.

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
    "F10": (
        "RAG: the retrieved passages did not hold the answer, so the model could only guess "
        "or decline. The retriever is at fault, not the model.",
        "Free with gold labels: none of the gold passages was retrieved. Without labels, a "
        "judge decides whether the passages hold what the question needs "
        "(`context_lacks_answer`). F10 takes precedence over F4: the unsupported claim is "
        "the symptom.",
    ),
    "F11": (
        "RAG: the answer was in the retrieved passages, but the model contradicted it, left "
        "it out, or said it was not there.",
        "Judge: finds the passage that holds the answer, then checks whether the model answer "
        "agrees with it (`answer_ignores_context`). The passage number is passed to the fix "
        "search, which moves that passage first.",
    ),
    "F12": (
        "Code: the code in the answer fails its tests. counterfix names what went wrong: no "
        "code, a syntax error, the wrong function name or signature, a disallowed import, an "
        "error when run, wrong results, or a timeout. Code cut off by the token limit is F1.",
        "With the `tests` metric (opt-in, it runs the code in a sandbox): cheapest check "
        "first, `ast` parse, the name and signature the tests call, the import allow-list, "
        "then each test on its own in a subprocess or a Docker container with no network. "
        "See [Code evaluation](../code.md).",
    ),
    "F13": (
        "Prompt injection: text the app gave the model but did not write itself (retrieved "
        "passages, web pages, tool results) carried an instruction aimed at the AI, and the "
        "answer carried it out: it said the planted text, added a link, switched language "
        "or format, or dropped the user's task.",
        "Judge (`InjectionCheck`): finds a planted instruction in the retrieved text or tool "
        "results and decides whether the answer followed it; skips calls without such text. "
        "For probes, a canary planted in the text and found in the answer is exact and free "
        "(`CanaryCheck`, the `injection` metric and benchmark). F13 takes precedence over "
        "F4, F6 and F11. See [Safety](../safety.md).",
    ),
    "F14": (
        "Tool use went wrong: the model called a tool that was not offered, the wrong tool, "
        "no tool when one was needed (or one when none was), gave arguments with wrong "
        "values, repeated the same call (a loop), or an agent run did not reach the user's "
        "goal. Arguments that are not JSON or break the schema are F2.",
        "Live and free (`ToolUseCheck`): unknown tool, loops, a tool the item expects that "
        "was never called, too many steps, on single calls and on agent runs "
        "(`counterfix.agents.record`). With an expected call, the `tool_call` metric names "
        "the cause. Judge (`AgentTaskCheck`): did the run complete the task, and if not, "
        "why. See [Agents and tool calls](../agents.md).",
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
        lines.append("None yet: no prompt-level change is verified for this type so far.")
    esc = ESCALATION.get(ft, [])
    if esc:
        lines += ["", "## When no fix verifies", "", *[f"- {e[0].upper()}{e[1:]}" for e in esc]]
    (OUT / f"{ft.value}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ft.value)
