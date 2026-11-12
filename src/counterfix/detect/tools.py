"""F14 tool misuse, from what the run did (live, free, exact).

Works on an agent run (``counterfix.detect.agents.record(task, run)``) and on a single model
call that offered tools: a call to a tool that was not offered, the same call repeated
(a loop), a tool the item expects that was never called, too many steps. Arguments that
are not JSON or break the tool's schema stay F2 (format). Whether the run reached the
user's goal is a judge question: :class:`AgentTaskCheck`.
"""

from __future__ import annotations

from typing import Any

from counterfix.core.config import Config
from counterfix.core.detector import SkipCheck
from counterfix.core.diagnosis import Evidence, Finding, Prescription
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType, PrescriptionKind
from counterfix.detect.judged import _JudgeCheck
from counterfix.judges import tasks

_FIXES = {
    "unknown_tool": "Name the tools exactly in the prompt, or offer the tool it reached for",
    "loop": "Tell the model not to repeat a call that did not help, cap the steps, and return "
    "tool errors as readable messages",
    "missing_call": "Say which tool the request needs (or set tool_choice), and describe each "
    "tool's purpose and arguments clearly",
    "too_many_steps": "Cap the steps and tell the model to answer once it has what it needs",
    "wrong_tool": "Make the tool descriptions say when to use each tool, or set tool_choice",
    "bad_arguments": "Describe each argument (type, format, an example) in the tool schema",
    "no_call": "Set tool_choice to required (or to the tool), and say the tool must be used",
    "extra_call": "Say when no tool is needed, or set tool_choice to none for such requests",
}


_READ_RUN = "Read the run: check the tool descriptions, the arguments and the results"


def tool_finding(
    cause: str, detail: str, detector: str, confidence: float = 0.95, **data: Any
) -> Finding:
    return Finding(
        failure_type=FailureType.TOOL_MISUSE,
        confidence=confidence,
        evidence=[Evidence(detector=detector, detail=detail, data={"cause": cause, **data})],
        prescription=Prescription(
            kind=PrescriptionKind.PROMPT_CHANGE,
            description=_FIXES.get(cause, "Describe the tools more clearly"),
        ),
    )


class ToolUseCheck:
    """F14 from the run itself: unknown tool, loop, missing expected call, too many steps."""

    name = "tool_use"
    covers = (FailureType.TOOL_MISUSE,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        from counterfix.detect import agents

        traj = agents.of(record)
        if traj is None:
            if not record.params.get("tools") or not record.tool_calls:
                raise SkipCheck("no tools offered or no tool calls on this call")
            traj = agents.from_records([record])
        expect = record.metadata.get(agents.EXPECT_KEY)
        limit = record.metadata.get("max_steps")
        issues = agents.problems(
            traj,
            expect=expect if isinstance(expect, list) else None,
            max_steps=limit if isinstance(limit, int) else None,
        )
        return [
            tool_finding(
                i["cause"],
                i["detail"],
                self.name,
                **{k: v for k, v in i.items() if k not in ("cause", "detail")},
            )
            for i in issues
        ]


class AgentTaskCheck(_JudgeCheck):
    """F14 by judge: the run did not reach the user's goal, and why (wrong tool, bad
    arguments, a tool result ignored or misread, a step missing, gave up)."""

    name = "judge_agent_task"
    covers = (FailureType.TOOL_MISUSE,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        from counterfix.detect import agents

        traj = agents.of(record)
        if traj is None or not (traj.calls or traj.final_answer):
            raise SkipCheck("no agent run on this call")
        verdict = self._ask(tasks.agent_task(traj))
        if verdict.label != "not_completed":
            return []
        cause = str(verdict.extra.get("cause") or "other")
        why = f"the run did not complete the task ({cause})"
        return [
            Finding(
                failure_type=FailureType.TOOL_MISUSE,
                confidence=round(verdict.confidence * 0.85, 4),
                evidence=[self._evidence(verdict, f"{why}: {verdict.reason}", cause=cause)],
                prescription=Prescription(
                    kind=PrescriptionKind.PROMPT_CHANGE,
                    description=_FIXES.get(cause, _READ_RUN),
                ),
            )
        ]
