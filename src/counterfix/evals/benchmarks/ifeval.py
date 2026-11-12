"""IFEval scored with the official checks (Google's code as shipped in lm-evaluation-harness),
so the numbers match the published metric: strict prompt-level accuracy.

Needs the ``benchmarks`` extra (``lm-eval[ifeval]``). When an answer misses, the failed
instruction ids are the reason, and the miss points at F6 (dropped instruction), so the
fix search tries counterfix's F6 fixes (an instruction checklist at the end of the prompt).
"""

from __future__ import annotations

from typing import Any

from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.evals.metrics.base import Metric, MetricResult


def _utils() -> Any:
    try:
        from lm_eval.tasks.ifeval import utils
    except ImportError as exc:
        raise ImportError('IFEval needs lm-eval: pip install "counterfix[benchmarks]"') from exc
    return utils


class IfEvalStrict(Metric):
    """1.0 when the answer follows every instruction of the IFEval prompt (strict).
    ``loose=True`` uses IFEval's loose variant (first/last line and markdown removed)."""

    __test__ = False
    name = "ifeval"
    default_threshold = 1.0
    failure_type = FailureType.DROPPED_INSTRUCTION

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        if not isinstance(reference, dict) or "instruction_id_list" not in reference:
            return None, "no IFEval instructions to check"
        utils = _utils()
        example = utils.InputExample(
            key=reference.get("key", 0),
            instruction_id_list=list(reference["instruction_id_list"]),
            prompt=reference["prompt"],
            kwargs=[dict(k or {}) for k in reference["kwargs"]],
        )
        test = (
            utils.test_instruction_following_loose
            if self.options.get("loose")
            else utils.test_instruction_following_strict
        )
        out = test(example, record.answer_text)
        missed = [
            i
            for i, ok in zip(out.instruction_id_list, out.follow_instruction_list, strict=False)
            if not ok
        ]
        kept = len(out.follow_instruction_list) - len(missed)
        if not missed:
            return 1.0, f"followed all {kept} instruction(s)"
        return 0.0, f"missed {', '.join(missed)} ({kept}/{len(missed) + kept} followed)"


def score_instructions(answer: str, reference: dict[str, Any]) -> MetricResult:
    """IFEval verdict for one answer (helper for scripts)."""
    return IfEvalStrict().score(CallRecord(response_text=answer), reference)
