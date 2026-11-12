"""Interventions for code that fails its tests (F12)."""

from __future__ import annotations

import copy
from typing import Any

from counterfix.core.diagnosis import Finding
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import PrescriptionKind
from counterfix.fixes.interventions.base import (
    Change,
    F,
    Intervention,
    _suffix,
)


def _code_check(r: CallRecord) -> dict[str, Any]:
    check = r.metadata.get("code_check")
    return check if isinstance(check, dict) else {}


def _error_feedback(r: CallRecord, f: Finding) -> Change | None:
    """Send the failure back once, with the model's own code, and ask for a fix."""
    check = _code_check(r)
    if not check.get("cause") or not r.response_text:
        return None
    from counterfix.detect.code import CAUSE_TEXT

    what = CAUSE_TEXT.get(str(check["cause"]), "the code failed")
    detail = f"\n\n{check['detail']}" if check.get("detail") else ""
    note = (
        f"Your code failed: {what}.{detail}\n\nFix it. Reply with the complete corrected "
        "code in one ```python block."
    )
    messages = [*copy.deepcopy(r.messages), Message(role="assistant", content=r.response_text)]
    messages.append(Message(role="user", content=note))
    return messages, dict(r.params)


def _restate_signature(r: CallRecord, f: Finding) -> Change | None:
    check = _code_check(r)
    if check.get("cause") not in ("name", "signature"):
        return None
    signature = check.get("signature")
    name = check.get("entry_point")
    if signature:
        text = f"Write the function with exactly this name and signature:\n{signature}"
    elif name:
        text = f"The function must be named {name} exactly."
    else:
        return None
    return _suffix(text)(r, f)


def _check_examples(r: CallRecord, f: Finding) -> Change | None:
    if _code_check(r).get("cause") not in ("wrong_output", "runtime"):
        return None
    return _suffix(
        "Before you write the code, work through each example in the task by hand and check "
        "that your approach gives exactly the expected result, including edge cases (empty "
        "input, one element, negative numbers). Then reply with the complete code in one "
        "```python block."
    )(r, f)


def _one_block(r: CallRecord, f: Finding) -> Change | None:
    if _code_check(r).get("cause") not in ("no_code", "syntax", "name", None):
        return None
    return _suffix(
        "Reply with the complete, runnable code in a single ```python code block, with the "
        "imports it needs, and no other code blocks."
    )(r, f)


CODE: list[Intervention] = [
    Intervention(
        "error_feedback",
        F.CODE_FAILURE,
        PrescriptionKind.PROMPT_CHANGE,
        "Send the error or failed test back once and ask for corrected code",
        _error_feedback,
    ),
    Intervention(
        "check_examples",
        F.CODE_FAILURE,
        PrescriptionKind.PROMPT_CHANGE,
        "Ask the model to work the task's examples by hand before writing the code",
        _check_examples,
    ),
    Intervention(
        "restate_signature",
        F.CODE_FAILURE,
        PrescriptionKind.PROMPT_CHANGE,
        "Restate the exact function name and signature the tests call",
        _restate_signature,
    ),
    Intervention(
        "one_code_block",
        F.CODE_FAILURE,
        PrescriptionKind.PROMPT_CHANGE,
        "Ask for the whole program in one code block",
        _one_block,
    ),
]
