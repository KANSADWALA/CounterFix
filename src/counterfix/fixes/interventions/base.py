"""The shape of an intervention and the small helpers that build prompt changes."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from counterfix.core.diagnosis import Finding
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType, PrescriptionKind

F = FailureType
Change = tuple[list[Message], dict[str, Any]]


@dataclass(frozen=True)
class Intervention:
    """One change aimed at one cause. ``apply`` returns new (messages, params), or None
    when it does not fit this call. ``metadata`` (RAG fixes) returns what the re-run's
    record metadata must say instead, e.g. the passages the new prompt holds."""

    name: str
    failure_type: FailureType
    kind: PrescriptionKind
    description: str
    apply: Callable[[CallRecord, Finding], Change | None]
    metadata: Callable[[CallRecord, Finding], dict[str, Any]] | None = None


def _with_system(messages: list[Message], text: str) -> list[Message]:
    out = copy.deepcopy(messages)
    if out and out[0].role in ("system", "developer") and isinstance(out[0].content, str):
        out[0].content = f"{out[0].content}\n\n{text}"
    else:
        out.insert(0, Message(role="system", content=text))
    return out


def _with_user_suffix(messages: list[Message], text: str) -> list[Message] | None:
    out = copy.deepcopy(messages)
    for m in reversed(out):
        if m.role == "user":
            if isinstance(m.content, str):
                m.content = f"{m.content}\n\n{text}"
            elif isinstance(m.content, list):
                m.content = [*m.content, {"type": "text", "text": text}]
            else:
                return None
            return out
    return None


def _system(text: str) -> Callable[[CallRecord, Finding], Change | None]:
    return lambda r, f: (_with_system(r.messages, text), dict(r.params))


def _suffix(text: str) -> Callable[[CallRecord, Finding], Change | None]:
    def apply(r: CallRecord, f: Finding) -> Change | None:
        messages = _with_user_suffix(r.messages, text)
        return None if messages is None else (messages, dict(r.params))

    return apply
