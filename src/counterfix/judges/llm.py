"""An LLM used as a judge, and a panel of them that votes."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from json_repair import repair_json

from counterfix.core.caller import ModelCaller
from counterfix.core.record import CallRecord
from counterfix.judges.base import Judge, JudgeTask, Verdict, Vote

logger = logging.getLogger("counterfix")

SYSTEM = (
    "You are a careful, impartial evaluator of AI model outputs. Judge only what you are "
    "asked, using the material given and well-established facts. Do not be swayed by "
    "confident wording. Reply with one JSON object and nothing else."
)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def render(task: JudgeTask) -> str:
    """The user message sent to an LLM judge."""
    parts = [f"Task: {task.instructions.strip()}", ""]
    for key, value in task.material.items():
        parts.append(f"<{key}>\n{value.strip()}\n</{key}>")
    fields = {
        "label": "one of: " + ", ".join(f'"{label}"' for label in task.labels),
        "confidence": "a number from 0 to 1",
        "reason": "one short sentence",
        **task.extra_fields,
    }
    shape = ", ".join(f'"{k}": <{v}>' for k, v in fields.items())
    parts += ["", f"Reply with JSON only: {{{shape}}}"]
    return "\n".join(parts)


def parse_reply(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a judge's reply, tolerating fences and small breakage."""
    text = text.strip()
    match = _FENCE.match(text)
    if match:
        text = match.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else text
        value = repair_json(candidate, return_objects=True)
    return value if isinstance(value, dict) else None


def to_verdict(task: JudgeTask, judge: str, reply: dict[str, Any] | None, raw: str) -> Verdict:
    if reply is None:
        return Verdict(
            task=task.name,
            label="error",
            confidence=0,
            judge=judge,
            reason=f"unparseable reply: {raw[:120]!r}",
        )
    label = str(reply.get("label", "")).strip().lower()
    allowed = {lab.lower(): lab for lab in task.labels}
    if label not in allowed:
        return Verdict(
            task=task.name,
            label="error",
            confidence=0,
            judge=judge,
            reason=f"label {label!r} not in {list(task.labels)}",
        )
    try:
        confidence = float(reply.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    extra = {k: v for k, v in reply.items() if k not in ("label", "confidence", "reason")}
    return Verdict(
        task=task.name,
        label=allowed[label],
        confidence=min(max(confidence, 0.0), 1.0),
        reason=str(reply.get("reason", ""))[:500],
        judge=judge,
        extra=extra,
    )


class LLMJudge:
    """One model answering judge tasks as JSON. Never raises: failures become an
    ``error`` verdict so one flaky judge cannot break a diagnosis."""

    def __init__(
        self,
        caller: ModelCaller,
        *,
        max_tokens: int = 4096,
        json_mode: bool = True,
        temperature: float = 0.0,
        params: dict[str, Any] | None = None,
    ):
        self.caller = caller
        self.max_tokens = max_tokens  # generous: reasoning models think before the JSON
        self.json_mode = json_mode
        self.temperature = temperature
        self.params = dict(params or {})  # extra request fields, e.g. reasoning_effort
        self.system_role = True  # some models (Gemma on AI Studio) reject system messages

    @property
    def name(self) -> str:
        return self.caller.name

    def _messages(self, task: JudgeTask) -> list[dict[str, str]]:
        if self.system_role:
            return [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": render(task)},
            ]
        return [{"role": "user", "content": f"{SYSTEM}\n\n{render(task)}"}]

    def _call_once(self, task: JudgeTask) -> CallRecord:
        """One request with the current settings, no fallbacks."""
        params: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.params,
        }
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        return self.caller.complete(self._messages(task), **params)

    def _call(self, task: JudgeTask) -> CallRecord:
        for _ in range(3):  # at most: drop JSON mode once, drop the system role once
            try:
                return self._call_once(task)
            except Exception as exc:
                message = str(exc).lower()
                if (
                    self.system_role
                    and ("developer instruction" in message or "system" in message)
                    and "rate" not in message
                ):
                    logger.info("judge %s: system messages refused, merging", self.name)
                    self.system_role = False
                    continue
                if self.json_mode and ("response_format" in message or "json" in message):
                    logger.info("judge %s: JSON mode refused, retrying without it", self.name)
                    self.json_mode = False
                    continue
                # Some endpoints answer an option they do not support with a bare 500
                # (Gemma on Google's OpenAI-compatible API does this for JSON mode).
                if "internal" in message or "500" in message:
                    if self.json_mode:
                        logger.info("judge %s: server error, retrying without JSON mode", self.name)
                        self.json_mode = False
                        continue
                    if self.system_role:
                        logger.info("judge %s: server error, merging the system prompt", self.name)
                        self.system_role = False
                        continue
                raise
        return self._call_once(task)

    def ask(self, task: JudgeTask) -> Verdict:
        try:
            record = self._call(task)
        except Exception as exc:
            return Verdict(
                task=task.name,
                label="error",
                confidence=0,
                judge=self.name,
                reason=f"{type(exc).__name__}: {exc}"[:300],
            )
        raw = record.answer_text
        return to_verdict(task, self.name, parse_reply(raw), raw)

    def __repr__(self) -> str:
        return f"LLMJudge({self.name!r})"


class JudgePanel:
    """Several judges vote; the majority label wins.

    Confidence = share of valid votes for the winner x their mean confidence, so a 2-1
    split is reported as less certain than 3-0. Ties go to the label with the higher
    summed confidence. Judges from different model families reduce self-preference bias
    (Verga et al., 2024, "Replacing Judges with Juries").
    """

    def __init__(self, judges: Sequence[Judge], *, parallel: bool = True):
        if not judges:
            raise ValueError("a panel needs at least one judge")
        self.judges = list(judges)
        self.parallel = parallel

    @property
    def name(self) -> str:
        return "panel(" + ", ".join(j.name for j in self.judges) + ")"

    def ask(self, task: JudgeTask) -> Verdict:
        if self.parallel and len(self.judges) > 1:
            with ThreadPoolExecutor(max_workers=len(self.judges)) as pool:
                verdicts = list(pool.map(lambda j: j.ask(task), self.judges))
        else:
            verdicts = [j.ask(task) for j in self.judges]
        votes = [
            Vote(judge=v.judge, label=v.label, confidence=v.confidence, reason=v.reason)
            for v in verdicts
        ]
        valid = [v for v in verdicts if not v.failed]
        if not valid:
            return Verdict(
                task=task.name,
                label="error",
                confidence=0,
                judge=self.name,
                reason="every judge failed",
                votes=votes,
            )
        counts = Counter(v.label for v in valid)
        best = max(counts.values())
        tied = [label for label, n in counts.items() if n == best]
        winner = max(tied, key=lambda lab: sum(v.confidence for v in valid if v.label == lab))
        backing = [v for v in valid if v.label == winner]
        confidence = (len(backing) / len(valid)) * (
            sum(v.confidence for v in backing) / len(backing)
        )
        return Verdict(
            task=task.name,
            label=winner,
            confidence=round(confidence, 4),
            reason=backing[0].reason,
            judge=self.name,
            extra=backing[0].extra,
            votes=votes,
        )
