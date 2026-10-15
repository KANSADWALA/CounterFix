"""Optional judge back-end: TypeSafe AI's Jev decision model (``pip install counterfix[jev]``).

Jev answers typed questions with calibrated probabilities instead of free text, so it is
fast and cheap, but it cannot return extra fields such as quoted claims or the F8
expression. Checks that need those fields get less detail (F8 cannot run on Jev).

API used, from docs.typesafe.ai (checked 15 Oct 2026): ``TypeSafeClient().system_one(
state=..., questions={name: Choice(instructions=..., criteria={label: None})})``; the
answer is ``response.choices[name]`` with ``.choice`` and ``.probabilities``. The key is
read from ``TYPESAFE_API_KEY``. Off unless you create a ``JevJudge`` yourself.
"""

from __future__ import annotations

from typing import Any

from counterfix.judges.base import JudgeTask, Verdict

QUESTION = "verdict"


class JevJudge:
    """A Judge backed by Jev. Never raises; failures become an ``error`` verdict."""

    def __init__(self, client: Any = None):
        try:
            import typesafe_sdk
        except ImportError as exc:
            raise ImportError(
                'JevJudge needs the TypeSafe SDK: pip install "counterfix[jev]"'
            ) from exc
        self._sdk = typesafe_sdk
        self.client = client if client is not None else typesafe_sdk.TypeSafeClient()

    @property
    def name(self) -> str:
        return "typesafe:jev"

    def ask(self, task: JudgeTask) -> Verdict:
        try:
            question = self._sdk.Choice(
                instructions=task.instructions,
                criteria={label: None for label in task.labels},
            )
            response = self.client.system_one(
                state=dict(task.material), questions={QUESTION: question}
            )
            answer = response.choices[QUESTION]
            label = str(answer.choice)
            probabilities = dict(getattr(answer, "probabilities", {}) or {})
            confidence = float(probabilities.get(label, getattr(answer, "confidence", 0.5)))
        except Exception as exc:
            return Verdict(
                task=task.name,
                label="error",
                confidence=0,
                judge=self.name,
                reason=f"{type(exc).__name__}: {exc}"[:300],
            )
        if label not in task.labels:
            return Verdict(
                task=task.name,
                label="error",
                confidence=0,
                judge=self.name,
                reason=f"label {label!r} not in {list(task.labels)}",
            )
        return Verdict(
            task=task.name,
            label=label,
            confidence=min(max(confidence, 0.0), 1.0),
            reason="Jev decision (no free-text reason)",
            judge=self.name,
            extra={"probabilities": probabilities},
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
