"""Token confidence: what the model's own token probabilities say about an answer.

When a provider returns log probabilities (``logprobs=True`` on OpenAI Chat Completions,
Ollama 0.12.11 or later, vLLM; not Groq, Anthropic or the OpenAI Responses API), each
record carries a summary in ``record.metadata["logprobs"]``: the number of tokens, the
mean and the lowest log probability, and the least likely tokens. The ``confidence``
metric turns it into a number, and a judge-confirmed F4 (invented facts) shows it as
extra evidence. Low confidence alone is not a failure: models are often unsure and
right, and sure and wrong; it is a signal, not a verdict.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from counterfix.core.record import CallRecord

LOGPROBS_KEY = "logprobs"


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def token_logprobs(choice: Any) -> list[tuple[str, float]]:
    """(token, logprob) pairs from a chat-completion choice (or a stream chunk's choice)."""
    content = _get(_get(choice, "logprobs"), "content") or []
    out: list[tuple[str, float]] = []
    for item in content:
        lp = _get(item, "logprob")
        if isinstance(lp, (int, float)) and not isinstance(lp, bool) and math.isfinite(lp):
            out.append((str(_get(item, "token") or ""), float(lp)))
    return out


def summarize(pairs: Iterable[tuple[str, float]], keep: int = 5) -> dict[str, Any] | None:
    """{"n", "mean", "min", "lowest": [[token, logprob], ...]} or None with no tokens."""
    items = list(pairs)
    if not items:
        return None
    values = [lp for _, lp in items]
    lowest = sorted(items, key=lambda p: p[1])[:keep]
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "lowest": [[t, round(lp, 4)] for t, lp in lowest],
    }


def _as_record(target: Any) -> CallRecord:
    """A CallRecord from a record, a Diagnosis or a guarded response."""
    if isinstance(target, CallRecord):
        return target
    from counterfix.diagnose import _target

    return _target(target)[0]


def of(record: Any) -> dict[str, Any] | None:
    """The token-probability summary of a record, a diagnosis or a guarded response, if the
    provider returned logprobs (ask with ``logprobs=True``)."""
    found = _as_record(record).metadata.get(LOGPROBS_KEY)
    return found if isinstance(found, dict) and found.get("n") else None


def mean_probability(record: Any) -> float | None:
    """exp(mean token logprob): the geometric-mean probability of the answer's tokens."""
    s = of(record)
    return math.exp(s["mean"]) if s else None


def describe(record: Any) -> str | None:
    """'token confidence 0.62 (geometric mean); least likely: "1889" 0.03, ...'."""
    s = of(record)
    if not s:
        return None
    low = ", ".join(f"{t!r} {math.exp(lp):.2f}" for t, lp in s["lowest"][:3])
    return f"token confidence {math.exp(s['mean']):.2f} over {s['n']} tokens; least likely: {low}"
