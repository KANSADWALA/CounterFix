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
from typing import Any

from counterfix.core.convert import LOGPROBS_KEY as LOGPROBS_KEY
from counterfix.core.convert import summarize as summarize
from counterfix.core.convert import token_logprobs as token_logprobs
from counterfix.core.record import CallRecord


def _as_record(target: Any) -> CallRecord:
    """A CallRecord from a record, a Diagnosis or a guarded response."""
    if isinstance(target, CallRecord):
        return target
    from counterfix.core.attach import target_record

    return target_record(target)[0]


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
