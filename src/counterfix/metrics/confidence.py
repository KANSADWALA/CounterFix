"""Token confidence as a metric (needs a provider that returns logprobs)."""

from __future__ import annotations

import math
from typing import Any

from counterfix.core.record import CallRecord
from counterfix.metrics.base import Metric


class Confidence(Metric):
    """The answer's geometric-mean token probability, exp(mean logprob), from 0 to 1;
    ``use="min"`` gives the least likely token's probability instead. Not scored when the
    provider returned no logprobs (ask with ``logprobs=True``; Groq, Anthropic and the
    OpenAI Responses API have none). No threshold by default: it is a signal, so
    ``confidence:0.5`` fails only answers below 0.5 if you ask for that."""

    name = "confidence"
    needs_reference = False

    def compute(self, record: CallRecord, reference: Any) -> tuple[float | None, str]:
        from counterfix.confidence import of

        s = of(record)
        if s is None:
            asked = bool(record.params.get("logprobs"))
            why = (
                "the provider returned no logprobs"
                if asked
                else "no logprobs (ask with --param logprobs=true)"
            )
            return None, why
        use_min = self.options.get("use") == "min"
        value = math.exp(s["min"] if use_min else s["mean"])
        what = "least likely token" if use_min else f"geometric mean over {s['n']} tokens"
        return value, f"{value:.2f} ({what})"
