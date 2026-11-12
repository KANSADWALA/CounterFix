"""Reading Anthropic Messages API calls (``client.messages.create``) into a CallRecord.

``guard(Anthropic())`` picks this up on its own: a client with ``messages`` and no
``chat`` is treated as an Anthropic client. The SDK is never imported; anything with the
same response shape works (``content`` blocks, ``stop_reason``, ``usage``).

Tool calls are stored in the OpenAI shape the detectors read; F2 checks ``tool_use``
arguments against each tool's ``input_schema``. Thinking blocks are
recorded as reasoning text. ``stream=True`` calls are checked when the stream ends;
the ``messages.stream()`` helper is not intercepted.
"""

from __future__ import annotations

from counterfix.core.caller import AnthropicCaller
from counterfix.core.convert import _blocks, record_from_anthropic

# Retries may only change these request fields (other fixes are OpenAI parameters).
ANTHROPIC_RETRY_PARAMS = frozenset({"max_tokens"})

__all__ = ["ANTHROPIC_RETRY_PARAMS", "AnthropicCaller", "_blocks", "record_from_anthropic"]
