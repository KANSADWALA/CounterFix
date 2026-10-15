"""The provider-neutral record of one LLM call.

Everything inside counterfix works on a :class:`CallRecord`. Adapters convert each
provider's request and response into one, so the core never imports a provider SDK.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One chat message. ``content`` may be text, a list of parts, or None."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None

    @property
    def text(self) -> str:
        """The message's text, joining text parts when content is a list."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts: list[str] = []
            for part in self.content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
        return str(self.content)


class Usage(BaseModel):
    """Token counts, when the provider returns them."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class CallRecord(BaseModel):
    """A single LLM call: what was asked, with which settings, and what came back."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    provider: str | None = None
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    response_text: str | None = None
    reasoning_text: str | None = None
    """Hidden "thinking" returned by reasoning models, when the provider exposes it."""
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: Usage | None = None
    latency_ms: float | None = None

    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def system_text(self) -> str:
        """All system messages joined, or an empty string."""
        return "\n".join(m.text for m in self.messages if m.role in ("system", "developer"))

    @property
    def prompt_text(self) -> str:
        """The text of the last user message, or an empty string."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.text
        return ""

    @property
    def answer_text(self) -> str:
        """The visible answer, with inline reasoning removed.

        Handles the tag styles models use for inline thinking: <think> (Qwen, DeepSeek),
        <thought> (Gemma) and <thinking>. An unclosed tag (cut off mid-thought) removes
        everything after it.
        """
        return _THINK.sub("", self.response_text or "").strip()

    @property
    def has_reasoning(self) -> bool:
        """True when the model spent tokens on reasoning (separate field or inline tags)."""
        return bool(
            self.reasoning_text
            or (self.usage is not None and (self.usage.reasoning_tokens or 0) > 0)
            or _THINK_OPEN.search(self.response_text or "")
        )

    @property
    def max_tokens(self) -> int | None:
        """The output-token limit the call used, if one was set."""
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = self.params.get(key)
            if isinstance(value, int):
                return value
        return None


_TAGS = r"(think|thought|thinking)"
_THINK = re.compile(rf"<{_TAGS}>.*?(</\1>|$)", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(rf"<{_TAGS}>", re.IGNORECASE)
