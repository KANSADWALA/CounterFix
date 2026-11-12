"""Checking streamed calls without delaying a single token.

The wrapped stream yields every chunk the moment it arrives and keeps a copy. When the
stream ends, the copy is assembled into one response and checked as usual; the result
is on ``stream.diagnosis`` (and goes to ``Config.on_diagnosis``). Streams are never
retried: the tokens have already reached the user. Works for OpenAI-compatible chunks,
OpenAI Responses events and Anthropic Messages events, sync and async.

The wrapper also notes when chunks arrive, so the record carries ``timing``: time to the
first answer chunk (TTFT; answer text or tool-call arguments), to the first reasoning
chunk, and to the last answer chunk.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from counterfix.core.convert import _get
from counterfix.core.record import Timing


class _OpenAIParts:
    def __init__(self) -> None:
        self.model: Any = None
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.finish: Any = None
        self.usage: Any = None
        self.tools: dict[int, dict[str, Any]] = {}
        self.logprobs: list[Any] = []

    def feed(self, chunk: Any) -> tuple[bool, bool]:
        """Add one chunk; returns (carried answer text, carried reasoning)."""
        text_seen = reasoning_seen = False
        self.model = _get(chunk, "model") or self.model
        self.usage = _get(chunk, "usage") or self.usage
        for choice in (_get(chunk, "choices") or [])[:1]:
            delta = _get(choice, "delta")
            self.finish = _get(choice, "finish_reason") or self.finish
            self.logprobs += _get(_get(choice, "logprobs"), "content") or []
            if (text := _get(delta, "content")) is not None:
                self.text.append(text)
                text_seen = text_seen or bool(text)
            if r := _get(delta, "reasoning") or _get(delta, "reasoning_content"):
                self.reasoning.append(r)
                reasoning_seen = True
            for call in _get(delta, "tool_calls") or []:
                slot = self.tools.setdefault(_get(call, "index", 0), {"name": "", "args": ""})
                slot["id"] = _get(call, "id") or slot.get("id")
                fn = _get(call, "function")
                slot["name"] += _get(fn, "name") or ""
                slot["args"] += _get(fn, "arguments") or ""
                text_seen = True  # a tool call is the answer's output too
        return text_seen, reasoning_seen

    def response(self) -> Any:
        calls = [
            {
                "id": t.get("id"),
                "type": "function",
                "function": {"name": t["name"], "arguments": t["args"]},
            }
            for _, t in sorted(self.tools.items())
        ]
        message = SimpleNamespace(
            role="assistant",
            content="".join(self.text) if self.text else None,
            reasoning="".join(self.reasoning) or None,
            tool_calls=calls or None,
        )
        logprobs = SimpleNamespace(content=self.logprobs) if self.logprobs else None
        choice = SimpleNamespace(
            index=0, message=message, finish_reason=self.finish, logprobs=logprobs
        )
        return SimpleNamespace(model=self.model, choices=[choice], usage=self.usage)


class _AnthropicParts:
    def __init__(self) -> None:
        self.model: Any = None
        self.blocks: list[dict[str, Any]] = []
        self.stop: Any = None
        self.input_tokens: Any = None
        self.output_tokens: Any = None

    def feed(self, event: Any) -> tuple[bool, bool]:
        """Add one event; returns (carried answer text, carried reasoning)."""
        text_seen = reasoning_seen = False
        kind = _get(event, "type")
        if kind == "message_start":
            message = _get(event, "message")
            self.model = _get(message, "model")
            self.input_tokens = _get(_get(message, "usage"), "input_tokens")
        elif kind == "content_block_start":
            block = _get(event, "content_block")
            self.blocks.append(
                {
                    "type": _get(block, "type"),
                    "id": _get(block, "id"),
                    "name": _get(block, "name"),
                    "text": "",
                    "thinking": "",
                    "json": "",
                }
            )
        elif kind == "content_block_delta" and self.blocks:
            delta, block = _get(event, "delta"), self.blocks[-1]
            text, thinking = _get(delta, "text") or "", _get(delta, "thinking") or ""
            block["text"] += text
            block["thinking"] += thinking
            block["json"] += _get(delta, "partial_json") or ""
            text_seen = bool(text) or bool(_get(delta, "partial_json"))
            reasoning_seen = bool(thinking)
        elif kind == "message_delta":
            self.stop = _get(_get(event, "delta"), "stop_reason") or self.stop
            self.output_tokens = _get(_get(event, "usage"), "output_tokens") or self.output_tokens
        return text_seen, reasoning_seen

    def response(self) -> Any:
        import json

        content = []
        for b in self.blocks:
            if b["type"] == "tool_use":
                try:
                    args: Any = json.loads(b["json"]) if b["json"] else {}
                except json.JSONDecodeError:
                    args = b["json"]  # F2 reports arguments that are not JSON
                content.append(
                    {"type": "tool_use", "id": b["id"], "name": b["name"], "input": args}
                )
            elif b["type"] in ("text", "thinking"):
                content.append({"type": b["type"], b["type"]: b[b["type"]]})
        usage = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        return {"model": self.model, "content": content, "stop_reason": self.stop, "usage": usage}


class GuardedStream:
    """Yields the provider's chunks unchanged; checks the whole answer at the end."""

    def __init__(
        self,
        stream: Any,
        kind: str | bool,
        done: Callable[[Any, Timing], Any],
        start: float | None = None,
    ):
        self._stream = stream
        # kind: "chat", "anthropic" or "responses" (True / False still mean anthropic / chat)
        if kind == "responses":
            from counterfix.adapters.responses import ResponsesParts

            self._parts: Any = ResponsesParts()
        elif kind is True or kind == "anthropic":
            self._parts = _AnthropicParts()
        else:
            self._parts = _OpenAIParts()
        self._done = done
        self._finished = False
        self._it: Any = None
        self._start = time.perf_counter() if start is None else start
        self.timing = Timing()
        self.diagnosis: Any = None

    def _feed(self, chunk: Any) -> None:
        text, reasoning = self._parts.feed(chunk)
        if not (text or reasoning):
            return
        ms = (time.perf_counter() - self._start) * 1000
        t = self.timing
        if reasoning and t.first_reasoning_ms is None:
            t.first_reasoning_ms = ms
        if text:
            if t.ttft_ms is None:
                t.ttft_ms = ms
            t.last_chunk_ms = ms
            t.chunks += 1

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            self.diagnosis = self._done(self._parts.response(), self.timing)

    def __iter__(self) -> GuardedStream:
        return self

    def __next__(self) -> Any:
        if self._it is None:
            self._it = iter(self._stream)
        try:
            chunk = next(self._it)
        except StopIteration:
            self._finish()
            raise
        self._feed(chunk)
        return chunk

    def __aiter__(self) -> GuardedStream:
        return self

    async def __anext__(self) -> Any:
        if self._it is None:
            self._it = aiter(self._stream)
        try:
            chunk = await anext(self._it)
        except StopAsyncIteration:
            self._finish()
            raise
        self._feed(chunk)
        return chunk

    def __enter__(self) -> GuardedStream:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._stream.__exit__(*exc) if hasattr(self._stream, "__exit__") else None

    async def __aenter__(self) -> GuardedStream:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*exc)
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
