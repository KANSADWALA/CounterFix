"""Checking streamed calls without delaying a single token.

The wrapped stream yields every chunk the moment it arrives and keeps a copy. When the
stream ends, the copy is assembled into one response and checked as usual; the result
is on ``stream.diagnosis`` (and goes to ``Config.on_diagnosis``). Streams are never
retried: the tokens have already reached the user. Works for OpenAI-compatible chunks
and Anthropic Messages events, sync and async.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from counterfix.adapters.openai_compat import _get


class _OpenAIParts:
    def __init__(self) -> None:
        self.model: Any = None
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.finish: Any = None
        self.usage: Any = None
        self.tools: dict[int, dict[str, Any]] = {}

    def feed(self, chunk: Any) -> None:
        self.model = _get(chunk, "model") or self.model
        self.usage = _get(chunk, "usage") or self.usage
        for choice in (_get(chunk, "choices") or [])[:1]:
            delta = _get(choice, "delta")
            self.finish = _get(choice, "finish_reason") or self.finish
            if (text := _get(delta, "content")) is not None:
                self.text.append(text)
            if r := _get(delta, "reasoning") or _get(delta, "reasoning_content"):
                self.reasoning.append(r)
            for call in _get(delta, "tool_calls") or []:
                slot = self.tools.setdefault(_get(call, "index", 0), {"name": "", "args": ""})
                slot["id"] = _get(call, "id") or slot.get("id")
                fn = _get(call, "function")
                slot["name"] += _get(fn, "name") or ""
                slot["args"] += _get(fn, "arguments") or ""

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
        choice = SimpleNamespace(index=0, message=message, finish_reason=self.finish)
        return SimpleNamespace(model=self.model, choices=[choice], usage=self.usage)


class _AnthropicParts:
    def __init__(self) -> None:
        self.model: Any = None
        self.blocks: list[dict[str, Any]] = []
        self.stop: Any = None
        self.input_tokens: Any = None
        self.output_tokens: Any = None

    def feed(self, event: Any) -> None:
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
            block["text"] += _get(delta, "text") or ""
            block["thinking"] += _get(delta, "thinking") or ""
            block["json"] += _get(delta, "partial_json") or ""
        elif kind == "message_delta":
            self.stop = _get(_get(event, "delta"), "stop_reason") or self.stop
            self.output_tokens = _get(_get(event, "usage"), "output_tokens") or self.output_tokens

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

    def __init__(self, stream: Any, anthropic: bool, done: Callable[[Any], Any]):
        self._stream = stream
        self._parts: Any = _AnthropicParts() if anthropic else _OpenAIParts()
        self._done = done
        self._finished = False
        self._it: Any = None
        self.diagnosis: Any = None

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            self.diagnosis = self._done(self._parts.response())

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
        self._parts.feed(chunk)
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
        self._parts.feed(chunk)
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
