"""Callbacks for frameworks that already make the call: LiteLLM and LangChain.

>>> litellm.callbacks = [counterfix.litellm_callback()]                    # doctest: +SKIP
>>> llm.invoke("...", config={"callbacks": [counterfix.langchain_callback()]})  # doctest: +SKIP

Both only observe: they run the live checks after each successful call and send the
result to ``Config.on_diagnosis`` (and background judge checks when ``sample_rate`` is
set). They never change the response and never retry. The latest diagnosis is also on
``handler.last_diagnosis``. Each needs its framework installed; counterfix imports it
only when you create the handler.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from counterfix.adapters.openai_compat import _get, _Guard, _to_dict, record_from_openai
from counterfix.core.config import Config
from counterfix.core.detector import Detector
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord, Message, Usage


def _state(config: Config | None, detectors: Sequence[Detector] | None, provider: str) -> _Guard:
    if detectors is None:
        from counterfix.detectors import default_detectors

        detectors = default_detectors()
    return _Guard(config or Config(), detectors, provider)


def _ms(start: Any, end: Any) -> float | None:
    try:
        return float((end - start).total_seconds() * 1000)
    except Exception:
        return None


# --- LiteLLM ----------------------------------------------------------------------------


def litellm_callback(
    *, config: Config | None = None, detectors: Sequence[Detector] | None = None
) -> Any:
    """A LiteLLM ``CustomLogger``: ``litellm.callbacks = [counterfix.litellm_callback()]``."""
    from litellm.integrations.custom_logger import CustomLogger

    state = _state(config, detectors, "litellm")

    class CounterfixLogger(CustomLogger):  # type: ignore[misc]
        last_diagnosis: Diagnosis | None = None

        def _check(self, kwargs: dict[str, Any], response: Any, start: Any, end: Any) -> None:
            try:
                request = {
                    **(kwargs.get("optional_params") or {}),
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages") or [],
                }
                record = record_from_openai(
                    request, response, latency_ms=_ms(start, end), provider="litellm"
                )
            except Exception:
                if not state.config.fail_open:
                    raise
                return
            self.last_diagnosis = state.check(record, response)

        def log_success_event(self, kwargs: Any, response_obj: Any, start: Any, end: Any) -> None:
            self._check(kwargs, response_obj, start, end)

        async def async_log_success_event(
            self, kwargs: Any, response_obj: Any, start: Any, end: Any
        ) -> None:
            self._check(kwargs, response_obj, start, end)

    return CounterfixLogger()


# --- LangChain --------------------------------------------------------------------------


def _lc_message(m: Any) -> Message:
    role = {"human": "user", "ai": "assistant"}.get(_get(m, "type"), _get(m, "type") or "user")
    return Message(role=role, content=_get(m, "content"))


def record_from_langchain(
    inputs: list[Message],
    response: Any,
    *,
    params: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> CallRecord:
    """A CallRecord from the prompt messages and a LangChain ``LLMResult``."""
    generations = _get(response, "generations") or [[]]
    first = generations[0][0] if generations and generations[0] else None
    message = _get(first, "message")
    info = _to_dict(_get(first, "generation_info") or {})
    meta = _to_dict(_get(message, "response_metadata") or {})
    finish = info.get("finish_reason") or meta.get("finish_reason") or meta.get("stop_reason")
    raw_usage = _to_dict(_get(message, "usage_metadata") or {})
    usage = None
    if raw_usage:
        usage = Usage(
            prompt_tokens=raw_usage.get("input_tokens"),
            completion_tokens=raw_usage.get("output_tokens"),
            total_tokens=raw_usage.get("total_tokens"),
        )
    content = _get(message, "content") if message is not None else _get(first, "text")
    text = content if isinstance(content, str) else Message(role="ai", content=content).text
    tool_calls = [
        {
            "id": c.get("id"),
            "type": "function",
            "function": {"name": c.get("name"), "arguments": c.get("args")},
        }
        for c in (_get(message, "tool_calls") or [])
    ]
    llm_output = _to_dict(_get(response, "llm_output") or {})
    params = params or {}
    model = params.get("model") or params.get("model_name")
    params = {k: v for k, v in params.items() if k not in ("model", "model_name", "_type")}
    return CallRecord(
        provider="langchain",
        model=meta.get("model_name") or meta.get("model") or llm_output.get("model_name") or model,
        messages=inputs,
        params=params,
        response_text=text,
        finish_reason=finish,
        tool_calls=tool_calls or None,
        usage=usage,
        latency_ms=latency_ms,
    )


def langchain_callback(
    *, config: Config | None = None, detectors: Sequence[Detector] | None = None
) -> Any:
    """A LangChain callback handler: pass it in ``config={"callbacks": [...]}``."""
    import time

    from langchain_core.callbacks import BaseCallbackHandler

    state = _state(config, detectors, "langchain")

    class CounterfixHandler(BaseCallbackHandler):  # type: ignore[misc]
        last_diagnosis: Diagnosis | None = None

        def __init__(self) -> None:
            self._runs: dict[Any, tuple[list[Message], dict[str, Any], float]] = {}

        def _start(self, run_id: Any, inputs: list[Message], kwargs: Any) -> None:
            params = _to_dict(kwargs.get("invocation_params") or {})
            self._runs[run_id] = (inputs, params, time.perf_counter())

        def on_chat_model_start(
            self, serialized: Any, messages: list[list[Any]], *, run_id: Any, **kwargs: Any
        ) -> None:
            self._start(run_id, [_lc_message(m) for m in messages[0]], kwargs)

        def on_llm_start(
            self, serialized: Any, prompts: list[str], *, run_id: Any, **kwargs: Any
        ) -> None:
            self._start(run_id, [Message(role="user", content=prompts[0])], kwargs)

        def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
            inputs, params, start = self._runs.pop(run_id, ([], {}, time.perf_counter()))
            try:
                ms = (time.perf_counter() - start) * 1000
                record = record_from_langchain(inputs, response, params=params, latency_ms=ms)
            except Exception:
                if not state.config.fail_open:
                    raise
                return
            self.last_diagnosis = state.check(record, response)

        def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
            self._runs.pop(run_id, None)  # provider errors stay the app's; nothing to check

    return CounterfixHandler()
