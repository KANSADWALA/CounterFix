from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import counterfix
from conftest import make_response
from counterfix import Config, FailureType


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> None:
    """Install a stand-in module (and its parents) so no framework is needed in CI."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        monkeypatch.setitem(sys.modules, ".".join(parts[:i]), ModuleType(".".join(parts[:i])))
    for key, value in attrs.items():
        setattr(sys.modules[name], key, value)


class Base:
    pass


def test_litellm_callback_checks_success_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(monkeypatch, "litellm.integrations.custom_logger", CustomLogger=Base)
    seen: list[Any] = []
    handler = counterfix.litellm_callback(config=Config(on_diagnosis=lambda r, d: seen.append(r)))
    assert isinstance(handler, Base)
    kwargs = {
        "model": "groq/llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Write an essay."}],
        "optional_params": {"max_tokens": 16},
    }
    start = datetime(2026, 10, 19)
    handler.log_success_event(
        kwargs, make_response("Essays", finish_reason="length"), start, start + timedelta(seconds=2)
    )
    assert handler.last_diagnosis.top.failure_type is FailureType.TRUNCATION
    assert seen[0].provider == "litellm" and seen[0].max_tokens == 16
    assert seen[0].latency_ms == 2000
    asyncio.run(handler.async_log_success_event(kwargs, make_response("Hi."), start, start))
    assert handler.last_diagnosis.ok
    handler.log_success_event(kwargs, object(), start, start)  # unreadable: ignored, no raise


def _lc_result(text: str, finish: str) -> Any:
    message = SimpleNamespace(
        content=text,
        response_metadata={"finish_reason": finish, "model_name": "gpt-4o-mini"},
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        tool_calls=[],
    )
    generation = SimpleNamespace(message=message, generation_info=None, text=text)
    return SimpleNamespace(generations=[[generation]], llm_output=None)


def test_langchain_callback_pairs_start_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(monkeypatch, "langchain_core.callbacks", BaseCallbackHandler=Base)
    seen: list[Any] = []
    handler = counterfix.langchain_callback(config=Config(on_diagnosis=lambda r, d: seen.append(r)))
    prompt = [
        SimpleNamespace(type="system", content="Be brief."),
        SimpleNamespace(type="human", content="Essay?"),
    ]
    handler.on_chat_model_start({}, [prompt], run_id=1, invocation_params={"max_tokens": 4})
    handler.on_llm_end(_lc_result("Essays are", "length"), run_id=1)
    assert handler.last_diagnosis.top.failure_type is FailureType.TRUNCATION
    record = seen[0]
    assert record.model == "gpt-4o-mini" and record.system_text == "Be brief."
    assert record.prompt_text == "Essay?" and record.max_tokens == 4
    assert record.usage.total_tokens == 7
    handler.on_llm_start({}, ["Say hi"], run_id=2)
    handler.on_llm_error(RuntimeError("x"), run_id=2)
    assert handler._runs == {}


def test_langchain_callback_with_a_real_chat_model() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    handler = counterfix.langchain_callback()
    llm = FakeListChatModel(responses=["The sky is usually blue on a clear day."])
    reply = llm.invoke("Answer in exactly one word: sky colour?", config={"callbacks": [handler]})
    assert reply.content.startswith("The sky")
    assert handler.last_diagnosis.top.failure_type is FailureType.DROPPED_INSTRUCTION
