"""Adapters that plug counterfix into LLM clients and frameworks."""

from counterfix.adapters.callbacks import langchain_callback, litellm_callback
from counterfix.adapters.decorator import watch
from counterfix.adapters.openai_compat import GuardedClient, guard
from counterfix.adapters.responses import record_from_responses
from counterfix.core.attach import get_diagnosis
from counterfix.core.convert import record_from_openai

__all__ = [
    "GuardedClient",
    "get_diagnosis",
    "guard",
    "langchain_callback",
    "litellm_callback",
    "record_from_openai",
    "record_from_responses",
    "watch",
]
