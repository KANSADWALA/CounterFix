"""Adapters that plug counterfix into LLM clients and frameworks."""

from counterfix.core.attach import get_diagnosis
from counterfix.core.convert import record_from_openai
from counterfix.integrations.callbacks import langchain_callback, litellm_callback
from counterfix.integrations.decorator import watch
from counterfix.integrations.openai_compat import GuardedClient, guard
from counterfix.integrations.responses import record_from_responses

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
