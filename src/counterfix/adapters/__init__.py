"""Adapters that plug counterfix into LLM clients and frameworks."""

from counterfix.adapters._attach import get_diagnosis
from counterfix.adapters.openai_compat import GuardedClient, guard, record_from_openai

__all__ = ["GuardedClient", "get_diagnosis", "guard", "record_from_openai"]
