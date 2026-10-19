"""Small helpers so counterfix's own extra calls (re-runs, judges) work with any client.

counterfix makes its extra calls synchronously. When the user's client is async
(``AsyncOpenAI``, ``AsyncAnthropic``), :func:`sync_client` builds the matching sync client
with the same key, URL, headers and timeouts; if that is not possible, :func:`resolve` runs
the coroutine to completion on a private event loop, so a call is never left un-awaited.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import threading
from collections.abc import Mapping
from typing import Any

# Request fields that only shape how a reply is delivered, never what the model says.
_DELIVERY_PARAMS = frozenset({"stream", "stream_options"})
_SDKS = frozenset({"openai", "anthropic"})  # packages whose Async* clients have a sync twin
_SETTINGS = ("api_key", "auth_token", "organization", "project", "base_url", "timeout",
             "max_retries")  # fmt: skip


def is_sentinel(value: Any) -> bool:
    """SDK placeholders meaning "not given" (openai/anthropic ``NOT_GIVEN``, ``Omit``)."""
    return type(value).__name__ in ("NotGiven", "Omit", "_NotGiven")


def clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Request parameters without "not given" placeholders."""
    return {k: v for k, v in params.items() if not is_sentinel(v)}


def rerun_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Parameters for a re-run: a complete reply, never a stream."""
    return {k: v for k, v in clean_params(params).items() if k not in _DELIVERY_PARAMS}


def sync_client(client: Any) -> Any:
    """The client itself if it is synchronous, else a sync twin with the same settings
    (``AsyncOpenAI`` -> ``OpenAI``), or the async client when no twin can be built."""
    cls = type(client)
    top = cls.__module__.split(".")[0]
    if not cls.__name__.startswith("Async") or top not in _SDKS:
        return client  # sync already, or an unknown async client: resolve() awaits it
    try:
        package = importlib.import_module(top)
        twin_cls = getattr(package, cls.__name__[len("Async") :], None)
        if twin_cls is None:
            return client
        accepted = inspect.signature(twin_cls).parameters
        kwargs = {
            name: getattr(client, name)
            for name in _SETTINGS
            if name in accepted and getattr(client, name, None) is not None
        }
        headers = getattr(client, "_custom_headers", None)
        if "default_headers" in accepted and headers:
            kwargs["default_headers"] = dict(headers)
        return twin_cls(**kwargs)
    except Exception:
        return client


def resolve(value: Any) -> Any:
    """``value``, or its result when it is awaitable (run on a private event loop)."""
    if not inspect.isawaitable(value):
        return value

    async def wait() -> Any:
        return await value

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wait())
    box: dict[str, Any] = {}

    def run() -> None:  # inside a running loop: finish on a separate thread
        try:
            box["value"] = asyncio.run(wait())
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, name="counterfix-await")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
