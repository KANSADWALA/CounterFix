"""``guard()`` for OpenAI-compatible clients.

Works with the official ``openai`` SDK and with any provider that exposes the same
``client.chat.completions.create(...)`` shape: OpenAI, Groq, Ollama, vLLM, Together,
OpenRouter and others. The wrapped client keeps the same API; each response gains a
``.diagnosis`` attribute (or use :func:`counterfix.get_diagnosis`).
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from counterfix.core.attach import attach
from counterfix.core.config import Config
from counterfix.core.convert import record_from_openai as record_from_openai
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis, Finding, RetryAttempt
from counterfix.core.record import CallRecord, Timing
from counterfix.core.types import PrescriptionKind

logger = logging.getLogger("counterfix")
ClientT = TypeVar("ClientT")

_CREATE_PATH = ("chat", "completions", "create")
# Every call that returns a chat completion: create, and structured outputs (parse).
_OPENAI_PATHS = (
    _CREATE_PATH,
    ("chat", "completions", "parse"),
    ("beta", "chat", "completions", "parse"),
)
_ANTHROPIC_PATHS = (("messages", "create"),)
# Client methods that return a new client of the same kind: the copy stays guarded.
_CLIENT_COPIES = frozenset({"with_options", "copy"})


class _Guard:
    """Shared state for one guarded client."""

    def __init__(
        self,
        config: Config,
        detectors: Sequence[Detector],
        provider: str | None,
        *,
        to_record: Callable[..., CallRecord] | None = None,
        create_paths: tuple[tuple[str, ...], ...] = _OPENAI_PATHS,
        retry_params: frozenset[str] | None = None,
        stream_kind: str | None = None,
        translate: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.config = config
        self.detectors = list(detectors)
        self.provider = provider
        self.client: Any = None  # the raw client, kept so diagnose() can fall back to it
        self.to_record = to_record or record_from_openai
        self.create_paths = create_paths
        self.anthropic = create_paths == _ANTHROPIC_PATHS
        self.stream_kind = stream_kind or ("anthropic" if self.anthropic else "chat")
        self.retry_params = retry_params  # request fields retries may change; None = any
        # Turns a parameter change written in chat terms into this API's terms.
        self.translate = translate
        # Guards for other calls on the same client (the Responses API next to chat).
        self.siblings: list[_Guard] = []

    def sibling(self, **kwargs: Any) -> _Guard:
        """A guard for another API on the same client, sharing settings and client."""
        other = _Guard(self.config, self.detectors, self.provider, **kwargs)
        other.client = self.client
        self.siblings.append(other)
        return other

    def owner(self, path: tuple[str, ...]) -> _Guard | None:
        """The guard that checks calls at ``path``, if any."""
        return next((g for g in (self, *self.siblings) if path in g.create_paths), None)

    def watches(self, path: tuple[str, ...]) -> bool:
        """True when ``path`` leads towards a checked call."""
        return any(path == p[: len(path)] for g in (self, *self.siblings) for p in g.create_paths)

    def diagnose(
        self,
        request: dict[str, Any],
        response: Any,
        latency_ms: float,
        timing: Timing | None = None,
    ) -> Diagnosis | None:
        """Build the record, run detectors, attach the result. Never raises if fail_open."""
        try:
            record = self.to_record(
                request, response, latency_ms=latency_ms, provider=self.provider
            )
            record.timing = timing
        except Exception as exc:
            if not self.config.fail_open:
                raise
            logger.warning("counterfix could not read this call: %s", exc)
            return None
        return self.check(record, response)

    def check(self, record: CallRecord, response: Any) -> Diagnosis | None:
        """Run detectors on a record, attach the result, call hooks. Never raises if fail_open."""
        try:
            from counterfix.context import apply

            diagnosis: Diagnosis = run_detectors(apply(record), self.detectors, self.config)
            diagnosis._client = self.client
            attach(response, diagnosis)
            if self.config.on_diagnosis is not None:
                self.config.on_diagnosis(record, diagnosis)
            from counterfix import background

            if background.wants(diagnosis, self.config):
                background.submit(record, self.config, self.client)
            return diagnosis
        except Exception as exc:
            if not self.config.fail_open:
                raise
            logger.warning("counterfix could not diagnose this call: %s", exc)
            return None

    # --- retry mode --------------------------------------------------------------------

    def plan_retry(
        self, diagnosis: Diagnosis, request: dict[str, Any]
    ) -> tuple[Finding, dict[str, Any], str] | None:
        """Pick the most confident fixable finding and the parameter change to try."""
        store = self.config.rule_store()
        model = request.get("model")
        params = {k: v for k, v in request.items() if k != "messages"}
        if self.translate is not None:  # the request is in this API's terms: compare in chat
            params = self.to_record(request, None).params
        tried = {(a.failure_type, repr(sorted(a.changes.items()))) for a in diagnosis.retries}
        for finding in sorted(diagnosis.model_failures, key=lambda f: -f.confidence):
            candidates: list[tuple[dict[str, Any], str]] = []
            rule = store.find(finding.failure_type, model if isinstance(model, str) else None)
            if rule is not None and rule.source != "builtin":
                candidates.append((rule.changes(params), f"rule:{rule.description}"))
            fix = finding.prescription
            if fix is not None and fix.kind is PrescriptionKind.PARAM_CHANGE and fix.patch:
                changed = {k: v for k, v in fix.patch.items() if params.get(k) != v}
                for k in [k for k in changed if k in RETRY_LIMIT_KEYS]:
                    # one retry only: a 5 -> 20 raise rarely helps; a limit is a ceiling
                    if isinstance(changed[k], int):
                        changed[k] = max(changed[k], RETRY_MIN_LIMIT)
                candidates.append((changed, "prescription"))
            if rule is not None and rule.source == "builtin":
                candidates.append((rule.changes(params), f"rule:{rule.description}"))
            for changes, source in candidates:
                if self.translate is not None:
                    changes = self.translate(changes, request)
                if self.retry_params is not None:
                    changes = {k: v for k, v in changes.items() if k in self.retry_params}
                key = (finding.failure_type, repr(sorted(changes.items())))
                if changes and key not in tried:
                    return finding, changes, source
        return None

    @staticmethod
    def judge_retry(
        before: Diagnosis, after: Diagnosis | None, finding: Finding
    ) -> Literal["fixed", "improved", "no_better"]:
        if after is None:
            return "no_better"
        after_types = {f.failure_type for f in after.model_failures}
        if not after.model_failures:
            return "fixed"
        if finding.failure_type not in after_types and len(after.model_failures) <= len(
            before.model_failures
        ):
            return "improved"
        return "no_better"

    def _record_attempt(
        self,
        before: Diagnosis,
        after: Diagnosis | None,
        finding: Finding,
        changes: dict[str, Any],
        source: str,
        error: Exception | None = None,
    ) -> RetryAttempt:
        outcome = "error" if error is not None else self.judge_retry(before, after, finding)
        return RetryAttempt(
            failure_type=finding.failure_type,
            changes=changes,
            source=source,
            outcome=outcome,
            error=f"{type(error).__name__}: {error}" if error is not None else None,
            findings_after=[f.failure_type for f in after.findings] if after is not None else [],
        )

    def _adopt(self, before: Diagnosis, after: Diagnosis, attempt: RetryAttempt) -> Diagnosis:
        """The retried response is better: its diagnosis carries the history."""
        after.retries = [*before.retries, attempt]
        after.retried_from = before.retried_from or before.findings
        return after

    # --- the wrapped create ------------------------------------------------------------

    def wrap_create(self, create: Callable[..., Any]) -> Callable[..., Any]:
        guard = self

        def guarded_create(*args: Any, **kwargs: Any) -> Any:
            if not guard.config.enabled:
                return create(*args, **kwargs)
            start = time.perf_counter()
            if kwargs.get("stream"):
                return guard.wrap_stream(create(*args, **kwargs), kwargs, start)
            result = create(*args, **kwargs)  # the user's own errors propagate unchanged
            if inspect.isawaitable(result):
                return guard._finish_async(create, args, result, kwargs, start)
            diagnosis = guard.diagnose(kwargs, result, (time.perf_counter() - start) * 1000)
            return guard._retry_sync(create, args, kwargs, result, diagnosis)

        guarded_create.__doc__ = create.__doc__
        guarded_create.__wrapped__ = create  # type: ignore[attr-defined]
        return guarded_create

    def wrap_stream(self, result: Any, kwargs: dict[str, Any], start: float) -> Any:
        """Pass a stream through chunk by chunk; check the whole answer when it ends."""
        from counterfix.adapters.streams import GuardedStream

        kind = self.stream_kind

        def done(response: Any, timing: Timing) -> Diagnosis | None:
            return self.diagnose(kwargs, response, (time.perf_counter() - start) * 1000, timing)

        if inspect.isawaitable(result):

            async def opened() -> Any:
                return GuardedStream(await result, kind, done, start)

            return opened()
        return GuardedStream(result, kind, done, start)

    def _retry_sync(
        self,
        create: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        diagnosis: Diagnosis | None,
    ) -> Any:
        for _ in range(self.config.retry_budget):
            if diagnosis is None or not diagnosis.model_failures:
                break
            plan = self._safe_plan(diagnosis, kwargs)
            if plan is None:
                break
            finding, changes, source = plan
            new_kwargs = {**kwargs, **changes}
            start = time.perf_counter()
            try:
                new_result = create(*args, **new_kwargs)
            except Exception as exc:
                diagnosis.retries.append(
                    self._record_attempt(diagnosis, None, finding, changes, source, exc)
                )
                break
            new_diag = self.diagnose(new_kwargs, new_result, (time.perf_counter() - start) * 1000)
            attempt = self._record_attempt(diagnosis, new_diag, finding, changes, source)
            if attempt.outcome in ("fixed", "improved") and new_diag is not None:
                result, kwargs, diagnosis = (
                    new_result,
                    new_kwargs,
                    self._adopt(diagnosis, new_diag, attempt),
                )
            else:
                diagnosis.retries.append(attempt)
                attach(result, diagnosis)
                break
        return result

    async def _finish_async(
        self,
        create: Callable[..., Any],
        args: tuple[Any, ...],
        pending: Any,
        kwargs: dict[str, Any],
        start: float,
    ) -> Any:
        result = await pending
        diagnosis = self.diagnose(kwargs, result, (time.perf_counter() - start) * 1000)
        for _ in range(self.config.retry_budget):
            if diagnosis is None or not diagnosis.model_failures:
                break
            plan = self._safe_plan(diagnosis, kwargs)
            if plan is None:
                break
            finding, changes, source = plan
            new_kwargs = {**kwargs, **changes}
            begin = time.perf_counter()
            try:
                new_result = await create(*args, **new_kwargs)
            except Exception as exc:
                diagnosis.retries.append(
                    self._record_attempt(diagnosis, None, finding, changes, source, exc)
                )
                break
            new_diag = self.diagnose(new_kwargs, new_result, (time.perf_counter() - begin) * 1000)
            attempt = self._record_attempt(diagnosis, new_diag, finding, changes, source)
            if attempt.outcome in ("fixed", "improved") and new_diag is not None:
                result, kwargs, diagnosis = (
                    new_result,
                    new_kwargs,
                    self._adopt(diagnosis, new_diag, attempt),
                )
            else:
                diagnosis.retries.append(attempt)
                attach(result, diagnosis)
                break
        return result

    def _safe_plan(
        self, diagnosis: Diagnosis, kwargs: dict[str, Any]
    ) -> tuple[Finding, dict[str, Any], str] | None:
        try:
            return self.plan_retry(diagnosis, kwargs)
        except Exception as exc:
            if not self.config.fail_open:
                raise
            logger.warning("counterfix could not plan a retry: %s", exc)
            return None


class GuardedClient:
    """A transparent proxy: everything behaves like the wrapped client, except that its
    calls that return a chat completion (``chat.completions.create`` / ``.parse``, or
    ``messages.create`` for Anthropic) also produce a diagnosis.

    ``with guard(OpenAI()) as client:`` and ``async with`` work, and ``with_options()`` /
    ``copy()`` return a guarded client. ``isinstance(client, OpenAI)`` is False: use
    ``client.unwrapped`` where a library needs the real client object.
    """

    __slots__ = ("_cf_guard", "_cf_path", "_cf_target")

    def __init__(self, target: Any, guard: _Guard, path: tuple[str, ...] = ()):
        object.__setattr__(self, "_cf_target", target)
        object.__setattr__(self, "_cf_guard", guard)
        object.__setattr__(self, "_cf_path", path)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_cf_target")
        guard = object.__getattribute__(self, "_cf_guard")
        path = (*object.__getattribute__(self, "_cf_path"), name)
        value = getattr(target, name)
        owner = guard.owner(path)
        if owner is not None and callable(value):
            return owner.wrap_create(value)
        if guard.watches(path):
            return GuardedClient(value, guard, path)
        if len(path) == 1 and name in _CLIENT_COPIES and callable(value):

            def guarded_copy(*args: Any, **kwargs: Any) -> Any:
                return GuardedClient(value(*args, **kwargs), guard)

            return guarded_copy
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_cf_target"), name, value)

    def __repr__(self) -> str:
        return f"counterfix.guard({object.__getattribute__(self, '_cf_target')!r})"

    def __dir__(self) -> list[str]:
        return sorted({*dir(object.__getattribute__(self, "_cf_target")), "unwrapped"})

    # Python looks these up on the type, not through __getattr__, so they are spelled out.
    def __enter__(self) -> GuardedClient:
        object.__getattribute__(self, "_cf_target").__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return object.__getattribute__(self, "_cf_target").__exit__(*exc)

    async def __aenter__(self) -> GuardedClient:
        await object.__getattribute__(self, "_cf_target").__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await object.__getattribute__(self, "_cf_target").__aexit__(*exc)

    @property
    def unwrapped(self) -> Any:
        """The original client."""
        return object.__getattribute__(self, "_cf_target")


RETRY_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
RETRY_MIN_LIMIT = 1024


def settings(
    config: Config | None = None,
    *,
    mode: Literal["flag", "retry"] | None = None,
    log: str | Path | None = None,
    judge: Any = None,
    sample_rate: float | None = None,
) -> Config:
    """The :class:`Config` for ``guard`` and ``watch``: ``config`` (or the defaults) with the
    keyword shortcuts applied, so common setups need no Config object.

    ``mode="retry"`` re-runs a flagged call once with a known fix; ``log`` appends every call
    to a JSONL file (after any ``on_diagnosis`` hook already set); ``judge`` picks the judge
    for background checks and ``sample_rate`` the share of unflagged calls it checks.
    """
    base = config if config is not None else Config()
    update: dict[str, Any] = {}
    if mode is not None:
        update["mode"] = mode
    if log is not None:
        from counterfix.core.config import hooks
        from counterfix.report import log_to

        update["on_diagnosis"] = hooks(base.on_diagnosis, log_to(log))
    if judge is not None:
        update["judge"] = judge
    if sample_rate is not None:
        update["sample_rate"] = sample_rate
    if not update:
        return base
    return Config.model_validate({**dict(base), **update})


def guard(
    client: ClientT,
    *,
    config: Config | None = None,
    detectors: Sequence[Detector] | None = None,
    provider: str | None = None,
    mode: Literal["flag", "retry"] | None = None,
    log: str | Path | None = None,
    judge: Any = None,
    sample_rate: float | None = None,
) -> ClientT:
    """Wrap an LLM client so every call is checked: an OpenAI-compatible client
    (``chat.completions.create``) or an Anthropic client (``messages.create``), sync or async.

    >>> from openai import OpenAI          # doctest: +SKIP
    >>> client = guard(OpenAI())           # doctest: +SKIP
    >>> r = client.chat.completions.create(model="gpt-4o-mini", messages=[...])  # doctest: +SKIP
    >>> r.diagnosis.summary()              # doctest: +SKIP

    Common settings need no :class:`Config`: ``guard(client, mode="retry")`` re-runs a
    flagged call once with a known fix, ``log="calls.jsonl"`` records every call, and
    ``judge=...`` with ``sample_rate=0.05`` adds background judge checks. A ``config`` and
    these shortcuts can be combined; the shortcuts win.

    ``detectors`` defaults to the built-in live detectors. ``provider`` labels each record
    (e.g. "groq"); when not given it is read from the client's base URL where known, so
    ``fix()`` and exported regression tests can re-run the call later. The result is typed
    as the client you passed, so editors keep their completions.
    """
    shortcuts = {"mode": mode, "log": log, "judge": judge, "sample_rate": sample_rate}
    if isinstance(client, GuardedClient):
        unchanged = config is None and detectors is None and provider is None
        if unchanged and all(v is None for v in shortcuts.values()):
            return client
        client = client.unwrapped  # re-wrap with the new settings
    if isinstance(client, type):
        raise TypeError(
            f"guard() needs a client instance: guard({client.__name__}()), "
            f"not guard({client.__name__})"
        )
    if not any(hasattr(client, a) for a in ("chat", "messages", "responses")):
        raise TypeError(
            f"guard() wraps an OpenAI-compatible or Anthropic client; {type(client).__name__} "
            "has no chat, responses or messages API. For your own function use "
            "@counterfix.watch; for LiteLLM or LangChain use counterfix.litellm_callback() or "
            "counterfix.langchain_callback()."
        )
    config = settings(config, **shortcuts)
    provider = provider or provider_from_url(getattr(client, "base_url", None))
    if detectors is None:
        from counterfix.detect import default_detectors

        detectors = default_detectors()
    if not hasattr(client, "chat") and hasattr(client, "messages"):
        from counterfix.adapters.anthropic import ANTHROPIC_RETRY_PARAMS, record_from_anthropic

        state = _Guard(
            config,
            detectors,
            provider or "anthropic",
            to_record=record_from_anthropic,
            create_paths=_ANTHROPIC_PATHS,
            retry_params=ANTHROPIC_RETRY_PARAMS,
        )
        state.client = client
    else:
        from counterfix.adapters.responses import (
            RESPONSES_PATHS,
            RESPONSES_RETRY_PARAMS,
            record_from_responses,
            responses_changes,
        )

        state = _Guard(config, detectors, provider)
        state.client = client
        if hasattr(client, "responses"):
            state.sibling(
                to_record=record_from_responses,
                create_paths=RESPONSES_PATHS,
                retry_params=RESPONSES_RETRY_PARAMS,
                stream_kind="responses",
                translate=responses_changes,
            )
    return cast(ClientT, GuardedClient(client, state))


_HOSTS = {
    "api.openai.com": "openai",
    "api.groq.com": "groq",
    "generativelanguage.googleapis.com": "gemini",
    "openrouter.ai": "openrouter",
    "localhost:11434": "ollama",
    "127.0.0.1:11434": "ollama",
}


def provider_from_url(base_url: Any) -> str | None:
    """'https://api.groq.com/openai/v1' -> 'groq'; None for hosts we have no preset for."""
    text = str(base_url or "")
    return next((name for host, name in _HOSTS.items() if host in text), None)
