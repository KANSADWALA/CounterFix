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
from typing import Any, Literal

from counterfix.adapters._attach import attach
from counterfix.core.config import Config
from counterfix.core.detector import Detector, run_detectors
from counterfix.core.diagnosis import Diagnosis, Finding, RetryAttempt
from counterfix.core.record import CallRecord, Message, Usage
from counterfix.core.types import PrescriptionKind

logger = logging.getLogger("counterfix")

_CREATE_PATH = ("chat", "completions", "create")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an object attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK objects to plain dicts."""
    if isinstance(obj, dict):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            result = dump()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    try:
        return dict(vars(obj))
    except TypeError:
        return {"value": repr(obj)}


def _usage(raw: Any) -> Usage:
    """Token usage, including reasoning tokens when the provider reports them."""
    usage = Usage.model_validate(_to_dict(raw))
    details = _get(raw, "completion_tokens_details")
    reasoning_tokens = _get(details, "reasoning_tokens")
    if isinstance(reasoning_tokens, int):
        usage.reasoning_tokens = reasoning_tokens
    return usage


def record_from_openai(
    request: dict[str, Any],
    response: Any,
    *,
    latency_ms: float | None = None,
    provider: str | None = None,
) -> CallRecord:
    """Build a :class:`CallRecord` from chat-completions kwargs and the response."""
    messages = [Message.model_validate(_to_dict(m)) for m in request.get("messages") or []]
    params = {k: v for k, v in request.items() if k not in ("messages", "model")}

    choices = _get(response, "choices") or []
    first = choices[0] if len(choices) > 0 else None
    message = _get(first, "message")
    raw_tool_calls = _get(message, "tool_calls")
    usage_raw = _get(response, "usage")
    usage = _usage(usage_raw) if usage_raw is not None else None
    # Reasoning models: Groq/OpenAI-style "reasoning", vLLM/DeepSeek-style "reasoning_content".
    reasoning = _get(message, "reasoning") or _get(message, "reasoning_content")

    return CallRecord(
        provider=provider,
        model=_get(response, "model") or request.get("model"),
        messages=messages,
        params=params,
        response_text=_get(message, "content"),
        reasoning_text=reasoning if isinstance(reasoning, str) else None,
        finish_reason=_get(first, "finish_reason"),
        tool_calls=[_to_dict(t) for t in raw_tool_calls] if raw_tool_calls else None,
        usage=usage,
        latency_ms=latency_ms,
        metadata={"n_choices": len(choices)},
    )


class _Guard:
    """Shared state for one guarded client."""

    def __init__(
        self,
        config: Config,
        detectors: Sequence[Detector],
        provider: str | None,
        *,
        to_record: Callable[..., CallRecord] | None = None,
        create_path: tuple[str, ...] = _CREATE_PATH,
        retry_params: frozenset[str] | None = None,
    ):
        self.config = config
        self.detectors = list(detectors)
        self.provider = provider
        self.client: Any = None  # the raw client, kept so diagnose() can fall back to it
        self.to_record = to_record or record_from_openai
        self.create_path = create_path
        self.retry_params = retry_params  # request fields retries may change; None = any

    def diagnose(
        self, request: dict[str, Any], response: Any, latency_ms: float
    ) -> Diagnosis | None:
        """Build the record, run detectors, attach the result. Never raises if fail_open."""
        try:
            record = self.to_record(
                request, response, latency_ms=latency_ms, provider=self.provider
            )
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
        tried = {(a.failure_type, repr(sorted(a.changes.items()))) for a in diagnosis.retries}
        for finding in sorted(diagnosis.model_failures, key=lambda f: -f.confidence):
            candidates: list[tuple[dict[str, Any], str]] = []
            rule = store.find(finding.failure_type, model if isinstance(model, str) else None)
            if rule is not None and rule.source != "builtin":
                candidates.append((rule.changes(params), f"rule:{rule.description}"))
            fix = finding.prescription
            if fix is not None and fix.kind is PrescriptionKind.PARAM_CHANGE and fix.patch:
                changed = {k: v for k, v in fix.patch.items() if params.get(k) != v}
                candidates.append((changed, "prescription"))
            if rule is not None and rule.source == "builtin":
                candidates.append((rule.changes(params), f"rule:{rule.description}"))
            for changes, source in candidates:
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

        anthropic = self.create_path == ("messages", "create")

        def done(response: Any) -> Diagnosis | None:
            return self.diagnose(kwargs, response, (time.perf_counter() - start) * 1000)

        if inspect.isawaitable(result):

            async def opened() -> Any:
                return GuardedStream(await result, anthropic, done)

            return opened()
        return GuardedStream(result, anthropic, done)

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
    create call (``chat.completions.create``, or ``messages.create`` for Anthropic) also
    produces a diagnosis."""

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
        if path == guard.create_path and callable(value):
            return guard.wrap_create(value)
        if path == guard.create_path[: len(path)]:
            return GuardedClient(value, guard, path)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_cf_target"), name, value)

    def __repr__(self) -> str:
        return f"counterfix.guard({object.__getattribute__(self, '_cf_target')!r})"

    @property
    def unwrapped(self) -> Any:
        """The original client."""
        return object.__getattribute__(self, "_cf_target")


def guard(
    client: Any,
    *,
    config: Config | None = None,
    detectors: Sequence[Detector] | None = None,
    provider: str | None = None,
) -> Any:
    """Wrap an LLM client so every call is checked: an OpenAI-compatible client
    (``chat.completions.create``) or an Anthropic client (``messages.create``), sync or async.

    >>> from openai import OpenAI          # doctest: +SKIP
    >>> client = guard(OpenAI())           # doctest: +SKIP
    >>> r = client.chat.completions.create(model="gpt-4o-mini", messages=[...])  # doctest: +SKIP
    >>> r.diagnosis.summary()              # doctest: +SKIP

    ``detectors`` defaults to the built-in live detectors. ``provider`` is an optional
    label stored on each record (e.g. "groq", "ollama").
    """
    if isinstance(client, GuardedClient):
        return client
    if detectors is None:
        from counterfix.detectors import default_detectors

        detectors = default_detectors()
    if not hasattr(client, "chat") and hasattr(client, "messages"):
        from counterfix.adapters.anthropic import ANTHROPIC_RETRY_PARAMS, record_from_anthropic

        state = _Guard(
            config or Config(),
            detectors,
            provider or "anthropic",
            to_record=record_from_anthropic,
            create_path=("messages", "create"),
            retry_params=ANTHROPIC_RETRY_PARAMS,
        )
    else:
        state = _Guard(config or Config(), detectors, provider)
    state.client = client
    return GuardedClient(client, state)
