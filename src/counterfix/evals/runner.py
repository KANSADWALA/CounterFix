"""The eval runner: run a model over a dataset, score every answer, then say why the
misses failed and which change was verified to fix them.

>>> result = counterfix.evaluate("gsm8k_50.jsonl", model="groq:openai/gpt-oss-120b",
...                              metrics=["number"], judge=True, fix=5)   # doctest: +SKIP
>>> print(result.text())                                                   # doctest: +SKIP

Every item is checked with the free live checks. Items that miss a metric are also
diagnosed (with the judge when ``judge`` is set), so the report groups the misses by
cause. With ``fix=N``, up to N missed items get a counterfactual fix search: a plain
retry first, then one targeted change at a time, and a change counts only when the item
passes its metrics with no new failure clearly more often than a plain retry (the test
in :mod:`counterfix.fixes.verify`).
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from counterfix.core.caller import DailyLimitError, ModelCaller
from counterfix.core.diagnosis import Diagnosis, Evidence, Finding
from counterfix.core.record import CallRecord
from counterfix.core.types import FailureType
from counterfix.evals.asking import (
    Progress as Progress,
    _ask_all as _ask_all,
    _cached as _cached,
    _fingerprint as _fingerprint,
    _question as _question,
    _run_app as _run_app,
)
from counterfix.evals.data import EvalItem, is_suite, load_suite, make_items, read_rows
from counterfix.evals.fixing import (
    _metric_miss as _metric_miss,
    _passed as _passed,
    _with_metric_findings as _with_metric_findings,
    fix_item as fix_item,
    score as score,
)
from counterfix.evals.metrics import Metric, get_metric
from counterfix.evals.results import (
    EvalResult as EvalResult,
    FixTry as FixTry,
    ItemFix as ItemFix,
    ItemResult as ItemResult,
    MetricSummary as MetricSummary,
    _id_mismatch_note as _id_mismatch_note,
    bootstrap_ci as bootstrap_ci,
    summarize as summarize,
)
from counterfix.evals.sampling import (
    SAMPLE_KEY as SAMPLE_KEY,
    _check_code_execution as _check_code_execution,
    _pass_at_k as _pass_at_k,
    _repeat as _repeat,
)
from counterfix.runtime.perf import speed_summary


def _caller(model: str | ModelCaller) -> ModelCaller:
    if not isinstance(model, str):
        return model
    from counterfix.core.caller import RateLimitedCaller
    from counterfix.core.presets import caller_for, pace_for

    return RateLimitedCaller(caller_for(model, timeout=120), pace_for(model))


def evaluate(
    data: str | Path | Sequence[Mapping[str, Any]] | Sequence[EvalItem],
    *,
    model: str | ModelCaller | None = None,
    metrics: Sequence[str | Metric] | None = None,
    judge: Any = False,
    fix: int = 0,
    max_runs: int | None = None,
    input: str | None = None,
    reference: str | None = None,
    template: str | None = None,
    system: str | None = None,
    params: Mapping[str, Any] | None = None,
    limit: int | None = None,
    out: str | Path | None = None,
    name: str | None = None,
    progress: Progress | None = None,
    app: Callable[[str], Any] | str | None = None,
    concurrency: int | None = None,
    samples: int | None = None,
) -> EvalResult:
    """Run (or score) a dataset and explain the misses. See the module docstring.

    ``data`` is a suite file, a dataset file, rows or :class:`EvalItem` objects.
    ``model`` is a ``provider:model`` spec or a ModelCaller; leave it out to score
    answers already in the data (``response`` field). ``metrics`` are names such as
    ``"exact_match"`` or ``"latency:2000"`` (default: exact_match). ``out`` keeps each
    answer as it arrives, so an interrupted run resumes where it stopped.

    ``app`` evaluates your own application instead of a bare model: a function (or
    ``"module:function"``) that takes the question and returns the answer: a string, a
    dict such as LangChain's ``{"answer": ..., "context": [...]}``, a LlamaIndex
    ``Response``, a guarded client's response or a ``CallRecord`` (see
    :func:`counterfix.detect.retrieval.record_from_output`). The
    passages it returns are what the RAG metrics and checks look at. The fix search needs
    ``model`` (it re-runs the model with changed settings), so it is off with ``app``.

    ``concurrency`` runs that many items at once (threads; answers keep their order).
    Free tiers are still paced by the rate-limited caller.

    ``samples=n`` asks each question n times and reports pass@k (the unbiased estimator)
    for every pass/fail metric; set a temperature above 0 in ``params`` for distinct
    samples (HumanEval used 0.8 for pass@10 and pass@100). Code metrics (``tests``) run
    the model's code, which must be turned on first (``COUNTERFIX_ALLOW_CODE=1``).
    """
    suite: dict[str, Any] = {}
    if isinstance(data, (str, Path)) and is_suite(data):
        suite = load_suite(data)
        data = suite["data"]

    def pick(key: str, value: Any) -> Any:
        return value if value is not None else suite.get(key)

    model = pick("model", model)
    metric_specs = list(pick("metrics", metrics) or ["exact_match"])
    if judge is False and suite.get("judge") is not None:
        judge = suite["judge"]
    fix = fix or int(suite.get("fix") or 0)
    max_runs = int(pick("max_runs", max_runs) or 30)
    if isinstance(data, (str, Path)):
        rows: Sequence[Any] = read_rows(data)
        name = name or suite.get("name") or Path(data).stem
    else:
        rows = data
    if rows and isinstance(rows[0], EvalItem):
        items = list(rows)[: pick("limit", limit)]
    else:
        items = make_items(
            rows,
            input=pick("input", input),
            reference=pick("reference", reference),
            template=pick("template", template),
            system=pick("system", system),
            params=pick("params", params),
            limit=pick("limit", limit),
        )
    chosen = [get_metric(m) for m in metric_specs]
    _check_code_execution(chosen)
    n_samples = max(1, int(pick("samples", samples) or 1))
    if n_samples > 1:
        items = _repeat(items, n_samples)
    app_spec = pick("app", app)
    if isinstance(app_spec, str):
        from counterfix.evals.data import load_callable

        app_spec = load_callable(app_spec)
    run: Callable[[str], Any] | None = app_spec
    concurrency = max(1, int(pick("concurrency", concurrency) or 1))
    if run is not None and model is not None:
        raise ValueError("give either model= or app=, not both")
    caller = _caller(model) if model is not None else None
    if caller is None and app is None and any(i.response is None for i in items):
        raise ValueError("no model given and some items have no recorded 'response' to score")
    app_name = (getattr(run, "__name__", None) or type(run).__name__) if run else None
    result = EvalResult(
        name=name or suite.get("name") or "dataset",
        model=caller.name if caller else (f"app:{app_name}" if run else None),
        samples=n_samples,
    )
    out_path = Path(out) if out else None
    cache = _cached(out_path, result.model)
    judging = judge not in (False, None)
    if judging:  # choose once, before any call: a missing key fails fast, not mid-run
        from counterfix.judges.auto import resolve_judge

        if isinstance(model, str):
            from counterfix.core.presets import load_env_file

            load_env_file(Path.cwd() / ".env")
        target = model.split(":", 1)[-1] if isinstance(model, str) else None
        wanted = None if judge in (True, "auto") else judge
        judge = resolve_judge(wanted, target_model=target)[0]

    records: list[tuple[EvalItem, CallRecord]] = []
    live = caller is not None or run is not None
    write_lock = threading.Lock()

    def ask(item: EvalItem) -> CallRecord:
        """Answer one item (a kept answer, the recorded one, the model's or the app's)."""
        asked = _fingerprint(item)
        if (item.id, asked) in cache:
            kept = cache[(item.id, asked)]
            kept.metadata = {**item.metadata, **kept.metadata}
            return kept
        if caller is not None:
            try:
                record = caller.complete(item.messages, **item.params)
            except DailyLimitError:
                raise
            except Exception as exc:
                record = CallRecord(
                    messages=item.messages, params=item.params, model=caller.name,
                    error=f"{type(exc).__name__}: {exc}",
                )  # fmt: skip
        elif run is not None:
            record = _run_app(run, item)
        else:
            record = CallRecord(
                messages=item.messages, response_text=item.response, params=item.params
            )
        record.metadata = {**item.metadata, **record.metadata}
        if out_path is not None and record.error is None and live:
            line = {
                "id": item.id,
                "model": result.model,
                "asked": asked,
                "record": record.model_dump(mode="json"),
            }
            with write_lock, out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return record

    if concurrency > 1 and live and len(items) > 1:
        records, stopped = _ask_all(items, ask, concurrency, progress)
        result.stopped = stopped or ""
    else:
        errors_in_a_row = 0
        for n, item in enumerate(items, 1):
            if progress:
                progress(n, len(items), item.id)
            try:
                record = ask(item)
            except DailyLimitError as exc:
                result.stopped = f"stopped after {len(records)} item(s): {exc}"
                break
            errors_in_a_row = errors_in_a_row + 1 if record.error else 0
            records.append((item, record))
            if errors_in_a_row >= 3 and len(records) == errors_in_a_row:
                result.stopped = (
                    f"stopped: the first {errors_in_a_row} calls failed ({record.error})"
                )
                break

    from counterfix.diagnose import diagnose

    judge_name: str | None = None
    diagnoses: list[Diagnosis] = []
    for item, record in records:
        results = score(record, item.reference, chosen)
        passed = _passed(record, results)
        if record.error is not None:  # the call failed: not the model's answer at all
            diagnosis = Diagnosis(
                record_id=record.id,
                findings=[
                    Finding(
                        failure_type=FailureType.INFRASTRUCTURE,
                        confidence=1.0,
                        evidence=[Evidence(detector="call", detail=record.error[:300])],
                    )
                ],
            )
        else:
            diagnosis = diagnose(record, judge=judge if (judging and not passed) else False)
        if judging and not passed and record.error is None:
            judge_name = judge_name or getattr(judge, "name", None) or diagnosis.judge
            if diagnosis.errors:
                first = next(iter(diagnosis.errors.values()))
                result.judge_errors.append(f"{item.id}: {first[:160]}")
        diagnosis = _with_metric_findings(diagnosis, results)
        causes = [
            f.failure_type.value for f in sorted(diagnosis.findings, key=lambda f: -f.confidence)
        ]
        model_causes = [
            f.failure_type.value
            for f in sorted(diagnosis.model_failures, key=lambda f: -f.confidence)
        ]
        entry = ItemResult(
            id=item.id, record=record, reference=item.reference, metrics=results,
            passed=passed, flagged=passed and bool(model_causes),
            causes=model_causes if passed else causes,
        )  # fmt: skip
        result.items.append(entry)
        diagnoses.append(diagnosis)

    # Fix search: missed items first, then flagged ones, up to `fix` items in all.
    if caller is not None and fix > 0:
        queue = [k for k, i in enumerate(result.items) if not i.passed and i.record.error is None]
        queue += [k for k, i in enumerate(result.items) if i.passed and i.flagged]
        todo = queue[:fix]
        for n, k in enumerate(todo, 1):
            entry, diagnosis = result.items[k], diagnoses[k]
            clear = {f.failure_type for f in diagnosis.model_failures} if entry.passed else None

            def on_run(calls: int, change: str, n: int = n, item_id: str = entry.id) -> None:
                if progress:
                    progress(
                        n, len(todo), f"fix search {item_id}: re-run {calls}/{max_runs} ({change})"
                    )

            try:
                entry.fix = fix_item(
                    entry.record, entry.reference, chosen, diagnosis, caller,
                    max_runs=max_runs, must_clear=clear, on_run=on_run,
                )  # fmt: skip
            except DailyLimitError as exc:
                result.stopped = result.stopped or f"fix search stopped: {exc}"
                break
            result.fix_tried += 1

    if run is not None and fix > 0:
        result.notes.append(
            "fix search skipped: with app= counterfix cannot change the prompt or the request "
            "inside your function. Diagnose the misses, or evaluate the model call itself "
            "(model=...) to search verified fixes."
        )
    id_note = _id_mismatch_note(result.items)
    if id_note:
        result.notes.append(id_note)
    result.judge = judge_name if judging else None
    result.metrics = [
        summarize(m.name, (i.metrics[k] for i in result.items)) for k, m in enumerate(chosen)
    ]
    result.by_cause = dict(
        Counter((i.top_cause or "none found") for i in result.failed).most_common()
    )
    result.flagged_by_cause = dict(
        Counter((i.top_cause or "none found") for i in result.flagged).most_common()
    )
    tally: Counter[str] = Counter()
    for i in result.items:
        if i.fix is None:
            continue
        if i.fix.fix is not None:
            tally[i.fix.fix.name] += 1
        elif i.fix.retry is not None and i.fix.retry.verified:
            tally["plain retry passes"] += 1
        else:
            tally["no verified fix"] += 1
    result.fixes = dict(tally.most_common())
    result.speed = speed_summary(i.record for i in result.items if live)
    if n_samples > 1:
        result.pass_at_k = _pass_at_k(result, chosen, n_samples)
    code_tally: Counter[str] = Counter()
    for i in result.failed:
        check = i.record.metadata.get("code_check")
        if isinstance(check, dict) and check.get("cause"):
            code_tally[str(check["cause"])] += 1
    result.code_causes = dict(code_tally.most_common())
    return result
