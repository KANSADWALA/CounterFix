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

import hashlib
import inspect
import json
import random
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.caller import DailyLimitError, ModelCaller
from counterfix.core.config import Config
from counterfix.core.detector import run_detectors
from counterfix.core.diagnosis import Diagnosis, Evidence, Finding
from counterfix.core.record import CallRecord, Message
from counterfix.core.types import FailureType
from counterfix.evals.data import EvalItem, is_suite, load_suite, make_items, read_rows
from counterfix.metrics import Metric, MetricResult, get_metric
from counterfix.perf import SpeedSummary, speed_summary

Progress = Callable[[int, int, str], None]


class FixTry(BaseModel):
    name: str
    description: str
    runs: int = 0
    passed: int = 0
    verified: bool = False
    p_value: float | None = None
    """For a change: the one-sided Fisher exact p-value against the plain retry."""
    errors: int = 0
    changed: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    why_not: list[str] = Field(default_factory=list)
    """Why runs that did not pass failed, one entry per reason per run, e.g.
    "latency over 5000" or "still F1"."""

    @property
    def observed(self) -> int:
        return self.runs - self.errors

    def reasons(self) -> str:
        """The reasons, counted: 'latency over 5000 in 3/3, still F1 in 1/3'."""
        counts = Counter(self.why_not)
        return ", ".join(f"{r} in {k}/{self.runs}" for r, k in counts.most_common(3))


class ItemFix(BaseModel):
    """The fix search on one missed item."""

    retry: FixTry | None = None
    tried: list[FixTry] = Field(default_factory=list)
    fix: FixTry | None = None
    calls: int = 0
    note: str = ""


class ItemResult(BaseModel):
    id: str
    record: CallRecord
    reference: Any = None
    metrics: list[MetricResult] = Field(default_factory=list)
    passed: bool = True
    flagged: bool = False
    """Passed its metrics, but the live checks found a model failure (e.g. a cut-off
    answer that still met a latency gate)."""
    causes: list[str] = Field(default_factory=list)
    """Failure types found on this item, e.g. ["F8"], most confident first."""
    fix: ItemFix | None = None

    @property
    def top_cause(self) -> str | None:
        return self.causes[0] if self.causes else None


class MetricSummary(BaseModel):
    name: str
    n: int = 0
    mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    passed: int = 0
    failed: int = 0
    not_scored: int = 0
    """Items the metric could not score (no reference, no TTFT...). Failed calls count as
    failed, not here; the mean is over scored items only."""
    threshold: float | None = None
    higher_is_better: bool = True

    @property
    def pass_rate(self) -> float | None:
        judged = self.passed + self.failed
        return self.passed / judged if judged else None


class EvalResult(BaseModel):
    name: str
    model: str | None = None
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    items: list[ItemResult] = Field(default_factory=list)
    metrics: list[MetricSummary] = Field(default_factory=list)
    by_cause: dict[str, int] = Field(default_factory=dict)
    """Missed items by their top cause; "none found" when no check explained the miss."""
    flagged_by_cause: dict[str, int] = Field(default_factory=dict)
    """Items that passed their metrics but that the live checks flagged, by top cause."""
    fixes: dict[str, int] = Field(default_factory=dict)
    """Missed items by the verified fix ("no verified fix", "plain retry passes" too)."""
    fix_tried: int = 0
    speed: SpeedSummary = Field(default_factory=SpeedSummary)
    judge: str | None = None
    judge_errors: list[str] = Field(default_factory=list)
    """Missed items the judge could not check (e.g. its daily limit), so their cause may
    be missing: "id: error"."""
    stopped: str = ""
    samples: int = 1
    """Answers per question (``samples=n``); items are then ``id#1`` .. ``id#n``."""
    pass_at_k: dict[str, dict[int, float]] = Field(default_factory=dict)
    """With samples > 1: per pass/fail metric, pass@k (unbiased estimator) for each k."""
    code_causes: dict[str, int] = Field(default_factory=dict)
    """Code that failed its tests, by sub-cause (syntax, wrong_output, timeout...)."""
    info: dict[str, Any] = Field(default_factory=dict)
    """About the run (benchmarks: source, licence, template version, seed, subset)."""
    notes: list[str] = Field(default_factory=list)
    """Things to check about the run itself (a fix search that could not run, gold
    passage ids that never match what was retrieved)."""

    @property
    def failed(self) -> list[ItemResult]:
        return [i for i in self.items if not i.passed]

    @property
    def flagged(self) -> list[ItemResult]:
        """Passed every metric, yet a live check found a model failure."""
        return [i for i in self.items if i.passed and i.flagged]

    def metric(self, name: str) -> MetricSummary:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(
            f"no metric {name!r} in this result ({', '.join(m.name for m in self.metrics)})"
        )

    def text(self) -> str:
        from counterfix.evals.report import text_report

        return text_report(self)

    def html(self) -> str:
        from counterfix.evals.report import html_report

        return html_report(self)

    def save(self, path: str | Path) -> Path:
        """Write the whole result as JSON."""
        target = Path(path)
        target.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return target


def _id_mismatch_note(items: Sequence[ItemResult]) -> str:
    """A note when items carry gold ``relevant`` ids but no retrieved passage ever matches
    one: almost always ids from a different scheme (the dataset's vs the retriever's)."""
    from counterfix.detect.retrieval import RELEVANT_KEY, chunks_of, grades

    labelled = retrieved = matched = 0
    seen_ids: list[str] = []
    gold_ids: list[str] = []
    for i in items:
        relevant = i.record.metadata.get(RELEVANT_KEY)
        chunks = chunks_of(i.record)
        if relevant is None or not chunks:
            continue
        labelled += 1
        retrieved += len(chunks)
        g = grades(i.record) or []
        matched += sum(1 for v in g if v > 0)
        seen_ids += [str(c.id) for c in chunks if c.id is not None][:2]
        labels = [relevant] if isinstance(relevant, str) else list(relevant)
        gold_ids += [str(x) for x in labels][:2]
    if labelled < 2 or matched or not seen_ids:
        return ""
    seen = ", ".join(dict.fromkeys(seen_ids[:4]))
    gold = ", ".join(dict.fromkeys(gold_ids[:4]))
    return (
        f"no retrieved passage matched a gold `relevant` id in any of {labelled} items "
        f"(retrieved ids like {seen}; labelled {gold}). If the ids come from different "
        "schemes, retrieval metrics and F10 are wrong: label with your retriever's ids, or "
        "with the passage text."
    )


# --- statistics ------------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], *, samples: int = 1000, seed: int = 0
) -> tuple[float | None, float | None]:
    """95% confidence interval of the mean by bootstrap (seeded, so runs repeat)."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def summarize(name: str, results: Iterable[MetricResult]) -> MetricSummary:
    rs = list(results)
    values = [r.value for r in rs if r.value is not None]
    low, high = bootstrap_ci(values)
    first = rs[0] if rs else None
    return MetricSummary(
        name=name,
        n=len(values),
        mean=statistics.fmean(values) if values else None,
        ci_low=low,
        ci_high=high,
        passed=sum(r.passed is True for r in rs),
        failed=sum(r.passed is False for r in rs),
        not_scored=sum(r.value is None and r.passed is not False for r in rs),
        threshold=first.threshold if first else None,
        higher_is_better=first.higher_is_better if first else True,
    )


# --- one item ---------------------------------------------------------------------------------


def score(record: CallRecord, reference: Any, metrics: Sequence[Metric]) -> list[MetricResult]:
    return [m.score(record, reference) for m in metrics]


def _passed(record: CallRecord, results: Sequence[MetricResult]) -> bool:
    return record.error is None and all(r.passed is not False for r in results)


def _with_metric_findings(diagnosis: Diagnosis, results: Sequence[MetricResult]) -> Diagnosis:
    """Add a finding for each failed metric that names a failure type (e.g. F2)."""
    have = {f.failure_type for f in diagnosis.findings}
    for r in results:
        if r.passed is False and r.failure_type is not None and r.failure_type not in have:
            diagnosis.findings.append(
                Finding(
                    failure_type=r.failure_type,
                    confidence=0.9,
                    evidence=[Evidence(detector=f"metric:{r.name}", detail=r.reason)],
                )
            )
            have.add(r.failure_type)
    from counterfix.core.precedence import resolve

    # a metric can name the root cause (F13, F12): let it supersede live findings it explains
    kept, suppressed = resolve(diagnosis.findings)
    diagnosis.findings = kept
    diagnosis.suppressed = [*diagnosis.suppressed, *suppressed]
    return diagnosis


def _metric_miss(r: MetricResult) -> str:
    """'latency over 5000' / 'exact_match missed': short, so repeats can be counted."""
    if r.threshold is not None and r.value is not None and not r.higher_is_better:
        return f"{r.name} over {r.threshold:g}"
    return f"{r.name} missed"


def fix_item(
    record: CallRecord,
    reference: Any,
    metrics: Sequence[Metric],
    diagnosis: Diagnosis,
    caller: ModelCaller,
    *,
    runs: int = 6,
    need: int | None = None,
    max_runs: int = 30,
    must_clear: set[FailureType] | None = None,
    on_run: Callable[[int, str], None] | None = None,
    alpha: float = 0.05,
) -> ItemFix:
    """Search for a change that makes this item pass clearly more often than a plain
    retry (see :mod:`counterfix.fixes.verify`; ``need`` is no longer used).

    A plain retry runs first; if it passes too often for any change to beat, no change
    is claimed. Then the
    engine's interventions for each cause found, most confident cause first. A run
    counts only when every metric passes and the live checks find nothing new; with
    ``must_clear`` (a flagged item, whose metrics already pass) those failure types must
    also be gone. ``on_run(calls, change)`` is called before each re-run (for progress).
    """
    from counterfix.detect import default_detectors
    from counterfix.fixes.engine import INTERVENTIONS
    from counterfix.fixes.verify import beatable, level, p_value, settled

    result = ItemFix()
    checks = default_detectors()
    before = {f.failure_type for f in diagnosis.model_failures}
    causes = {f.failure_type for f in diagnosis.model_failures}
    at = level(alpha, sum(1 for i in INTERVENTIONS if i.failure_type in causes))
    base = [0, 0]

    def trial(
        attempt: FixTry, messages: list[Message], params: dict[str, Any], is_retry: bool = False
    ) -> FixTry:
        for _ in range(runs):
            if result.calls >= max_runs:
                break
            result.calls += 1
            attempt.runs += 1
            if on_run is not None:
                on_run(result.calls, attempt.name)
            try:
                new = caller.complete(messages, **params)
            except DailyLimitError:
                raise
            except Exception as exc:
                attempt.why_not.append(f"call failed ({type(exc).__name__})")
                attempt.errors += 1
                continue  # an unreadable run is not evidence either way
            new.metadata = {**record.metadata, **new.metadata, **attempt.metadata}
            left = {f.failure_type for f in run_detectors(new, checks, Config()).model_failures}
            still = (must_clear or set()) & left
            results = score(new, reference, metrics)
            if _passed(new, results) and left <= before and not still:
                attempt.passed += 1
            else:
                attempt.why_not += [_metric_miss(r) for r in results if r.passed is False]
                attempt.why_not += [f"still {t.value}" for t in sorted(still, key=str)]
                attempt.why_not += [f"new {t.value}" for t in sorted(left - before, key=str)]
            if is_retry:
                if not beatable(attempt.passed, attempt.observed, runs, at):
                    break
            elif settled(
                base[0],
                base[1],
                attempt.passed,
                attempt.observed,
                attempt.observed + (runs - attempt.runs),
                at,
            ):
                break
        if is_retry:
            attempt.verified = not beatable(attempt.passed, attempt.observed, runs, at)
        else:
            attempt.p_value = p_value(base[0], base[1], attempt.passed, attempt.observed)
            attempt.verified = attempt.passed > 0 and attempt.p_value <= at
        return attempt

    if need is not None:
        import warnings

        warnings.warn(
            "fix_item(need=...) is ignored: a change is now verified when it beats the "
            "plain retry (see counterfix.fixes.verify)",
            DeprecationWarning,
            stacklevel=2,
        )
    result.retry = trial(
        FixTry(name="plain_retry", description="Re-run unchanged"),
        record.messages,
        record.params,
        is_retry=True,
    )
    base[0], base[1] = result.retry.passed, result.retry.observed
    if result.retry.verified:
        result.note = "a plain retry passes: the failure was not repeatable"
        return result
    failures = sorted(diagnosis.model_failures, key=lambda f: -f.confidence)
    if not failures:
        result.note = "no cause found, so no targeted change to try"
        return result
    for finding in failures:
        for iv in INTERVENTIONS:
            if iv.failure_type is not finding.failure_type:
                continue
            change = iv.apply(record, finding)
            if change is None:
                continue
            if result.calls >= max_runs:
                result.note = f"stopped at the budget of {max_runs} re-runs"
                return result
            messages, params = change
            attempt = FixTry(
                name=iv.name,
                description=iv.description,
                changed={k: v for k, v in params.items() if record.params.get(k) != v},
                messages=None if messages == record.messages else messages,
                metadata=iv.metadata(record, finding) if iv.metadata else {},
            )
            result.tried.append(trial(attempt, messages, params))
            if attempt.verified:
                result.fix = attempt
                result.note = "stopped at the first verified fix"
                return result
    result.note = result.note or "no change beat the plain retry clearly enough"
    return result


# --- the run ----------------------------------------------------------------------------------


def _caller(model: str | ModelCaller) -> ModelCaller:
    if not isinstance(model, str):
        return model
    from counterfix.core.caller import RateLimitedCaller
    from counterfix.core.presets import caller_for, pace_for

    return RateLimitedCaller(caller_for(model, timeout=120), pace_for(model))


def _fingerprint(item: EvalItem) -> str:
    """What was asked: a kept answer is reused only for the same messages and params."""
    payload = json.dumps(
        [[m.model_dump(mode="json", exclude_none=True) for m in item.messages], item.params],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cached(path: Path | None, model: str | None) -> dict[tuple[str, str], CallRecord]:
    if path is None or not path.exists():
        return {}
    done: dict[tuple[str, str], CallRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("model") == model:
                key = (str(row["id"]), str(row.get("asked", "")))
                done[key] = CallRecord.model_validate(row["record"])
    return done


def _question(item: EvalItem) -> str:
    """The question an app is asked: the dataset's question, not the prompt built for it."""
    from counterfix.detect.retrieval import QUESTION_KEY
    from counterfix.evals.data import INPUT_KEY

    for key in (QUESTION_KEY, INPUT_KEY):
        asked = item.metadata.get(key)
        if isinstance(asked, str) and asked:
            return asked
    users = [m for m in item.messages if m.role == "user"]
    return users[-1].text if users else ""


def _run_app(app: Callable[[str], Any], item: EvalItem) -> CallRecord:
    """Call the user's app on one item and read what it returned."""
    from counterfix.detect.retrieval import QUESTION_KEY, record_from_output

    question = _question(item)
    start = time.perf_counter()
    try:
        output = app(question)
        if inspect.isawaitable(output):
            from counterfix.core.sync import resolve

            output = resolve(output)
    except DailyLimitError:
        raise
    except Exception as exc:
        return CallRecord(
            messages=[Message(role="user", content=question)],
            error=f"{type(exc).__name__}: {exc}",
            metadata={QUESTION_KEY: question},
        )
    latency = (time.perf_counter() - start) * 1000
    return record_from_output(question, output, latency_ms=latency)


def _ask_all(
    items: Sequence[EvalItem],
    ask: Callable[[EvalItem], CallRecord],
    workers: int,
    progress: Progress | None,
) -> tuple[list[tuple[EvalItem, CallRecord]], str | None]:
    """Answer items on ``workers`` threads; results keep the items' order. A daily limit
    stops new work; items already answered are kept."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stop = threading.Event()
    answers: dict[int, CallRecord] = {}
    stopped: str | None = None

    def one(k: int) -> CallRecord | None:
        return None if stop.is_set() else ask(items[k])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, k): k for k in range(len(items))}
        done = 0
        for future in as_completed(futures):
            k = futures[future]
            done += 1
            try:
                record = future.result()
            except DailyLimitError as exc:
                stop.set()
                stopped = stopped or f"stopped early: {exc}"
                continue
            if record is not None:
                answers[k] = record
            if progress:
                progress(done, len(items), items[k].id)
    ordered = [(items[k], answers[k]) for k in sorted(answers)]
    if ordered and all(r.error for _, r in ordered) and len(ordered) >= 3:
        stopped = stopped or f"every call failed ({ordered[0][1].error})"
    return ordered, stopped


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
    from counterfix.metrics import JudgeMetric

    judged_metrics = [m for m in chosen if isinstance(m, JudgeMetric) and m.judge is None]
    if judged_metrics and judge in (False, None):
        judge = True  # judge metrics need a judge; it also explains the misses (F10/F11)
    judging = judge not in (False, None)
    if judging:  # choose once, before any call: a missing key fails fast, not mid-run
        from counterfix.judges.auto import resolve_judge

        if isinstance(model, str):
            from counterfix.core.presets import load_env_file

            load_env_file(Path.cwd() / ".env")
        target = model.split(":", 1)[-1] if isinstance(model, str) else None
        wanted = None if judge in (True, "auto") else judge
        judge = resolve_judge(wanted, target_model=target)[0]
        for m in judged_metrics:
            m.bind(judge)

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


def _check_code_execution(metrics: Sequence[Metric]) -> None:
    """Fail before any call when a metric would run code that is not allowed to run."""
    from counterfix.detect.code.sandbox import CodeExecutionDisabled, execution_allowed

    for m in metrics:
        if getattr(m, "needs_code_execution", False) and not execution_allowed(
            m.options.get("allow")
        ):
            raise CodeExecutionDisabled()


def _repeat(items: Sequence[EvalItem], n: int) -> list[EvalItem]:
    """Each item n times, as ``id#1`` .. ``id#n`` (the base id kept in metadata)."""
    out: list[EvalItem] = []
    for item in items:
        for k in range(1, n + 1):
            copy = item.model_copy(deep=True)
            copy.id = f"{item.id}#{k}"
            copy.metadata[SAMPLE_KEY] = item.id
            out.append(copy)
    return out


SAMPLE_KEY = "sample_of"


def _pass_at_k(
    result: EvalResult, metrics: Sequence[Metric], n: int
) -> dict[str, dict[int, float]]:
    """pass@k per pass/fail metric over the questions, for k in 1, 5, 10, 100 up to n."""
    from counterfix.detect.code.passk import mean_pass_at_k

    ks = [k for k in (1, 5, 10, 100) if k <= n] or [1]
    out: dict[str, dict[int, float]] = {}
    for j, m in enumerate(metrics):
        if m.threshold is None:
            continue
        groups: dict[str, list[bool]] = {}
        for item in result.items:
            r = item.metrics[j]
            base = str(item.record.metadata.get(SAMPLE_KEY) or item.id.rsplit("#", 1)[0])
            groups.setdefault(base, []).append(r.passed is True)
        counts = [(len(v), sum(v)) for v in groups.values()]
        values = {k: v for k in ks if (v := mean_pass_at_k(counts, k)) is not None}
        if values:
            out[m.name] = values
    return out
