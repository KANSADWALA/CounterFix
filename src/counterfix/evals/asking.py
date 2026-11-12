"""Getting the answers: from the model (in parallel, with a resumable cache) or from your
own app function."""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from counterfix.core.caller import DailyLimitError
from counterfix.core.record import CallRecord, Message
from counterfix.evals.data import EvalItem

Progress = Callable[[int, int, str], None]


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
