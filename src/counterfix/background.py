"""Judge checks on live traffic, off the request path.

With ``Config(sample_rate=0.05)``, every call the live checks flag, plus about 5% of the
others, is queued for :func:`counterfix.diagnose` on one background thread. The user's
call returns at once; the full diagnosis goes to ``Config.on_judged`` (e.g. ``log_to``).

Cost stays bounded: at most ``Config.max_judged_per_hour`` calls are judged per rolling
hour, and when the queue is full new calls are dropped (counted in ``stats``), never
waited on. Pending work is dropped at exit unless you call :func:`flush`.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from collections import deque
from typing import Any

from counterfix.core.config import Config
from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord

logger = logging.getLogger("counterfix")

QUEUE_SIZE = 100
stats = {"queued": 0, "judged": 0, "dropped_full": 0, "dropped_cap": 0, "errors": 0}

_queue: queue.Queue[tuple[CallRecord, Config, Any]] = queue.Queue(QUEUE_SIZE)
_worker: threading.Thread | None = None
_start_lock = threading.Lock()
_judged_at: deque[float] = deque()  # ponytail: one cap for the process, per-Config if needed


def wants(diagnosis: Diagnosis, config: Config) -> bool:
    """Should this call get judge checks? Flagged calls always; others at the sample rate."""
    if config.sample_rate <= 0:
        return False
    return not diagnosis.ok or random.random() < config.sample_rate


def submit(record: CallRecord, config: Config, client: Any = None) -> bool:
    """Queue one call for judge checks. Never blocks; returns False if it was dropped."""
    now = time.monotonic()
    while _judged_at and now - _judged_at[0] > 3600:
        _judged_at.popleft()
    if len(_judged_at) >= config.max_judged_per_hour:
        stats["dropped_cap"] += 1
        return False
    try:
        _queue.put_nowait((record, config, client))
    except queue.Full:
        stats["dropped_full"] += 1
        return False
    _judged_at.append(now)
    stats["queued"] += 1
    _ensure_worker()
    return True


def flush(timeout: float | None = None) -> bool:
    """Wait until queued calls are judged. True if the queue emptied in time."""
    end = None if timeout is None else time.monotonic() + timeout
    while _queue.unfinished_tasks:
        if end is not None and time.monotonic() > end:
            return False
        time.sleep(0.05)
    return True


def _ensure_worker() -> None:
    global _worker
    with _start_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="counterfix-judge", daemon=True)
            _worker.start()


def _run() -> None:
    from counterfix.diagnose import _judged

    while True:
        record, config, client = _queue.get()
        try:
            diagnosis = _judged(record, client, config.judge, config)
            stats["judged"] += 1
            if config.on_judged is not None:
                config.on_judged(record, diagnosis)
        except Exception as exc:  # never let one bad call stop the worker
            stats["errors"] += 1
            logger.warning("counterfix background judge check failed: %s", exc)
        finally:
            _queue.task_done()
