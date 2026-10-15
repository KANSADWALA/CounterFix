"""A file cache for judge verdicts, so re-running an evaluation costs no API calls."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from counterfix.judges.base import Judge, JudgeTask, Verdict


class CachedJudge:
    """Wraps a judge; stores verdicts in a JSON-lines file keyed by (judge, task).
    Error verdicts are never cached, so failed calls are retried next time."""

    def __init__(self, judge: Judge, path: str | Path, *, offline: bool = False):
        self.judge = judge
        self.offline = offline  # True: answer only from the cache, never call the judge
        self.path = Path(path)
        self._lock = threading.Lock()
        self._store: dict[str, Verdict] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self._store[item["key"]] = Verdict.model_validate(item["verdict"])
        self.hits = 0

    @property
    def name(self) -> str:
        return self.judge.name

    def key(self, task: JudgeTask) -> str:
        blob = json.dumps([self.judge.name, task.model_dump(mode="json")], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def ask(self, task: JudgeTask) -> Verdict:
        key = self.key(task)
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        if self.offline:
            return Verdict(
                task=task.name,
                label="error",
                confidence=0,
                judge=self.name,
                reason="not in cache (offline mode)",
            )
        verdict = self.judge.ask(task)
        if not verdict.failed:
            with self._lock:
                self._store[key] = verdict
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"key": key, "verdict": verdict.model_dump(mode="json")}) + "\n"
                    )
        return verdict

    def cache_only(self) -> CachedJudge:
        """A view of the same cache that never calls the judge."""
        view = CachedJudge.__new__(CachedJudge)
        view.__dict__.update(self.__dict__)
        view.offline = True
        return view

    def __len__(self) -> int:
        return len(self._store)
