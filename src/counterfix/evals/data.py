"""Eval datasets and suite files.

A dataset is JSONL, JSON (a list) or CSV, one item per row. Each row becomes chat
messages, an optional reference answer and, when the answers already exist, the
response to score (then no model is called):

* the prompt: ``messages``, else ``template`` filled from the row's fields, else the
  ``input`` field, else the first of ``prompt``, ``question``, ``input``;
* the reference: the ``reference`` field, else the first of ``reference``, ``answer``,
  ``expected``, ``target``, ``ideal``;
* a recorded answer: ``response`` or ``output``;
* RAG: retrieved passages in ``contexts`` (or ``context``, ``chunks``, ``retrieved``,
  ``documents``, ``passages``; strings or ``{"id", "text"}``), in rank order; gold labels
  in ``relevant`` (ids or passage texts); gold passage texts in ``gold_passages``. With
  passages and no template, the prompt is built from the question and the numbered
  passages (a template can use ``{context}``).

A suite file (``.json``, or ``.yaml`` with PyYAML installed) names the data and the
settings, so a run is one command::

    name: gsm8k-sample
    data: gsm8k_50.jsonl      # relative to the suite file
    model: groq:openai/gpt-oss-120b
    metrics: [number, latency:5000]
    template: "{question}\\nGive the final answer after ####."
    reference: answer
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from counterfix.core.record import Message

_INPUTS = ("prompt", "question", "input")
_CHUNKS = ("contexts", "context", "chunks", "retrieved", "documents", "passages")
_RELEVANT = ("relevant", "relevant_ids", "gold_ids", "gold")
_REFERENCES = ("reference", "answer", "expected", "target", "ideal")
_RESPONSES = ("response", "output")
INPUT_KEY = "eval_input"
SUITE_KEYS = {
    "name", "data", "model", "metrics", "input", "reference", "template", "system",
    "params", "limit", "judge", "fix", "max_runs", "fail_under", "fail_on_flagged",
    "app", "concurrency", "samples",
}  # fmt: skip


class EvalItem(BaseModel):
    """One test case."""

    id: str
    messages: list[Message]
    params: dict[str, Any] = Field(default_factory=dict)
    reference: Any = None
    response: str | None = None
    """A recorded answer to score instead of calling a model."""
    fields: dict[str, Any] = Field(default_factory=dict)
    """The original row, for reports."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Put on the call's record: retrieved passages, gold labels, the reference."""


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Rows from a .jsonl, .json (list, or object with "items") or .csv file."""
    p = Path(path)
    if p.suffix == ".csv":  # newline="" keeps quoted multi-line cells whole
        with p.open(encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    text = p.read_text(encoding="utf-8-sig")  # -sig: files saved by Excel start with a BOM
    if p.suffix == ".json":
        data = json.loads(text)
        rows = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a list of items")
        return rows
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{n}: not JSON ({exc})") from exc
    return rows


def _first(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def make_items(
    rows: Iterable[Mapping[str, Any]],
    *,
    input: str | None = None,
    reference: str | None = None,
    template: str | None = None,
    system: str | None = None,
    params: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> list[EvalItem]:
    """Turn rows into :class:`EvalItem` (see the module docstring for the field rules)."""
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, row in enumerate(rows):
        if limit is not None and len(items) >= limit:
            break
        if not isinstance(row, Mapping):
            raise ValueError(f"item {n + 1}: expected an object, got {type(row).__name__}")
        meta: dict[str, Any] = {}
        raw_chunks = _first(row, _CHUNKS)
        chunks = None
        if raw_chunks not in (None, "", []):
            from counterfix.judges.tasks import CONTEXT_KEY
            from counterfix.rag import CHUNKS_KEY, RELEVANT_KEY, joined, to_chunks

            if isinstance(raw_chunks, str):
                raw_chunks = [raw_chunks]
            chunks = to_chunks(raw_chunks)
            meta[CHUNKS_KEY] = [c.model_dump() for c in chunks]
            meta[CONTEXT_KEY] = joined(chunks)
        relevant = _first(row, _RELEVANT)
        if relevant not in (None, "", []):  # gold labels also score an app's own retrieval
            from counterfix.rag import RELEVANT_KEY

            meta[RELEVANT_KEY] = relevant
        if isinstance(row.get("gold_passages"), list):
            meta["gold_passages"] = row["gold_passages"]
        if isinstance(row.get("expect_tools"), list):  # agent runs: tools that must be called
            meta["expect_tools"] = [str(t) for t in row["expect_tools"]]
        asked = row.get(input) if input else _first(row, _INPUTS)
        if isinstance(asked, str) and asked:
            meta[INPUT_KEY] = asked  # the bare question, for evaluate(app=...)
        if row.get("messages"):
            messages = [Message.model_validate(m) for m in row["messages"]]
        elif chunks is not None and template is None:
            from counterfix.rag import PROMPT_KEY, QUESTION_KEY, rag_messages

            question = row.get(input) if input else _first(row, _INPUTS)
            if question in (None, ""):
                raise ValueError(f"item {n + 1}: has passages but no question")
            messages = rag_messages(str(question), chunks, system=system)
            meta.update({PROMPT_KEY: True, QUESTION_KEY: str(question)})
            if system:
                meta["rag_system"] = system
        else:
            prompt: Any
            if template is not None:
                try:
                    fill = dict(row)
                    if chunks is not None:  # {context} in a template: the numbered passages
                        from counterfix.rag import joined

                        fill["context"] = joined(chunks)
                    prompt = template.format(**fill)
                except KeyError as exc:
                    raise ValueError(f"item {n + 1}: template field {exc} is missing") from exc
            else:
                prompt = row.get(input) if input else _first(row, _INPUTS)
            if prompt in (None, ""):
                raise ValueError(
                    f"item {n + 1}: no prompt (give messages, a template, or one of "
                    f"{', '.join(_INPUTS)})"
                )
            messages = [Message(role="user", content=str(prompt))]
        if system and not any(m.role == "system" for m in messages):
            messages.insert(0, Message(role="system", content=system))
        ref = row.get(reference) if reference else _first(row, _REFERENCES)
        if isinstance(ref, str):
            from counterfix.judges.tasks import REFERENCE_KEY

            meta[REFERENCE_KEY] = ref
        item_id = str(row.get("id", n + 1))
        if item_id in seen:
            raise ValueError(f"item {n + 1}: id {item_id!r} is used twice; ids must be unique")
        seen.add(item_id)
        items.append(
            EvalItem(
                id=item_id,
                messages=messages,
                params={
                    **(params or {}),
                    **({"tools": row["tools"]} if isinstance(row.get("tools"), list) else {}),
                    **(row.get("params") or {}),
                },
                reference=ref,
                response=_first(row, _RESPONSES),
                fields=dict(row),
                metadata=meta,
            )
        )
    return items


def load_suite(path: str | Path) -> dict[str, Any]:
    """A suite file as a dict; ``data`` is resolved relative to the suite file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError('YAML suites need PyYAML: pip install "counterfix[eval]"') from exc
        suite = yaml.safe_load(text)  # type: ignore[no-untyped-call, unused-ignore]
    else:
        suite = json.loads(text)
    if not isinstance(suite, dict) or "data" not in suite:
        raise ValueError(f"{p}: a suite needs at least a 'data' entry")
    unknown = set(suite) - SUITE_KEYS
    if unknown:
        raise ValueError(
            f"{p}: unknown suite keys {sorted(unknown)}; allowed: {sorted(SUITE_KEYS)}"
        )
    data = Path(suite["data"])
    suite["data"] = str(data if data.is_absolute() else p.parent / data)
    suite.setdefault("name", p.stem)
    return suite


def is_suite(path: str | Path) -> bool:
    """True for a suite file: .yaml/.yml, or a .json object with a "data" entry."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return True
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(data, dict) and "data" in data
    return False


def load_callable(spec: str) -> Any:
    """``package.module:function`` (or ``path/to/file.py:function``) -> the function."""
    import importlib
    import importlib.util
    import sys

    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError("use module:function, e.g. myapp.rag:answer")
    if target.endswith(".py"):
        location = Path(target).resolve()
        loaded = importlib.util.spec_from_file_location(location.stem, location)
        if loaded is None or loaded.loader is None:
            raise ImportError(f"cannot load {target}")
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
    else:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        module = importlib.import_module(target)
    fn = getattr(module, attr)
    if not callable(fn):
        raise ValueError(f"{attr} is not callable")
    return fn
