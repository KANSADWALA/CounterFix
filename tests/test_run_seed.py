from __future__ import annotations

from typing import Any

from bench import run_seed
from bench.seed import load_seed

from conftest import FakeClient, FakeCompletions, make_response


def test_run_item_and_summary() -> None:
    items = {i.id: i for i in load_seed()}
    cut = FakeClient(FakeCompletions(response=make_response("A long essay", "length")))
    row = run_seed.run_item(cut, "m", items["F1-001"])
    assert row["found"] == ["F1"] and row["finish_reason"] == "length"

    bad_json = FakeClient(FakeCompletions(response=make_response('{"name": "Priya", "age": "29"}')))
    row2 = run_seed.run_item(bad_json, "m", items["F2-001"])
    assert row2["found"] == ["F2"]  # schema mismatch: age is a string
    assert "L4" in row2["lint"]

    class Failing(FakeCompletions):
        def create(self, **kwargs: Any) -> Any:
            raise ValueError("boom")

    row3 = run_seed.run_item(FakeClient(Failing()), "m", items["F3-001"], retries=0)
    assert row3["error"].startswith("ValueError")

    summary = "\n".join(run_seed.summarise([row, row2, row3]))
    assert "F1" in summary and "1/1" in summary and "F3" in summary
