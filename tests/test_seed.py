from __future__ import annotations

from collections import Counter

import jsonschema
from bench.seed import load_seed

from counterfix import FailureType


def test_seed_loads_and_covers_every_failure_type() -> None:
    items = load_seed()
    assert len(items) >= 50
    assert len({i.id for i in items}) == len(items)
    counts = Counter(i.failure_type for i in items)
    for ft in FailureType:
        if ft.is_model_failure:
            assert counts[ft] >= 5, ft
    for item in items:
        assert item.id.startswith(item.failure_type.value + "-")


def test_type_specific_fields_are_present() -> None:
    for item in load_seed():
        ft = item.failure_type
        if ft is FailureType.TRUNCATION:
            assert isinstance(item.params.get("max_tokens"), int)
        if ft is FailureType.FORMAT_BREAK:
            assert item.schema_ is not None
            jsonschema.Draft202012Validator.check_schema(item.schema_)
        if ft is FailureType.HALLUCINATION:
            assert item.gold and item.gold_context
        if ft is FailureType.DROPPED_INSTRUCTION:
            assert item.constraints and len(item.constraints) >= 2
        if ft is FailureType.PROMPT_FRAGILITY:
            assert item.variants and len(item.variants) >= 3
        if ft is FailureType.AMBIGUITY:
            assert item.readings


def test_f8_gold_answers_are_correct() -> None:
    gold = {i.id: i.gold for i in load_seed() if i.failure_type is FailureType.COMPUTATION}
    assert gold["F8-001"] == str("strawberry".count("r"))
    assert gold["F8-002"] == str(48763 * 9284)
    assert gold["F8-003"] == str("nevertheless".count("e"))
    assert gold["F8-004"] == "encyclopedia"[::-1]
    assert gold["F8-005"] == str(sum(int(c) for c in "987654321987"))
    assert gold["F8-006"] == str(17**5)
