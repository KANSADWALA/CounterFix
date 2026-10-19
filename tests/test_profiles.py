from __future__ import annotations

import json
from pathlib import Path

import counterfix
from counterfix import profiles
from counterfix.cli import main
from counterfix.profiles import Profile
from test_engine import Scripted, cut_off


def test_shipped_probes_match_the_bench_seed_set() -> None:
    bench = Path(__file__).resolve().parents[1] / "bench" / "seed_failures.jsonl"
    assert profiles.PROBES.read_text(encoding="utf-8") == bench.read_text(encoding="utf-8")


def test_fix_order_prefers_what_worked_and_demotes_what_never_did() -> None:
    p = Profile(model="m", fixes={"F1": {"raise_max_tokens": [0, 3], "ask_shorter": [2, 2]}})
    names = ["raise_max_tokens", "raise_max_tokens_more", "ask_shorter"]
    assert p.fix_order("F1", names) == ["ask_shorter", "raise_max_tokens_more", "raise_max_tokens"]
    assert p.fix_order("F2", ["a", "b"]) == ["a", "b"]


def test_engine_uses_the_stored_profile_to_save_re_runs() -> None:
    def answer(m, p):
        return ("Mercury ... Neptune.", "stop") if p["max_tokens"] > 100 else ("Me", "length")

    plain = counterfix.fix(cut_off(), judge=False, caller=Scripted(answer), profile=None)
    assert plain.fix is not None and plain.fix.name == "raise_max_tokens_more"
    learned = Profile(model="m")
    learned.learn(plain)
    assert learned.fixes["F1"] == {"raise_max_tokens": [0, 1], "raise_max_tokens_more": [1, 1]}
    profiles.save(learned)
    again = counterfix.fix(cut_off(), judge=False, caller=Scripted(answer))  # profile="auto"
    assert again.attempts[0].name == "raise_max_tokens_more"
    assert again.calls < plain.calls


def test_build_compare_drift_route(tmp_path: Path) -> None:
    probes = tmp_path / "probes.jsonl"
    rows = [
        {
            "id": "a",
            "failure_type": "F1",
            "messages": [{"role": "user", "content": "Essay"}],
            "params": {"max_tokens": 4},
        },
        {"id": "b", "failure_type": "F2", "messages": [{"role": "user", "content": "JSON"}]},
        {
            "id": "c",
            "failure_type": "F2",
            "messages": [{"role": "user", "content": "x"}],
            "control": True,
        },
    ]
    probes.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    weak = profiles.build_profile(
        Scripted(lambda m, p: ("Es", "length") if p.get("max_tokens") else ("Done.", "stop")),
        probes=probes,
    )
    strong = profiles.build_profile(Scripted(lambda m, p: ("Done.", "stop")), probes=probes)
    assert weak.categories == {"F1": [1, 1], "F2": [0, 1]}  # the control probe is skipped
    assert strong.failure_rate("F1") == 0
    assert "better" in profiles.compare(weak, strong)
    assert profiles.drift(weak, strong) == ["F1 long answers under a token limit: 100% -> 0%"]
    weak.model, strong.model = "cheap", "pricey"
    weak.cost_per_mtok, strong.cost_per_mtok = 0.1, 2.0
    assert profiles.route([strong, weak]) == {"F1": "pricey", "F2": "cheap"}

    fixed = profiles.build_profile(
        Scripted(lambda m, p: ("Es", "length") if p.get("max_tokens") == 4 else ("Done.", "stop")),
        probes=probes,
        fix=True,
    )
    assert fixed.fixes["F1"]["raise_max_tokens"] == [1, 1]

    profiles.save(weak, tmp_path / "cheap.json")
    profiles.save(strong, tmp_path / "pricey.json")
    assert main(["compare", str(tmp_path / "cheap.json"), str(tmp_path / "pricey.json")]) == 0
    assert main(["route", str(tmp_path / "cheap.json"), str(tmp_path / "pricey.json")]) == 0
    assert main(["route", "missing-model"]) == 2


def test_docs_list_every_intervention_on_its_failure_page() -> None:
    from counterfix.engine import INTERVENTIONS

    docs = Path(__file__).resolve().parents[1] / "docs" / "failures"
    for iv in INTERVENTIONS:
        page = (docs / f"{iv.failure_type.value}.md").read_text(encoding="utf-8")
        assert f"`{iv.name}`" in page, f"docs/failures/{iv.failure_type.value}.md lacks {iv.name}"
