"""Features that duplicate other tools warn in 0.3 (removal planned for 0.4); kept ones never do."""

from __future__ import annotations

import sys
import warnings

import pytest

from counterfix._internal import deprecation
from counterfix.evals.benchmarks import get_benchmark
from counterfix.evals.metrics import METRICS, get_metric


@pytest.mark.parametrize("name", sorted(deprecation.METRICS))
def test_deprecated_metrics_warn_and_name_an_alternative(name: str) -> None:
    with pytest.warns(FutureWarning, match=r"removed in counterfix 0\.4: use "):
        get_metric(name)


def test_deprecated_names_are_real() -> None:
    assert set(deprecation.METRICS) <= set(METRICS)
    for name in deprecation.BENCHMARKS:
        with pytest.warns(FutureWarning):
            get_benchmark(name)


@pytest.mark.parametrize("name", sorted(set(METRICS) - set(deprecation.METRICS)), ids=lambda n: n)
def test_kept_metrics_do_not_warn(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        get_metric(name)


@pytest.mark.parametrize("name", ["injection", "tools"])
def test_own_probe_benchmarks_are_kept(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        get_benchmark(name)


def test_safety_and_perplexity_functions_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    from counterfix.evals.safety import find_pii

    with pytest.warns(FutureWarning, match="Presidio"):
        assert find_pii("mail me at a@b.co")
    from counterfix.evals.perplexity import perplexity

    monkeypatch.setitem(sys.modules, "torch", None)  # stop before loading any model
    with pytest.warns(FutureWarning, match="lm-evaluation-harness"), pytest.raises(ImportError):
        perplexity("text")


def test_a_deprecated_metric_warns_once() -> None:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        metric = get_metric("pii")
        from counterfix.core.record import CallRecord, Message

        record = CallRecord(
            messages=[Message(role="user", content="hi")], response_text="call 555-0100"
        )
        metric.score(record, None)
    assert len([w for w in seen if issubclass(w.category, FutureWarning)]) == 1
