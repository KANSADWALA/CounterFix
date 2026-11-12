"""The public surface of counterfix, frozen: the everyday names, the shortcuts, the
subpackages people import from, and the entry points. Moving code is fine; breaking these
is not. When a name is deliberately retired, remove it here in the same commit and say so
in the CHANGELOG."""

from __future__ import annotations

import importlib

import pytest

import counterfix

EVERYDAY = [
    "CallRecord",
    "Config",
    "Diagnosis",
    "FailureType",
    "FixResult",
    "Message",
    "__version__",
    "configure",
    "diagnose",
    "fix",
    "guard",
    "log_to",
    "watch",
]

SHORTCUTS = [
    "Detector",
    "EvalResult",
    "Evidence",
    "Finding",
    "FixAttempt",
    "FixRule",
    "GuardedClient",
    "LintIssue",
    "NoJudgeError",
    "Prescription",
    "PrescriptionKind",
    "RetryAttempt",
    "RuleStore",
    "SelfJudgeWarning",
    "SkipCheck",
    "Timing",
    "Usage",
    "background",
    "context",
    "evaluate",
    "get_diagnosis",
    "hooks",
    "langchain_callback",
    "lint",
    "litellm_callback",
    "load_log",
    "metrics",
    "otel",
    "perf",
    "record_from_openai",
    "record_from_responses",
    "run_detectors",
]

MODULES = [
    "counterfix.core",
    "counterfix.detect",
    "counterfix.diagnose",
    "counterfix.evals",
    "counterfix.evals.benchmarks",
    "counterfix.evals.metrics",
    "counterfix.fixes",
    "counterfix.fixes.engine",
    "counterfix.fixes.verify",
    "counterfix.integrations",
    "counterfix.integrations.eval_results",
    "counterfix.integrations.otel",
    "counterfix.judges",
    "counterfix.lint",
    "counterfix.runtime",
    "counterfix.runtime.context",
    "counterfix.runtime.perf",
    "counterfix.runtime.report",
    "counterfix.testing",
]

RETIRED_IN_0_4 = [
    "counterfix.adapters",  # 0.2 paths: now counterfix.integrations, .detect, .fixes ...
    "counterfix.detectors",
    "counterfix.engine",
    "counterfix.evals.perplexity",  # use lm-evaluation-harness
    "counterfix.evals.safety",  # use Presidio, Detoxify
    "counterfix.evals.metrics.rag",  # use Ragas or DeepEval
    "counterfix.evals.metrics.semantic",  # use bert-score, sentence-transformers
    "counterfix.evals.benchmarks.evalplus",  # use EvalPlus
]


def test_all_is_the_everyday_api() -> None:
    assert sorted(counterfix.__all__) == sorted(EVERYDAY)


@pytest.mark.parametrize("name", EVERYDAY + SHORTCUTS)
def test_name_imports(name: str) -> None:
    assert getattr(counterfix, name) is not None


@pytest.mark.parametrize("path", MODULES)
def test_module_imports(path: str) -> None:
    importlib.import_module(path)


@pytest.mark.parametrize("path", RETIRED_IN_0_4)
def test_retired_paths_are_gone(path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


def test_pytest_plugin_entry_point_loads() -> None:
    from importlib.metadata import entry_points

    (plugin,) = [ep for ep in entry_points(group="pytest11") if ep.name == "counterfix"]
    assert plugin.value == "counterfix.testing"
    plugin.load()


def test_cli_entry_point_loads() -> None:
    from importlib.metadata import entry_points

    (script,) = [ep for ep in entry_points(group="console_scripts") if ep.name == "counterfix"]
    assert callable(script.load())
