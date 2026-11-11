"""Evals: run a model over a dataset, score it, explain the misses, verify fixes."""

from counterfix.evals.data import EvalItem, load_suite, make_items, read_rows
from counterfix.evals.runner import (
    EvalResult,
    ItemFix,
    ItemResult,
    MetricSummary,
    bootstrap_ci,
    evaluate,
    fix_item,
)

__all__ = [
    "EvalItem",
    "EvalResult",
    "ItemFix",
    "ItemResult",
    "MetricSummary",
    "bootstrap_ci",
    "evaluate",
    "fix_item",
    "load_suite",
    "make_items",
    "read_rows",
]
