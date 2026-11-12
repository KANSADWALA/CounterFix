"""Moved to :mod:`counterfix.core.rules` (a Config holds a RuleStore, so it belongs in
core). This path keeps working."""

from counterfix.core.rules import BUILTIN_RULES, FixRule, RuleStore

__all__ = ["BUILTIN_RULES", "FixRule", "RuleStore"]
