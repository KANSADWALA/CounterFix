"""When a change counts as a fix: the test behind "verified".

A changed call and a plain retry of the original call are each re-run the same number of
times (``runs``, default 6). The change is *verified* only if it passes clearly more
often than the plain retry: a one-sided Fisher exact test on the two pass counts must
give a p-value at or below ``alpha`` divided by the number of changes the search may
try (Bonferroni), so trying many changes does not make a lucky one more likely to be
called a fix. With the defaults, a model that fails at random, whatever the request
says, gets a change called verified in well under 5% of searches.

What it means in practice, with 6 runs each and up to 3 changes tried (level 0.0167):

====================  ===================  ===========
plain retry passed    the change passed    verified?
====================  ===================  ===========
0/6                   6/6 or 5/6           yes
0/6                   4/6                  no (p=0.03)
1/6                   6/6                  yes
2/6                   6/6                  no: a retry passes too often to tell
====================  ===================  ===========

What it does not mean: that the change works on other prompts, or on this prompt every
time. It means that, on this call, the change did better than asking again.

Runs stop early once the outcome is settled either way, so a change that cannot win any
more costs no further calls.
"""

from __future__ import annotations

from math import comb

DEFAULT_RUNS = 6
"""Re-runs of the plain retry and of each change."""

DEFAULT_ALPHA = 0.05
"""Chance, across all the changes one search tries, of calling a useless change a fix."""


def p_value(base_passed: int, base_runs: int, passed: int, runs: int) -> float:
    """One-sided Fisher exact test: how likely a change that is no better than the plain
    retry would pass at least ``passed`` of ``runs``, given the pass counts of both."""
    if runs <= 0 or base_runs < 0:
        return 1.0
    total = base_runs + runs
    k = base_passed + passed
    denom = comb(total, runs)
    if denom == 0:
        return 1.0
    tail = sum(comb(k, x) * comb(total - k, runs - x) for x in range(passed, min(k, runs) + 1))
    return min(1.0, tail / denom)


def level(alpha: float, tests: int) -> float:
    """The p-value one change must reach when a search may try ``tests`` changes."""
    return alpha / max(1, tests)


def wins(base_passed: int, base_runs: int, passed: int, runs: int, at: float) -> bool:
    """True when ``passed`` of ``runs`` beats ``base_passed`` of ``base_runs`` at ``at``."""
    return passed > 0 and p_value(base_passed, base_runs, passed, runs) <= at


def settled(
    base_passed: int, base_runs: int, passed: int, done: int, planned: int, at: float
) -> bool:
    """True when more runs of a change cannot change the verdict: it already wins even if
    every remaining run fails, or it cannot win even if every remaining run passes."""
    left = max(0, planned - done)
    if left == 0:
        return True
    if wins(base_passed, base_runs, passed, done + left, at):  # even if the rest all fail
        return True
    return not wins(base_passed, base_runs, passed + left, done + left, at)  # or all pass


def beatable(base_passed: int, base_done: int, planned: int, at: float) -> bool:
    """False once the plain retry passes so often that even a change passing every one of
    ``planned`` runs could not be shown to do better (counting the retry's remaining runs
    as failures, the most generous case for a change)."""
    return wins(base_passed, max(base_done, planned), planned, planned, at)


__all__ = ["DEFAULT_ALPHA", "DEFAULT_RUNS", "beatable", "level", "p_value", "settled", "wins"]
