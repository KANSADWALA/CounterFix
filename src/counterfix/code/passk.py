"""pass@k, the unbiased estimator from the HumanEval paper (Chen et al., 2021):
with n samples of which c pass, pass@k = 1 - C(n-c, k) / C(n, k)."""

from __future__ import annotations

from collections.abc import Iterable


def pass_at_k(n: int, c: int, k: int) -> float:
    """The chance that at least one of k samples, drawn from these n, passes."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be between 0 and n ({n}), got {c}")
    if n - c < k:
        return 1.0
    value = 1.0
    for i in range(n - c + 1, n + 1):  # the numerically stable product form
        value *= 1.0 - k / i
    return 1.0 - value


def mean_pass_at_k(counts: Iterable[tuple[int, int]], k: int) -> float | None:
    """pass@k averaged over problems, from (n, c) per problem; None with no problems or
    when a problem has fewer than k samples."""
    rows = list(counts)
    values = [pass_at_k(n, c, k) for n, c in rows if n >= k]
    if not values or len(values) != len(rows):
        return None
    return sum(values) / len(values)
