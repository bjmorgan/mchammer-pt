"""Pair-set rotation and Metropolis acceptance helpers for
parallel-tempering exchange proposals.

- `pair_set_for_cycle` returns the indices of adjacent replica pairs
  whose exchanges should be attempted in a given cycle. Even and odd
  pair sets alternate cycle-by-cycle to decorrelate successive swap
  attempts.
- `metropolis_accept` applies the standard Metropolis acceptance
  rule to a log-probability ratio.
"""

from __future__ import annotations

import numpy as np


def pair_set_for_cycle(n_replicas: int, cycle: int) -> np.ndarray:
    """Indices of adjacent pairs attempted for exchange in this cycle.

    Pair ``i`` refers to replicas ``i`` and ``i + 1``. Even-indexed
    pairs are attempted on even cycles; odd-indexed pairs on odd
    cycles. This alternation is standard in parallel-tempering
    practice; each pair is proposed on exactly every other cycle.

    Args:
        n_replicas: total number of replicas.
        cycle: zero-based cycle index.

    Returns:
        1-D integer array of pair indices in ascending order.
    """
    start = cycle % 2
    return np.arange(start, n_replicas - 1, 2)


def matching_for_boundary(
    n_walkers_lo: int,
    n_walkers_hi: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Random matching of two windows' walkers for one boundary.

    Pairs walkers of the lower window against walkers of the upper
    window by independently shuffling each window's walker indices and
    zipping to ``min(n_walkers_lo, n_walkers_hi)`` disjoint pairs. When
    the counts differ, the surplus walkers of the larger window are
    left unpaired this cycle. The draw is a symmetric proposal, so
    detailed balance holds per pair.

    Args:
        n_walkers_lo: walker count in the lower-index window.
        n_walkers_hi: walker count in the upper-index window.
        rng: generator for the shuffles.

    Returns:
        List of ``(slot_lo, slot_hi)`` index pairs, disjoint in each
        coordinate. Length ``min(n_walkers_lo, n_walkers_hi)``.
    """
    lo = rng.permutation(n_walkers_lo)
    hi = rng.permutation(n_walkers_hi)
    k = min(n_walkers_lo, n_walkers_hi)
    return [(int(a), int(b)) for a, b in zip(lo[:k], hi[:k], strict=True)]


def metropolis_accept(log_prob_ratio: float, rng: np.random.Generator) -> bool:
    """Apply the Metropolis criterion to an exchange proposal.

    Accept with probability ``min(1, exp(log_prob_ratio))``.

    Args:
        log_prob_ratio: log of the acceptance ratio for the proposed
            exchange.
        rng: numpy random generator used to draw the acceptance
            uniform.

    Returns:
        True if the exchange is accepted.
    """
    if log_prob_ratio >= 0.0:
        # Non-negative ratio accepts unconditionally; short-circuit here
        # so `np.exp` never sees a large positive input that would overflow.
        return True
    return bool(rng.random() < float(np.exp(log_prob_ratio)))
