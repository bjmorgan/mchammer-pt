"""Helpers for parallel-tempering exchange proposals.

Two responsibilities, kept separate from the orchestrator so they can
be unit-tested in isolation:

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
    practice and ensures each pair is proposed every other cycle on
    average.

    Args:
        n_replicas: total number of replicas.
        cycle: zero-based cycle index.

    Returns:
        1-D integer array of pair indices in ascending order.
    """
    start = cycle % 2
    return np.arange(start, n_replicas - 1, 2)


def metropolis_accept(log_prob_ratio: float, rng: np.random.Generator) -> bool:
    """Apply the Metropolis criterion to an exchange proposal.

    Accept with probability ``min(1, exp(log_prob_ratio))``. A
    non-negative ratio accepts unconditionally (avoiding an `np.exp`
    overflow on large positive inputs).

    Args:
        log_prob_ratio: log of the acceptance ratio for the proposed
            exchange.
        rng: numpy random generator used to draw the acceptance
            uniform.

    Returns:
        True if the exchange is accepted.
    """
    if log_prob_ratio >= 0.0:
        return True
    return bool(rng.random() < float(np.exp(log_prob_ratio)))
