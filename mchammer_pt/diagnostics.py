"""PT diagnostics: round-trip counting, autocorrelation, swap rates.

Pure functions operating on `ExchangeHistory` (or its constituent
arrays). They do not touch the orchestrator, are trivially unit-
testable, and form the basis for any user-built quality checks.
"""
from __future__ import annotations

import numpy as np

from .history import ExchangeHistory


def round_trip_counts(replica_labels_per_cycle: np.ndarray) -> np.ndarray:
    """Per-label count of full ladder traversals.

    A round trip is a visit to the top rung followed by a subsequent
    visit to the bottom rung (or vice versa). Only the number of
    lowest-to-highest-and-back-again trips is counted — partial
    traversals do not count.

    Args:
        replica_labels_per_cycle: shape ``(n_cycles+1, n_replicas)``,
            the integer label at each temperature index per cycle.

    Returns:
        1-D array of shape ``(n_replicas,)`` giving the round-trip
        count for each label.
    """
    labels = np.asarray(replica_labels_per_cycle)
    n_cycles_plus_one, n_replicas = labels.shape
    counts = np.zeros(n_replicas, dtype=np.int64)
    top = n_replicas - 1
    # For each label, track its current "direction" state:
    #   0 = neither endpoint yet visited,
    #   1 = most recent endpoint visit was the bottom,
    #   2 = most recent endpoint visit was the top.
    # A round trip is counted each time the state transitions 1 -> 2 -> 1
    # or 2 -> 1 -> 2.
    state = np.zeros(n_replicas, dtype=np.int8)
    for cycle in range(n_cycles_plus_one):
        row = labels[cycle]
        for t_index in range(n_replicas):
            label = int(row[t_index])
            if t_index == 0:
                new_state = 1
            elif t_index == top:
                new_state = 2
            else:
                continue
            if state[label] != 0 and state[label] != new_state:
                counts[label] += 1
            state[label] = new_state
    # A full round trip is two endpoint transitions (e.g. bottom -> top ->
    # bottom) which we counted as two transitions. Halve to get trips.
    return counts // 2


def swap_acceptance_rates(history: ExchangeHistory) -> np.ndarray:
    """Per-pair acceptance fractions, NaN where no attempts were made."""
    attempts = history.swap_attempted.astype(np.float64)
    accepts = history.swap_accepted.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(attempts > 0, accepts / np.maximum(attempts, 1), np.nan)
    return rates


def energy_autocorrelation_time(energies: np.ndarray) -> float:
    """Integrated autocorrelation time of a 1-D energy trace.

    Uses the standard windowed estimator with Sokal's automatic window
    selection (``M = min_M { M >= c * tau(M) }``, c = 5). Returns a
    float; for IID data the estimate is ~1.
    """
    x = np.asarray(energies, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return float("nan")
    x = x - x.mean()
    # Autocovariance via FFT for speed.
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    if acf[0] == 0.0:
        return float("nan")
    acf = acf / acf[0]

    tau = 1.0
    c = 5.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * float(np.sum(acf[1 : m + 1]))
        if m >= c * tau:
            break
    return tau
