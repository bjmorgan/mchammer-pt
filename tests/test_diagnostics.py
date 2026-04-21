"""Tests for PT diagnostics: round-trip counting and autocorrelation."""

from __future__ import annotations

import numpy as np

from mchammer_pt.diagnostics import (
    energy_autocorrelation_time,
    round_trip_counts,
    swap_acceptance_rates,
)
from mchammer_pt.history import ExchangeHistory


def test_round_trip_counts_zero_for_constant_labels():
    labels = np.tile(np.arange(3), (10, 1))  # no motion
    counts = round_trip_counts(labels)
    assert counts.shape == (3,)
    np.testing.assert_array_equal(counts, [0, 0, 0])


def test_round_trip_counts_single_traversal_counts_one():
    # 3 replicas. Label 0 starts at T-index 0, walks up to T-index 2,
    # and back to T-index 0 -> one round trip for label 0.
    labels = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
            [2, 0, 1],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    counts = round_trip_counts(labels)
    # Only label 0 completes a full traversal in this trace.
    assert counts[0] == 1


def test_round_trip_counts_two_traversals():
    """Two complete bottom-top-bottom cycles for label 0 count as two trips."""
    # Label 0's temperature index (first column reads: the label AT T=0).
    # Schedule is: 0 at bottom -> 0 at top -> 0 at bottom -> 0 at top -> 0 at bottom.
    labels = np.array(
        [
            [0, 1, 2],  # label 0 at bottom
            [1, 2, 0],  # label 0 at top
            [0, 2, 1],  # label 0 at bottom (one round trip complete)
            [1, 2, 0],  # label 0 at top
            [0, 1, 2],  # label 0 at bottom (second round trip complete)
        ],
        dtype=np.int64,
    )
    counts = round_trip_counts(labels)
    assert counts[0] == 2


def test_round_trip_counts_partial_traversal_does_not_count():
    """A bottom-to-top move with no return does not count as a round trip."""
    labels = np.array(
        [
            [0, 1, 2],  # label 0 at bottom
            [1, 0, 2],  # label 0 in middle
            [1, 2, 0],  # label 0 at top (half a trip)
            [1, 2, 0],  # stays at top
            [2, 1, 0],  # still at top
        ],
        dtype=np.int64,
    )
    counts = round_trip_counts(labels)
    # No label returns to bottom after reaching top, so no round trips.
    assert counts[0] == 0


def test_swap_acceptance_rates_from_history():
    h = ExchangeHistory.empty(n_cycles=5, n_replicas=4)
    h.swap_attempted[:] = [10, 0, 20]
    h.swap_accepted[:] = [3, 0, 10]
    rates = swap_acceptance_rates(h)
    assert rates[0] == 0.3
    assert np.isnan(rates[1])
    assert rates[2] == 0.5


def test_energy_autocorrelation_time_uncorrelated_near_one():
    rng = np.random.default_rng(0)
    # Draw IID samples; integrated autocorrelation time is ~1.
    energies = rng.normal(size=10_000)
    tau = energy_autocorrelation_time(energies)
    assert 0.5 < tau < 2.5


def test_energy_autocorrelation_time_detects_strong_correlation():
    # AR(1) with rho = 0.95 has integrated autocorrelation time
    # tau = (1 + rho) / (1 - rho) ~= 39.
    rng = np.random.default_rng(0)
    n = 50_000
    rho = 0.95
    x = np.zeros(n)
    eps = rng.normal(size=n) * np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    tau = energy_autocorrelation_time(x)
    # Very generous tolerance: the emcee-style window estimator is noisy.
    assert tau > 20
