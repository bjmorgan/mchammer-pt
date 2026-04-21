"""Tests for PT pair-set rotation and Metropolis acceptance arithmetic."""

from __future__ import annotations

import numpy as np

from mchammer_pt.exchange import metropolis_accept, pair_set_for_cycle


def test_pair_set_even_cycle_returns_even_indices():
    # 6 replicas, 5 pairs (0..4). Even cycle should attempt pairs 0, 2, 4.
    result = pair_set_for_cycle(n_replicas=6, cycle=0)
    assert result.tolist() == [0, 2, 4]


def test_pair_set_odd_cycle_returns_odd_indices():
    # 6 replicas. Odd cycle should attempt pairs 1, 3.
    result = pair_set_for_cycle(n_replicas=6, cycle=1)
    assert result.tolist() == [1, 3]


def test_pair_set_alternates_even_odd_over_cycles():
    even = pair_set_for_cycle(n_replicas=5, cycle=2)  # still even
    odd = pair_set_for_cycle(n_replicas=5, cycle=3)  # odd
    assert even.tolist() == [0, 2]
    assert odd.tolist() == [1, 3]


def test_pair_set_two_replicas():
    # Only pair 0 exists; even cycles include it, odd cycles skip it.
    assert pair_set_for_cycle(2, 0).tolist() == [0]
    assert pair_set_for_cycle(2, 1).tolist() == []


def test_metropolis_accept_nonnegative_log_ratio_always_accepts():
    rng = np.random.default_rng(0)
    # With log_prob_ratio >= 0 the acceptance probability is 1.
    for _ in range(100):
        assert metropolis_accept(log_prob_ratio=0.5, rng=rng)
        assert metropolis_accept(log_prob_ratio=0.0, rng=rng)


def test_metropolis_accept_very_negative_log_ratio_rejects():
    rng = np.random.default_rng(0)
    # log_prob_ratio = -50 => acceptance prob ~ exp(-50) ~ 1e-22;
    # no rng draw will be smaller than this in 1000 tries.
    for _ in range(1000):
        assert not metropolis_accept(log_prob_ratio=-50.0, rng=rng)


def test_metropolis_accept_statistics_match_target_probability():
    # With log_prob_ratio = log(0.3), acceptance rate should converge to 0.3.
    rng = np.random.default_rng(0)
    trials = 20_000
    accepts = sum(metropolis_accept(np.log(0.3), rng) for _ in range(trials))
    assert abs(accepts / trials - 0.3) < 0.02
