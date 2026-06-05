"""Tests for PT pair-set rotation and Metropolis acceptance arithmetic."""

from __future__ import annotations

import numpy as np

from mchammer_pt.exchange import matching_for_boundary, metropolis_accept, pair_set_for_cycle


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


def test_matching_single_walker_is_one_pair():
    rng = np.random.default_rng(0)
    assert matching_for_boundary(1, 1, rng) == [(0, 0)]


def test_matching_equal_counts_is_a_perfect_matching():
    rng = np.random.default_rng(1)
    pairs = matching_for_boundary(3, 3, rng)
    assert len(pairs) == 3
    assert sorted(a for a, _ in pairs) == [0, 1, 2]
    assert sorted(b for _, b in pairs) == [0, 1, 2]


def test_matching_unequal_counts_covers_min_and_leaves_rest():
    rng = np.random.default_rng(2)
    pairs = matching_for_boundary(2, 5, rng)
    assert len(pairs) == 2
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    # The smaller (lo) window is fully covered; the larger (hi) window
    # contributes a disjoint *subset*, so we assert coverage on lo but
    # only distinctness and range on hi.
    assert sorted(left) == [0, 1]
    assert len(set(right)) == 2
    assert all(0 <= b < 5 for b in right)


def test_matching_is_reproducible_under_seed():
    assert matching_for_boundary(4, 4, np.random.default_rng(7)) == \
        matching_for_boundary(4, 4, np.random.default_rng(7))


def test_matching_pairs_are_disjoint():
    rng = np.random.default_rng(3)
    pairs = matching_for_boundary(4, 4, rng)
    assert len({a for a, _ in pairs}) == len(pairs)
    assert len({b for _, b in pairs}) == len(pairs)
