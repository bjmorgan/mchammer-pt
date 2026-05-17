"""Unit tests for WangLandauWindowGroup."""

from __future__ import annotations

import numpy as np
import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _make_replicas(n: int = 2):
    """N WangLandauReplica instances sharing a wide window."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    return [
        WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=0.1,
            energy_limit_left=e0 - 100.0,
            energy_limit_right=e0 + 100.0,
            random_seed=i,
        )
        for i in range(n)
    ]


def test_merge_entropies_averages_all_bins():
    """Entropy is averaged bin-wise; unvisited bins treat as 0.0."""
    from mchammer_pt.wl_window_group import merge_entropies

    merged = merge_entropies([{0: 2.0, 1: 4.0}, {0: 6.0, 2: 8.0}])

    # bin 0: (2.0 + 6.0) / 2 = 4.0
    # bin 1: (4.0 + 0.0) / 2 = 2.0  (replica 1 never visited bin 1)
    # bin 2: (0.0 + 8.0) / 2 = 4.0  (replica 0 never visited bin 2)
    assert merged[0] == pytest.approx(4.0)
    assert merged[1] == pytest.approx(2.0)
    assert merged[2] == pytest.approx(4.0)


def test_merge_entropies_noop_on_empty_entropies():
    """No KeyError or crash when all entropy dicts are empty."""
    from mchammer_pt.wl_window_group import merge_entropies

    assert merge_entropies([{}, {}]) == {}


def test_advance_calls_all_replicas_and_updates_exchange_idx():
    """advance() advances each replica then runs the coordinator block."""
    from unittest.mock import MagicMock

    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    for r in replicas:
        r.advance = MagicMock()
    group._run_coordinator_block = MagicMock()

    group.advance(100)

    for r in replicas:
        r.advance.assert_called_once_with(100)
    group._run_coordinator_block.assert_called_once()


def test_exchange_methods_target_same_replica():
    """current_energy, current_occupations, set_occupations all target _exchange_idx."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._exchange_idx = 1

    assert group.current_energy() == replicas[1].current_energy()
    assert np.array_equal(
        group.current_occupations(), replicas[1].current_occupations()
    )


def test_log_g_returns_merged_value():
    """log_g delegates to replica 0 (all identical after merge)."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    e = replicas[0].current_energy()
    assert group.log_g(e) == replicas[0].log_g(e)


def test_converged_requires_all_replicas():
    """converged is True only when every replica is converged."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    assert not group.converged

    # Simulate convergence by setting mchammer's internal convergence flag.
    for r in replicas:
        r.ensemble._converged = True
    assert group.converged


def test_all_data_containers_returns_w_containers():
    """all_data_containers() returns one container per walker."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(3)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    containers = group.all_data_containers()
    assert len(containers) == 3


def test_mismatched_windows_raises():
    """Replicas with different energy windows cannot form a group."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    r0 = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
    )
    r1 = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 50.0,  # different window
        energy_limit_right=e0 + 50.0,
        random_seed=1,
    )
    with pytest.raises(ValueError, match="energy window"):
        WangLandauWindowGroup([r0, r1], random_seed=0)


def test_snapshot_for_checkpoint_raises():
    """snapshot_for_checkpoint raises NotImplementedError."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    group = WangLandauWindowGroup(_make_replicas(2), random_seed=0)
    with pytest.raises(NotImplementedError, match="checkpointing"):
        group.snapshot_for_checkpoint()


def test_attach_observer_factory_type_check():
    """attach_observer_factory raises TypeError if factory returns non-BaseObserver."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    group = WangLandauWindowGroup(_make_replicas(2), random_seed=0)
    with pytest.raises(TypeError, match="not a BaseObserver"):
        group.attach_observer_factory(lambda replica: "not an observer")


def test_decide_collective_halve_all_flat_returns_true():
    """All walkers flat → collective halve fires."""
    from mchammer_pt.wl_window_group import decide_collective_halve

    assert decide_collective_halve([True, True], policy="block") is True
    assert decide_collective_halve([True, True], policy="halving") is True


def test_decide_collective_halve_some_not_flat_returns_false():
    """Any walker not flat → no halve."""
    from mchammer_pt.wl_window_group import decide_collective_halve

    assert decide_collective_halve([True, False], policy="block") is False
    assert decide_collective_halve([True, False], policy="halving") is False
    assert decide_collective_halve([False, False], policy="block") is False


def test_decide_collective_halve_empty_returns_false():
    """Empty flag list: degenerate case, no halve."""
    from mchammer_pt.wl_window_group import decide_collective_halve

    assert decide_collective_halve([], policy="block") is False


def test_decide_bp_switch_all_eligible_returns_true():
    """All walkers in halving phase with 1/t > f → switch."""
    from mchammer_pt.wl_window_group import decide_bp_switch

    # phases all "halving", t/f such that 1/t > f for both.
    assert (
        decide_bp_switch(
            phases=["halving", "halving"], ts=[100, 100], fs=[0.005, 0.005]
        )
        is True
    )


def test_decide_bp_switch_any_walker_below_threshold_returns_false():
    """One walker with 1/t <= f → no switch."""
    from mchammer_pt.wl_window_group import decide_bp_switch

    # walker 1: 1/100 = 0.01, f = 0.02 -> 1/t < f
    assert (
        decide_bp_switch(
            phases=["halving", "halving"], ts=[100, 100], fs=[0.005, 0.02]
        )
        is False
    )


def test_decide_bp_switch_already_switched_returns_false():
    """Any walker already in 1_over_t phase → no further switch."""
    from mchammer_pt.wl_window_group import decide_bp_switch

    assert (
        decide_bp_switch(
            phases=["halving", "1_over_t"],
            ts=[100, 100],
            fs=[0.001, 0.001],
        )
        is False
    )


def test_is_flat_all_walkers_flat_returns_true():
    """Group's is_flat returns True iff all walkers are flat."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._histogram = {0: 1000, 1: 1000}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    assert group.is_flat() is True


def test_is_flat_one_walker_not_flat_returns_false():
    """Group's is_flat returns False if any walker is not flat."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[1].ensemble._reached_energy_window = True
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}  # not flat

    group = WangLandauWindowGroup(replicas, random_seed=0)
    assert group.is_flat() is False


def test_advance_collective_halve_fires_when_all_flat():
    """All walkers flat after block → collective halve fires."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._histogram = {0: 1000, 1: 1000}
        r.ensemble._entropy = {0: 5.0, 1: 5.0}
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._run_coordinator_block()

    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)
        assert len(r.ensemble._fill_factor_history) == 1
        assert all(v == 0 for v in r.ensemble._histogram.values())


def test_advance_no_halve_when_one_walker_not_flat():
    """No halve when any walker fails flatness."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[0].ensemble._entropy = {0: 5.0, 1: 5.0}
    replicas[0].ensemble._fill_factor = 1.0
    replicas[1].ensemble._reached_energy_window = True
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}  # not flat
    replicas[1].ensemble._entropy = {0: 5.0, 1: 5.0}
    replicas[1].ensemble._fill_factor = 1.0

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._run_coordinator_block()

    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(1.0)


def test_advance_block_policy_merges_entropy_every_block():
    """sync_policy='block' merges entropy every block, even without halve."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 100, 1: 1000}  # not flat
    replicas[0].ensemble._entropy = {0: 2.0, 1: 4.0}
    replicas[1].ensemble._reached_energy_window = True
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}
    replicas[1].ensemble._entropy = {0: 6.0, 1: 8.0}

    group = WangLandauWindowGroup(replicas, random_seed=0, sync_policy="block")
    group._run_coordinator_block()

    # Merged: {0: 4.0, 1: 6.0}; both walkers receive this.
    for r in replicas:
        assert r.ensemble._entropy[0] == pytest.approx(4.0)
        assert r.ensemble._entropy[1] == pytest.approx(6.0)


def test_advance_halving_policy_skips_non_halving_merge():
    """sync_policy='halving' does not merge entropy when no halve fires."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 100, 1: 1000}  # not flat
    replicas[0].ensemble._entropy = {0: 2.0, 1: 4.0}
    replicas[1].ensemble._reached_energy_window = True
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}
    replicas[1].ensemble._entropy = {0: 6.0, 1: 8.0}

    group = WangLandauWindowGroup(
        replicas, random_seed=0, sync_policy="halving"
    )
    group._run_coordinator_block()

    # No merge; original per-walker values preserved.
    assert replicas[0].ensemble._entropy[0] == pytest.approx(2.0)
    assert replicas[1].ensemble._entropy[0] == pytest.approx(6.0)


def test_advance_halving_policy_merges_at_halving_event():
    """sync_policy='halving' merges entropy at collective halve."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._histogram = {0: 1000, 1: 1000}
    replicas[0].ensemble._entropy = {0: 2.0, 1: 4.0}
    replicas[1].ensemble._entropy = {0: 6.0, 1: 8.0}
    for r in replicas:
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}

    group = WangLandauWindowGroup(
        replicas, random_seed=0, sync_policy="halving"
    )
    group._run_coordinator_block()

    # Halve fires → merge fires.
    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)
        # Merged entropy distributed to all walkers AFTER force_halve.
        assert r.ensemble._entropy[0] == pytest.approx(4.0)
        assert r.ensemble._entropy[1] == pytest.approx(6.0)
