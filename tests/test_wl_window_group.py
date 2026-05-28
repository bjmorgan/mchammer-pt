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


def test_merge_entropies_aligns_constants_via_intersection_mean():
    """Two walkers with same shape offset by different constants merge cleanly."""
    from mchammer_pt.wl_coordinator import merge_entropies

    # Both walkers visit bins {0, 1, 2}. Walker A's S = m(E) + 10;
    # walker B's S = m(E) + 50, where m = {0: 0.0, 1: 5.0, 2: 10.0}.
    a = {0: 10.0, 1: 15.0, 2: 20.0}
    b = {0: 50.0, 1: 55.0, 2: 60.0}
    merged = merge_entropies([a, b])

    # After intersection-mean rebasing, walker A becomes {0: -5, 1: 0, 2: 5}
    # and walker B becomes {0: -5, 1: 0, 2: 5}. Average = {0: -5, 1: 0, 2: 5}.
    # Post-shift so min=0: {0: 0, 1: 5, 2: 10}.
    assert merged[0] == pytest.approx(0.0)
    assert merged[1] == pytest.approx(5.0)
    assert merged[2] == pytest.approx(10.0)


def test_merge_entropies_no_coverage_boundary_distortion():
    """Partial coverage does not introduce shape artefacts at coverage edges."""
    from mchammer_pt.wl_coordinator import merge_entropies

    # Walker A visits {0, 1, 2}; walker B visits {1, 2, 3}. Both share
    # m(E) = E + 0.0 on their visited bins (zero additive constants).
    a = {0: 0.0, 1: 1.0, 2: 2.0}
    b = {1: 1.0, 2: 2.0, 3: 3.0}
    merged = merge_entropies([a, b])

    # Common bins = {1, 2}. A's mean over common = 1.5; B's mean = 1.5.
    # Rebased A = {0: -1.5, 1: -0.5, 2: 0.5}
    # Rebased B = {1: -0.5, 2: 0.5, 3: 1.5}
    # Bin-wise average: {0: -1.5, 1: -0.5, 2: 0.5, 3: 1.5}.
    # Post-shift so min=0: {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}.
    # Shape is preserved across the coverage boundary at bins 0 and 3.
    assert merged[0] == pytest.approx(0.0)
    assert merged[1] == pytest.approx(1.0)
    assert merged[2] == pytest.approx(2.0)
    assert merged[3] == pytest.approx(3.0)


def test_merge_entropies_partial_coverage_uses_visiting_walkers_only():
    """Bin visited only by one walker takes that walker's rebased value."""
    from mchammer_pt.wl_coordinator import merge_entropies

    # Common = {1, 2}. Bin 0 is in A only; bin 3 is in B only.
    a = {0: 0.0, 1: 1.0, 2: 2.0}
    b = {1: 11.0, 2: 12.0, 3: 13.0}
    merged = merge_entropies([a, b])

    # Walker A mean over common = 1.5; rebased A = {0: -1.5, 1: -0.5, 2: 0.5}.
    # Walker B mean over common = 11.5; rebased B = {1: -0.5, 2: 0.5, 3: 1.5}.
    # Bin 0: only A contributes -> -1.5. Bin 3: only B contributes -> 1.5.
    # Post-shift so min=0: {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}.
    assert merged[0] == pytest.approx(0.0)
    assert merged[1] == pytest.approx(1.0)
    assert merged[2] == pytest.approx(2.0)
    assert merged[3] == pytest.approx(3.0)


def test_merge_entropies_filters_unentered_walkers():
    """Walker with empty entropy dict is excluded from the merge."""
    from mchammer_pt.wl_coordinator import merge_entropies

    a = {0: 0.0, 1: 1.0}
    empty: dict[int, float] = {}
    merged = merge_entropies([a, empty])

    # Only walker A contributes; result equals a min-shifted to 0.
    assert merged == {0: 0.0, 1: 1.0}


def test_merge_entropies_single_walker_fast_path():
    """One walker after filtering returns its dict shifted so min=0."""
    from mchammer_pt.wl_coordinator import merge_entropies

    a = {0: 7.0, 1: 10.0, 2: 8.0}
    merged = merge_entropies([a])

    # Single walker: subtract its own min (7.0).
    assert merged == {0: 0.0, 1: 3.0, 2: 1.0}


def test_merge_entropies_no_walkers_returns_empty():
    """All walkers empty (or zero walkers) returns empty dict."""
    from mchammer_pt.wl_coordinator import merge_entropies

    assert merge_entropies([]) == {}
    assert merge_entropies([{}, {}]) == {}


def test_merge_entropies_empty_intersection_raises():
    """Walkers with no shared bin cannot be rebased; raise RuntimeError."""
    from mchammer_pt.wl_coordinator import merge_entropies

    a = {0: 0.0, 1: 1.0}
    b = {2: 2.0, 3: 3.0}
    with pytest.raises(RuntimeError, match="no common bin"):
        merge_entropies([a, b])


def test_merge_entropies_min_value_is_zero():
    """Result always satisfies the icet convention min(merged) == 0."""
    from mchammer_pt.wl_coordinator import merge_entropies

    a = {0: 100.0, 1: 105.0, 2: 102.0}
    b = {0: 200.0, 1: 205.0, 2: 202.0}
    merged = merge_entropies([a, b])

    assert min(merged.values()) == pytest.approx(0.0)



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


def test_mismatched_schedules_raises():
    """Replicas with different WL schedules cannot form a group."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    common_kwargs = dict(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
    )
    r0 = WangLandauReplica(
        **common_kwargs, random_seed=0,
        ensemble_kwargs={"schedule": "halving"},
    )
    r1 = WangLandauReplica(
        **common_kwargs, random_seed=1,
        ensemble_kwargs={"schedule": "1_over_t"},
    )
    with pytest.raises(ValueError, match="same schedule"):
        WangLandauWindowGroup([r0, r1], random_seed=0)


def test_mismatched_flatness_limits_raises():
    """Replicas with different flatness_limits cannot form a group."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    common_kwargs = dict(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
    )
    r0 = WangLandauReplica(
        **common_kwargs, random_seed=0,
        ensemble_kwargs={"flatness_limit": 0.8},
    )
    r1 = WangLandauReplica(
        **common_kwargs, random_seed=1,
        ensemble_kwargs={"flatness_limit": 0.5},
    )
    with pytest.raises(ValueError, match="same flatness_limit"):
        WangLandauWindowGroup([r0, r1], random_seed=0)


def test_window_group_snapshot_returns_per_walker_and_group_dicts():
    """snapshot_for_checkpoint returns a dict with per_walker
    (list of len W) and group (dict with rng_state, exchange_idx, phase)."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    group = WangLandauWindowGroup(_make_replicas(2), random_seed=0)
    snap = group.snapshot_for_checkpoint()

    assert set(snap.keys()) == {"per_walker", "group_state"}
    assert len(snap["per_walker"]) == 2
    for entry in snap["per_walker"]:
        assert "sites_by_species" in entry
    assert set(snap["group_state"].keys()) == {"rng_state", "exchange_idx", "phase"}
    assert isinstance(snap["group_state"]["rng_state"], str)
    assert isinstance(snap["group_state"]["exchange_idx"], int)
    assert snap["group_state"]["phase"] in {"halving", "1_over_t"}


def test_attach_observer_factory_type_check():
    """attach_observer_factory raises TypeError if factory returns non-BaseObserver."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    group = WangLandauWindowGroup(_make_replicas(2), random_seed=0)
    with pytest.raises(TypeError, match="not a BaseObserver"):
        group.attach_observer_factory(lambda replica: "not an observer")


def test_decide_bp_switch_all_eligible_returns_true():
    """All walkers with 1/t > f → switch."""
    from mchammer_pt.wl_coordinator import decide_bp_switch

    # t/f such that 1/t > f for both.
    assert decide_bp_switch(ts=[100, 100], fs=[0.005, 0.005]) is True


def test_decide_bp_switch_any_walker_below_threshold_returns_false():
    """One walker with 1/t <= f → no switch."""
    from mchammer_pt.wl_coordinator import decide_bp_switch

    # walker 1: 1/100 = 0.01, f = 0.02 -> 1/t < f
    assert decide_bp_switch(ts=[100, 100], fs=[0.005, 0.02]) is False


def test_halving_criterion_met_all_walkers_flat_returns_true():
    """Group's halving_criterion_met returns True iff every walker meets it."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._histogram = {0: 1000, 1: 1000}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    assert group.halving_criterion_met() is True


def test_halving_criterion_met_one_walker_not_flat_returns_false():
    """Group's halving_criterion_met returns False if any walker fails it."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[1].ensemble._reached_energy_window = True
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}  # not flat

    group = WangLandauWindowGroup(replicas, random_seed=0)
    assert group.halving_criterion_met() is False











def test_validate_flatness_mode_accepts_known_values():
    from mchammer_pt.wl_coordinator import _validate_flatness_mode

    _validate_flatness_mode("per_walker")
    _validate_flatness_mode("pooled")


def test_validate_flatness_mode_rejects_unknown_value():
    from mchammer_pt.wl_coordinator import _validate_flatness_mode

    with pytest.raises(ValueError, match="flatness_mode must be one of"):
        _validate_flatness_mode("bogus")


def test_validate_merge_cadence_accepts_known_values():
    from mchammer_pt.wl_coordinator import _validate_merge_cadence

    _validate_merge_cadence("at_halve")
    _validate_merge_cadence("never")


def test_validate_merge_cadence_rejects_unknown_value():
    from mchammer_pt.wl_coordinator import _validate_merge_cadence

    with pytest.raises(ValueError, match="merge_cadence must be one of"):
        _validate_merge_cadence("bogus")


def test_summed_histogram_halving_criterion_met_pooled_case():
    """Pooled flatness from snapshots: pooling fixes per-walker gaps."""
    from mchammer_pt.wl_coordinator import (
        WalkerPostBlockState,
        _summed_histogram_halving_criterion_met,
    )

    # Walker A has skewed coverage; walker B has the complementary skew.
    # Per-walker: each would fail the flatness test independently.
    # Pooled: combined histogram is even, passes the test.
    snapshots = [
        WalkerPostBlockState(
            halving_criterion_met=False,
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=0,
            histogram={0: 100, 1: 1000},
            reached_energy_window=True,
        ),
        WalkerPostBlockState(
            halving_criterion_met=False,
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=0,
            histogram={0: 1000, 1: 100},
            reached_energy_window=True,
        ),
    ]
    # Pooled: {0: 1100, 1: 1100}, mean 1100, limit 880 (0.8 * 1100);
    # both >= 880 -> True.
    assert (
        _summed_histogram_halving_criterion_met(snapshots, 0.8, schedule="halving")
        is True
    )


def test_summed_histogram_halving_criterion_met_false_when_unentered():
    """Pooled flatness from snapshots returns False if any walker has not entered."""
    from mchammer_pt.wl_coordinator import (
        WalkerPostBlockState,
        _summed_histogram_halving_criterion_met,
    )

    snapshots = [
        WalkerPostBlockState(
            halving_criterion_met=True,
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=0,
            histogram={0: 1000, 1: 1000},
            reached_energy_window=True,
        ),
        WalkerPostBlockState(
            halving_criterion_met=False,
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=None,
            histogram={},
            reached_energy_window=False,
        ),
    ]
    assert (
        _summed_histogram_halving_criterion_met(snapshots, 0.8, schedule="halving")
        is False
    )



def test_advance_refreshes_walker_states():
    """advance() populates walker_states from live ensemble state."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    group = WangLandauWindowGroup(replicas, random_seed=0)

    # Before any advance, walker_states should hold the zero-initialised values.
    for state in group.walker_states:
        assert state.step == 0

    group.advance(10)

    # After advance, walker_states entries must reflect post-advance ensemble state.
    for i, state in enumerate(group.walker_states):
        e = replicas[i].ensemble
        assert state.step == e.step
        assert state.fill_factor == pytest.approx(float(e._fill_factor))
        assert state.reached_energy_window == bool(e._reached_energy_window)
        assert state.histogram == dict(e._histogram)


def test_advance_walker_states_length_matches_replicas():
    """walker_states has one entry per replica after advance."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(3)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    group.advance(5)
    assert len(group.walker_states) == 3


def test_apply_plan_halve_only():
    """apply_plan with halve=True halves fill factors and resets histograms."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}
        r.ensemble._histogram = {0: 500, 1: 600}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    plan = CoordinatorPlan(halve=True, merged_entropy=None, switch_to_phase=None)
    group.apply_plan(plan)

    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)
        assert all(v == 0 for v in r.ensemble._histogram.values())


def test_apply_plan_no_halve_is_noop():
    """apply_plan with halve=False leaves fill factors unchanged."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._fill_factor = 1.0

    group = WangLandauWindowGroup(replicas, random_seed=0)
    plan = CoordinatorPlan(halve=False, merged_entropy=None, switch_to_phase=None)
    group.apply_plan(plan)

    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(1.0)


def test_apply_plan_writes_merged_entropy():
    """apply_plan with merged_entropy writes the dict to every walker."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}
        r.ensemble._histogram = {0: 1000, 1: 1000}
    replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}
    replicas[1].ensemble._entropy = {0: 3.0, 1: 4.0}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    merged = {0: 0.0, 1: 5.0}
    plan = CoordinatorPlan(halve=True, merged_entropy=merged, switch_to_phase=None)
    group.apply_plan(plan)

    for r in replicas:
        assert r.ensemble._entropy == {0: 0.0, 1: 5.0}


def test_apply_plan_switches_to_one_over_t():
    """apply_plan with switch_to_phase sets phase and updates fill factors."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 0.001
        r.ensemble._fill_factor_history = {}
        r.ensemble._histogram = {0: 1000, 1: 1000}
        r.ensemble._phase = "halving"
        r.ensemble._step = 100
        r.ensemble._window_entry_step = 0

    group = WangLandauWindowGroup(replicas, random_seed=0)
    plan = CoordinatorPlan(halve=True, merged_entropy=None, switch_to_phase="1_over_t")
    group.apply_plan(plan)

    for r in replicas:
        assert r.ensemble._phase == "1_over_t"
        # t = step - entry + 1 = 100 - 0 + 1 = 101
        assert r.ensemble._fill_factor == pytest.approx(1.0 / 101)


def test_apply_plan_phase_switch_does_not_change_fill_factor_history():
    """Phase switch via apply_plan does not add to fill_factor_history."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 0.001
        r.ensemble._fill_factor_history = {100: 0.001}
        r.ensemble._histogram = {0: 1000, 1: 1000}
        r.ensemble._phase = "halving"
        r.ensemble._step = 100
        r.ensemble._window_entry_step = 0

    group = WangLandauWindowGroup(replicas, random_seed=0)
    plan = CoordinatorPlan(halve=False, merged_entropy=None, switch_to_phase="1_over_t")
    group.apply_plan(plan)

    for r in replicas:
        # History must be unchanged; the BP switch is not a halve.
        assert r.ensemble._fill_factor_history == {100: 0.001}


def test_finalise_for_reporting_merges_into_all_walkers():
    """finalise_for_reporting merges entropies and writes to every walker."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    # Identical shape, different additive constants.
    replicas[0].ensemble._entropy = {0: 0.0, 1: 5.0}
    replicas[1].ensemble._entropy = {0: 10.0, 1: 15.0}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group.finalise_for_reporting()

    # Both walkers receive the merged, min-shifted dict.
    for r in replicas:
        assert r.ensemble._entropy[0] == pytest.approx(0.0)
        assert r.ensemble._entropy[1] == pytest.approx(5.0)


def test_per_window_stats_injects_flatness_mode_from_pool():
    """Pool.per_window_stats injects flatness_mode; group.window_stats omits it."""
    from mchammer_pt.parallel.serial import SerialWangLandauPool
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 500, 1: 1000}  # flat_min = 500/750 ~ 0.667
    replicas[1].ensemble._histogram = {0: 900, 1: 1000}  # flat_min = 900/950 ~ 0.947

    group = WangLandauWindowGroup(replicas, random_seed=0)

    # Group-level stats do not include flatness_mode.
    group_stats = group.window_stats()
    assert "flatness_mode" not in group_stats

    # Pool injects it from its own config.
    pool = SerialWangLandauPool(
        [group],
        energy_spacing=0.1,
        flatness_mode="per_walker",
    )
    pool_stats = pool.per_window_stats()
    assert pool_stats[0]["flatness_mode"] == "per_walker"
    # per_walker_flat_min is still computed by the group and surfaced via the pool.
    # walker 0: 500 / 750 = 0.667 (smaller)
    # walker 1: 900 / 950 = 0.947
    assert pool_stats[0]["per_walker_flat_min"] == pytest.approx(500 / 750)


def test_window_stats_per_walker_flat_min_none_when_walker_has_no_histogram():
    """per_walker_flat_min is None if any walker's histogram is empty."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[1].ensemble._reached_energy_window = False
    replicas[1].ensemble._histogram = {}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    stats = group.window_stats()

    assert stats["per_walker_flat_min"] is None


def test_window_stats_reports_phase():
    """Multi-walker group exposes phase from the first replica.

    All walkers in a group share the same phase (enforced by
    apply_plan), so reading from replica 0 is canonical.
    """
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    group = WangLandauWindowGroup(replicas, random_seed=0)
    assert group.window_stats()["phase"] == "halving"
    for r in replicas:
        r.ensemble._phase = "1_over_t"
    assert group.window_stats()["phase"] == "1_over_t"


def test_window_stats_reports_bins_visited_and_bins_known_combined():
    """For a window group, bins_visited is the union of all walkers'
    _visited_bins; bins_known is len(combined_histogram).
    """
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    # Walker 0: has been at bins 0 and 1.
    replicas[0].ensemble._visited_bins = {0, 1}
    replicas[0].ensemble._histogram = {0: 0, 1: 100}
    replicas[0].ensemble._reached_energy_window = True
    # Walker 1: has been at bins 1 and 2.
    replicas[1].ensemble._visited_bins = {1, 2}
    replicas[1].ensemble._histogram = {1: 100, 2: 50}
    replicas[1].ensemble._reached_energy_window = True

    group = WangLandauWindowGroup(replicas, random_seed=0)
    stats = group.window_stats()

    # Union of _visited_bins across walkers: {0, 1, 2} → 3.
    # Combined histogram keys: {0, 1, 2} → 3.
    assert stats["bins_visited"] == 3
    assert stats["bins_known"] == 3


def test_finalise_for_reporting_idempotent():
    """Calling finalise twice produces the same state as calling it once."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    replicas[0].ensemble._entropy = {0: 0.0, 1: 5.0}
    replicas[1].ensemble._entropy = {0: 10.0, 1: 15.0}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group.finalise_for_reporting()
    state_after_first = [dict(r.ensemble._entropy) for r in replicas]

    group.finalise_for_reporting()
    state_after_second = [dict(r.ensemble._entropy) for r in replicas]

    assert state_after_second == state_after_first


def test_window_group_restore_state_round_trips_exchange_rng():
    """After snapshot -> mutate -> restore, the next exchange index
    selection produces the same draw as the unmutated group would."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    group_a = WangLandauWindowGroup(_make_replicas(2), random_seed=0)
    group_b = WangLandauWindowGroup(_make_replicas(2), random_seed=0)

    # Drift group_b's exchange RNG so it diverges from group_a.
    for _ in range(5):
        group_b.reroll_exchange_idx()

    snap = group_a.snapshot_for_checkpoint()
    group_b.restore_state(
        containers=[r.data_container() for r in group_a._replicas],
        per_walker_extras=snap["per_walker"],
        group_state=snap["group_state"],
    )
    # Both groups must now produce identical next exchange indices.
    group_a.reroll_exchange_idx()
    group_b.reroll_exchange_idx()
    assert group_a.exchange_idx == group_b.exchange_idx


def test_window_group_restore_state_rejects_wrong_length_containers():
    """restore_state validates containers length matches walker count."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    group = WangLandauWindowGroup(_make_replicas(2), random_seed=0)
    snap = group.snapshot_for_checkpoint()
    with pytest.raises(ValueError, match="restore_state expects 2 containers"):
        group.restore_state(
            containers=[],
            per_walker_extras=snap["per_walker"],
            group_state=snap["group_state"],
        )
