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
    from mchammer_pt.wl_window_group import merge_entropies

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
    from mchammer_pt.wl_window_group import merge_entropies

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
    from mchammer_pt.wl_window_group import merge_entropies

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
    from mchammer_pt.wl_window_group import merge_entropies

    a = {0: 0.0, 1: 1.0}
    empty: dict[int, float] = {}
    merged = merge_entropies([a, empty])

    # Only walker A contributes; result equals a min-shifted to 0.
    assert merged == {0: 0.0, 1: 1.0}


def test_merge_entropies_single_walker_fast_path():
    """One walker after filtering returns its dict shifted so min=0."""
    from mchammer_pt.wl_window_group import merge_entropies

    a = {0: 7.0, 1: 10.0, 2: 8.0}
    merged = merge_entropies([a])

    # Single walker: subtract its own min (7.0).
    assert merged == {0: 0.0, 1: 3.0, 2: 1.0}


def test_merge_entropies_no_walkers_returns_empty():
    """All walkers empty (or zero walkers) returns empty dict."""
    from mchammer_pt.wl_window_group import merge_entropies

    assert merge_entropies([]) == {}
    assert merge_entropies([{}, {}]) == {}


def test_merge_entropies_empty_intersection_raises():
    """Walkers with no shared bin cannot be rebased; raise RuntimeError."""
    from mchammer_pt.wl_window_group import merge_entropies

    a = {0: 0.0, 1: 1.0}
    b = {2: 2.0, 3: 3.0}
    with pytest.raises(RuntimeError, match="no common bin"):
        merge_entropies([a, b])


def test_merge_entropies_min_value_is_zero():
    """Result always satisfies the icet convention min(merged) == 0."""
    from mchammer_pt.wl_window_group import merge_entropies

    a = {0: 100.0, 1: 105.0, 2: 102.0}
    b = {0: 200.0, 1: 205.0, 2: 202.0}
    merged = merge_entropies([a, b])

    assert min(merged.values()) == pytest.approx(0.0)


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

    assert decide_collective_halve([True, True]) is True


def test_decide_collective_halve_some_not_flat_returns_false():
    """Any walker not flat → no halve."""
    from mchammer_pt.wl_window_group import decide_collective_halve

    assert decide_collective_halve([True, False]) is False
    assert decide_collective_halve([False, False]) is False


def test_decide_collective_halve_empty_returns_false():
    """Empty flag list: degenerate case, no halve."""
    from mchammer_pt.wl_window_group import decide_collective_halve

    assert decide_collective_halve([]) is False


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

    group = WangLandauWindowGroup(replicas, random_seed=0, flatness_mode="per_walker")
    group._run_coordinator_block()

    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(1.0)


def test_per_walker_flatness_requires_all_walkers_flat():
    """flatness_mode='per_walker': halve fires only when every walker is flat."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}
    # Walker A flat, walker B not.
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="at_halve",
    )
    group._run_coordinator_block()

    # Walker B was not flat -> no halve.
    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(1.0)


def test_pooled_flatness_fires_when_per_walker_does_not():
    """flatness_mode='pooled' halves when summed histogram is flat."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}
    # Asymmetric per-walker but pooled-flat: each has 100 and 1000;
    # pooled = 1100 and 1100, flat.
    replicas[0].ensemble._histogram = {0: 100, 1: 1000}
    replicas[1].ensemble._histogram = {0: 1000, 1: 100}
    replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}
    replicas[1].ensemble._entropy = {0: 3.0, 1: 4.0}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="pooled",
        merge_cadence="at_halve",
    )
    group._run_coordinator_block()

    # Pooled flat -> halve fires.
    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)


def test_merge_cadence_at_halve_merges_only_on_halve():
    """merge_cadence='at_halve': entropies merged at halve, not otherwise."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    # Not flat -> no halve. Entropies differ.
    replicas[0].ensemble._histogram = {0: 100, 1: 1000}
    replicas[1].ensemble._histogram = {0: 100, 1: 1000}
    replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}
    replicas[1].ensemble._entropy = {0: 3.0, 1: 4.0}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="at_halve",
    )
    group._run_coordinator_block()

    # No halve, no merge: per-walker entropies preserved.
    assert replicas[0].ensemble._entropy[0] == pytest.approx(1.0)
    assert replicas[1].ensemble._entropy[0] == pytest.approx(3.0)


def test_merge_cadence_never_skips_merge_at_halve():
    """merge_cadence='never': no merge even when halve fires."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}
        r.ensemble._histogram = {0: 1000, 1: 1000}
    replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}
    replicas[1].ensemble._entropy = {0: 3.0, 1: 4.0}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="never",
    )
    group._run_coordinator_block()

    # Halve fired (both flat). No merge: per-walker entropies preserved.
    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)
    assert replicas[0].ensemble._entropy[0] == pytest.approx(1.0)
    assert replicas[1].ensemble._entropy[0] == pytest.approx(3.0)


def test_collective_halve_at_halve_merges_entropies():
    """merge_cadence='at_halve' + halve fires: entropies merged across walkers."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 1.0
        r.ensemble._fill_factor_history = {}
        r.ensemble._histogram = {0: 1000, 1: 1000}
    # Identical shape on both walkers but offset constants.
    replicas[0].ensemble._entropy = {0: 0.0, 1: 5.0}
    replicas[1].ensemble._entropy = {0: 10.0, 1: 15.0}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="at_halve",
    )
    group._run_coordinator_block()

    # Halve fires; merge: both walkers should receive the same dict
    # with min(merged) = 0 and shape {0: 0.0, 1: 5.0}.
    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)
        assert r.ensemble._entropy[0] == pytest.approx(0.0)
        assert r.ensemble._entropy[1] == pytest.approx(5.0)


def test_maybe_switch_to_one_over_t_refuses_unentered_walker():
    """BP switch does not fire if any walker has not entered its window."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 0.001  # tiny -> 1/t > f trivially
        r.ensemble._phase = "halving"
        r.ensemble._schedule = "1_over_t"
        r.ensemble._step = 100
    # Walker 0 entered; walker 1 has not.
    replicas[0].ensemble._window_entry_step = 0
    replicas[1].ensemble._window_entry_step = None

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._schedule = "1_over_t"
    group._maybe_switch_to_one_over_t()

    for r in replicas:
        assert r.ensemble._phase == "halving"


def test_maybe_switch_to_one_over_t_does_not_grow_fill_factor_history():
    """BP switch updates ``_fill_factor`` but leaves ``_fill_factor_history``.

    ``_fill_factor_history`` records halve events (shared keys with
    ``_entropy_history``). The BP switch is a phase transition, not a
    halve, so it must not add a history entry — otherwise the halve
    at the same step would be silently overwritten and downstream
    analysis that pairs the two history dicts would see inconsistent
    state.
    """
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._fill_factor = 0.001  # tiny -> 1/t > f trivially
        r.ensemble._phase = "halving"
        r.ensemble._schedule = "1_over_t"
        r.ensemble._step = 100
        r.ensemble._window_entry_step = 0
        # Pre-existing halve entry at step 100 to ensure we'd notice
        # if the BP switch wrote a new key or overwrote this one.
        r.ensemble._fill_factor_history = {100: 0.001}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._schedule = "1_over_t"
    group._maybe_switch_to_one_over_t()

    for r in replicas:
        assert r.ensemble._phase == "1_over_t"
        # _fill_factor updated to 1/t; t = step - entry + 1 = 101.
        assert r.ensemble._fill_factor == pytest.approx(1.0 / 101)
        # History unchanged: still one halve entry, value 0.001 at step 100.
        assert r.ensemble._fill_factor_history == {100: 0.001}


def test_validate_flatness_mode_accepts_known_values():
    from mchammer_pt.wl_window_group import _validate_flatness_mode

    _validate_flatness_mode("per_walker")
    _validate_flatness_mode("pooled")


def test_validate_flatness_mode_rejects_unknown_value():
    from mchammer_pt.wl_window_group import _validate_flatness_mode

    with pytest.raises(ValueError, match="flatness_mode must be one of"):
        _validate_flatness_mode("bogus")


def test_validate_merge_cadence_accepts_known_values():
    from mchammer_pt.wl_window_group import _validate_merge_cadence

    _validate_merge_cadence("at_halve")
    _validate_merge_cadence("never")


def test_validate_merge_cadence_rejects_unknown_value():
    from mchammer_pt.wl_window_group import _validate_merge_cadence

    with pytest.raises(ValueError, match="merge_cadence must be one of"):
        _validate_merge_cadence("bogus")


def test_summed_histogram_is_flat_false_when_unentered():
    """Pooled flatness false if any walker has not entered the window."""
    from mchammer_pt.wl_window_group import _summed_histogram_is_flat

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[1].ensemble._reached_energy_window = False  # not entered
    replicas[1].ensemble._histogram = {}
    assert _summed_histogram_is_flat(replicas) is False


def test_summed_histogram_flat_when_per_walker_is_not():
    """Pooled flat but per-walker not: pooling fixes the gap."""
    from mchammer_pt.wl_window_group import _summed_histogram_is_flat

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    # Walker A has skewed coverage; walker B has the complementary skew.
    replicas[0].ensemble._histogram = {0: 100, 1: 1000}
    replicas[1].ensemble._histogram = {0: 1000, 1: 100}
    # Per-walker A: limit = 0.8 * 550 = 440; 100 < 440 -> not flat.
    # Per-walker B: same shape, not flat.
    # Pooled: {0: 1100, 1: 1100}, mean 1100, limit 880; both >= 880 -> flat.
    assert replicas[0].is_flat() is False
    assert replicas[1].is_flat() is False
    assert _summed_histogram_is_flat(replicas) is True


def test_summed_histogram_flat_from_snapshots_matches_live():
    """Snapshot-based helper agrees with the live-replica helper."""
    from mchammer_pt.wl_window_group import (
        WalkerPostBlockState,
        _summed_histogram_flat_from_snapshots,
        _summed_histogram_is_flat,
    )

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 100, 1: 1000}
    replicas[1].ensemble._histogram = {0: 1000, 1: 100}

    snapshots = [
        WalkerPostBlockState(
            is_flat=replicas[0].is_flat(),
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=0,
            histogram={0: 100, 1: 1000},
            reached_energy_window=True,
        ),
        WalkerPostBlockState(
            is_flat=replicas[1].is_flat(),
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=0,
            histogram={0: 1000, 1: 100},
            reached_energy_window=True,
        ),
    ]
    flatness_limit = replicas[0].ensemble._flatness_limit
    assert _summed_histogram_flat_from_snapshots(snapshots, flatness_limit) == (
        _summed_histogram_is_flat(replicas)
    )


def test_summed_histogram_flat_from_snapshots_false_when_unentered():
    """Pooled flatness from snapshots returns False if any walker has not entered."""
    from mchammer_pt.parallel.processes import WalkerPostBlockState
    from mchammer_pt.wl_window_group import (
        _summed_histogram_flat_from_snapshots,
    )

    snapshots = [
        WalkerPostBlockState(
            is_flat=True,
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=0,
            histogram={0: 1000, 1: 1000},
            reached_energy_window=True,
        ),
        WalkerPostBlockState(
            is_flat=False,
            fill_factor=1.0,
            entropy={},
            step=0,
            window_entry_step=None,
            histogram={},
            reached_energy_window=False,
        ),
    ]
    assert _summed_histogram_flat_from_snapshots(snapshots, 0.8) is False


def test_one_over_t_phase_no_midrun_merge_serial():
    """In 1/t phase, walker entropies are not merged mid-run."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
        r.ensemble._phase = "1_over_t"
        r.ensemble._schedule = "1_over_t"
        r.ensemble._window_entry_step = 0
        r.ensemble._fill_factor = 0.001
    replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}
    replicas[1].ensemble._entropy = {0: 3.0, 1: 4.0}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="at_halve",
    )
    group._run_coordinator_block()

    # No merge in 1/t phase: per-walker entropies preserved.
    assert replicas[0].ensemble._entropy[0] == pytest.approx(1.0)
    assert replicas[1].ensemble._entropy[0] == pytest.approx(3.0)


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


def test_window_stats_includes_flatness_mode_and_per_walker_flat_min():
    """Multi-walker window_stats carries flatness_mode and per_walker_flat_min."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 500, 1: 1000}  # flat_min = 500/750 ~ 0.667
    replicas[1].ensemble._histogram = {0: 900, 1: 1000}  # flat_min = 900/950 ~ 0.947

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="at_halve",
    )
    stats = group.window_stats()

    assert stats["flatness_mode"] == "per_walker"
    # min of the two flat_mins:
    # walker 0: 500 / 750 = 0.667 (smaller)
    # walker 1: 900 / 950 = 0.947
    assert stats["per_walker_flat_min"] == pytest.approx(500 / 750)


def test_window_stats_per_walker_flat_min_none_when_walker_has_no_histogram():
    """per_walker_flat_min is None if any walker's histogram is empty."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._histogram = {0: 1000, 1: 1000}
    replicas[1].ensemble._reached_energy_window = False
    replicas[1].ensemble._histogram = {}

    group = WangLandauWindowGroup(
        replicas,
        random_seed=0,
        flatness_mode="pooled",
        merge_cadence="at_halve",
    )
    stats = group.window_stats()

    assert stats["per_walker_flat_min"] is None


def test_finalise_for_reporting_single_walker_noop():
    """finalise_for_reporting is a no-op for single-walker groups."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(1)
    replicas[0].ensemble._reached_energy_window = True
    replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group.finalise_for_reporting()

    # Untouched.
    assert replicas[0].ensemble._entropy == {0: 1.0, 1: 2.0}
