"""Unit tests for WangLandauReplica."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import WangLandauEnsemble

from mchammer_pt.wl_replica import WangLandauReplica
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _e0() -> float:
    """Initial energy for the standard test atoms."""
    atoms = make_wl_atoms()
    ce = make_wl_ce()
    return float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers
    ))


def _make_wl_replica(schedule: str | None = None) -> WangLandauReplica:
    """Construct a WangLandauReplica over a wide window.

    Args:
        schedule: optional schedule override (e.g. ``"1_over_t"``).
            Silently skipped when the installed icet does not support
            the ``schedule`` parameter.
    """
    ce = make_wl_ce()
    atoms = make_wl_atoms()
    e0 = _e0()
    ensemble_kwargs: dict | None = None
    if schedule is not None:
        if "schedule" not in inspect.signature(WangLandauEnsemble.__init__).parameters:
            pytest.skip("requires icet with WangLandauEnsemble schedule parameter")
        ensemble_kwargs = {"schedule": schedule}
    return WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
        ensemble_kwargs=ensemble_kwargs,
    )


@pytest.fixture
def wl_replica_factory():
    """Return a factory callable that produces a fresh WangLandauReplica.

    Call with no arguments for a default halving-schedule replica, or
    ``wl_replica_factory(schedule="1_over_t")`` for the BP schedule.
    """
    return _make_wl_replica


def test_wl_replica_constructs_with_in_window_initial_energy():
    """A WL replica builds when its initial energy falls in window."""
    from mchammer_pt.wl_replica import WangLandauReplica

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    # First-look energy lookup: build a throwaway calculator to see
    # what energy our initial atoms have.
    from mchammer.calculators import ClusterExpansionCalculator
    calc = ClusterExpansionCalculator(atoms, ce)
    e0 = float(calc.calculate_total(occupations=atoms.numbers))
    # Window covers an interval of width 10 centred on the initial energy.
    replica = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 5.0,
        energy_limit_right=e0 + 5.0,
        random_seed=0,
    )
    assert replica.current_energy() == pytest.approx(e0)


def test_wl_replica_rejects_out_of_window_initial_energy():
    """Initial energy outside the window raises ValueError."""
    from mchammer_pt.wl_replica import WangLandauReplica

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    from mchammer.calculators import ClusterExpansionCalculator
    calc = ClusterExpansionCalculator(atoms, ce)
    e0 = float(calc.calculate_total(occupations=atoms.numbers))
    with pytest.raises(ValueError, match="outside window"):
        WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=0.1,
            energy_limit_left=e0 + 1.0,
            energy_limit_right=e0 + 2.0,
            random_seed=0,
        )


def test_init_seeds_starting_bin_into_histogram_and_entropy():
    """The constructor records the starting bin in _histogram and _entropy.

    With the seed, the flatness gate sees the starting bin from cycle 0,
    so a walker that leaves it on step 0 and never returns cannot
    silently saturate the histogram over a strict subset of the window
    and trigger a premature halving.
    """
    replica = _make_wl_replica()
    e = replica.ensemble
    bin_init = e._get_bin_index(e._potential)

    assert bin_init in e._histogram
    assert e._histogram[bin_init] == 0
    assert bin_init in e._entropy
    assert e._entropy[bin_init] == 0.0


def test_wl_replica_log_g_returns_minus_inf_out_of_window():
    """log_g at an out-of-window energy returns -inf."""
    from mchammer_pt.wl_replica import WangLandauReplica
    ce, atoms = make_wl_ce(), make_wl_atoms()
    from mchammer.calculators import ClusterExpansionCalculator
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))
    replica = WangLandauReplica(
        cluster_expansion=ce, atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 1.0, energy_limit_right=e0 + 1.0,
        random_seed=0,
    )
    assert replica.log_g(e0 + 1000.0) == -np.inf
    assert replica.log_g(e0 - 1000.0) == -np.inf


def test_wl_replica_log_g_returns_zero_for_unvisited_in_window_bin():
    """log_g for an unvisited in-window bin is 0.0 (entropy defaults to 0)."""
    from mchammer_pt.wl_replica import WangLandauReplica
    ce, atoms = make_wl_ce(), make_wl_atoms()
    from mchammer.calculators import ClusterExpansionCalculator
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))
    replica = WangLandauReplica(
        cluster_expansion=ce, atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 1.0, energy_limit_right=e0 + 1.0,
        random_seed=0,
    )
    # No MC advance yet, so entropy dict is empty.
    assert replica.log_g(e0) == 0.0


def test_wl_replica_set_occupations_refreshes_potential():
    """After set_occupations, _potential and current_energy reflect the new state."""
    from mchammer_pt.wl_replica import WangLandauReplica
    ce, atoms = make_wl_ce(), make_wl_atoms()
    from mchammer.calculators import ClusterExpansionCalculator
    calc = ClusterExpansionCalculator(atoms, ce)
    e0 = float(calc.calculate_total(occupations=atoms.numbers))

    replica = WangLandauReplica(
        cluster_expansion=ce, atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0, energy_limit_right=e0 + 100.0,
        random_seed=0,
    )

    # Build a different occupation vector by swapping two atoms.
    occ = atoms.numbers.copy()
    occ[[0, -1]] = occ[[-1, 0]]
    expected = float(calc.calculate_total(occupations=occ))

    replica.set_occupations(occ)
    assert replica.current_energy() == pytest.approx(expected)
    # _reached_energy_window should be True because expected is in window.
    assert replica.ensemble._reached_energy_window is True


def test_wl_replica_advance_is_rng_isolated():
    """Two replicas with the same seed advance identically when each runs alone."""
    from mchammer_pt.wl_replica import WangLandauReplica
    ce, atoms = make_wl_ce(), make_wl_atoms()
    from mchammer.calculators import ClusterExpansionCalculator
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))

    a = WangLandauReplica(
        cluster_expansion=ce, atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0, energy_limit_right=e0 + 100.0,
        random_seed=42,
    )
    b = WangLandauReplica(
        cluster_expansion=ce, atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0, energy_limit_right=e0 + 100.0,
        random_seed=42,
    )
    a.advance(100)
    b.advance(100)
    np.testing.assert_array_equal(
        a.current_occupations(), b.current_occupations()
    )


def test_wl_replica_snapshot_and_restore_round_trip(tmp_path):
    """Snapshot, write the container, restore into a fresh replica; state matches."""
    from mchammer_pt.wl_replica import WangLandauReplica
    ce, atoms = make_wl_ce(), make_wl_atoms()
    from mchammer.calculators import ClusterExpansionCalculator
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))

    replica = WangLandauReplica(
        cluster_expansion=ce, atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0, energy_limit_right=e0 + 100.0,
        random_seed=7,
    )
    replica.advance(50)
    extras = replica.snapshot_for_checkpoint()
    assert "sites_by_species" in extras

    dc_path = tmp_path / "wl.dc"
    replica.data_container().write(str(dc_path))

    from mchammer.data_containers.wang_landau_data_container import (  # type: ignore[import-untyped]
        WangLandauDataContainer,
    )
    container = WangLandauDataContainer.read(str(dc_path))

    restored = WangLandauReplica.restart_from(
        container,
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0, energy_limit_right=e0 + 100.0,
        random_seed=7,
        sites_by_species=extras["sites_by_species"],
    )
    np.testing.assert_array_equal(
        restored.current_occupations(), replica.current_occupations()
    )
    assert restored.current_energy() == pytest.approx(replica.current_energy())


def test_wl_replica_one_over_t_snapshot_round_trips(tmp_path):
    """1/t-schedule fields round-trip through snapshot/restore.

    Requires icet with ``schedule`` parameter on
    ``WangLandauEnsemble``. Skips cleanly on mainline icet.
    """
    import inspect

    from mchammer.ensembles import WangLandauEnsemble

    if "schedule" not in inspect.signature(WangLandauEnsemble.__init__).parameters:
        pytest.skip("requires icet with WangLandauEnsemble schedule parameter")

    from mchammer.calculators import (
        ClusterExpansionCalculator,
    )
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )

    from mchammer_pt.wl_replica import WangLandauReplica

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))

    replica = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=7,
        ensemble_kwargs={"schedule": "1_over_t"},
    )
    # Advance enough to enter the window (so _window_entry_step is set).
    replica.advance(100)
    extras = replica.snapshot_for_checkpoint()
    assert "sites_by_species" in extras

    dc_path = tmp_path / "wl_one_over_t.dc"
    replica.data_container().write(str(dc_path))
    container = WangLandauDataContainer.read(str(dc_path))

    assert container._last_state["schedule"] == "1_over_t"
    assert "phase" in container._last_state
    assert "window_entry_step" in container._last_state

    restored = WangLandauReplica.restart_from(
        container,
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=7,
        ensemble_kwargs={"schedule": "1_over_t"},
        sites_by_species=extras["sites_by_species"],
    )
    assert restored.ensemble._phase == replica.ensemble._phase
    assert (
        restored.ensemble._window_entry_step
        == replica.ensemble._window_entry_step
    )


def test_wl_replica_restore_state_does_not_mutate_caller_container(tmp_path):
    """restore_state deep-copies _last_state, so the caller's container
    is unchanged and not aliased by the ensemble's internal state.
    """
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.data_containers.base_data_container import (
        BaseDataContainer,
    )

    from mchammer_pt.wl_replica import WangLandauReplica

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))

    replica = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=42,
    )
    replica.advance(10)
    replica.snapshot_for_checkpoint()
    # Write and read back to get a separate container object,
    # mirroring the real resume path through read_hdf5.
    dc_path = tmp_path / "container.dc"
    replica.data_container().write(str(dc_path))
    container = BaseDataContainer.read(str(dc_path))

    original_last_state = copy.deepcopy(container._last_state)
    replica.restore_state(container)
    # Caller's container must be value-equal and not aliased.
    assert container._last_state == original_last_state
    assert container._last_state is not replica._ensemble._data_container._last_state


def test_is_flat_returns_false_before_window_reached():
    """Walker outside its window is not flat."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    r = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
    )
    # Override _reached_energy_window to simulate "outside" state.
    r.ensemble._reached_energy_window = False
    assert r.is_flat() is False


def test_is_flat_returns_true_on_flat_histogram():
    """A perfectly flat histogram passes mchammer's flatness criterion."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    r = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
    )
    r.ensemble._reached_energy_window = True
    r.ensemble._histogram = {0: 1000, 1: 1000, 2: 1000}
    assert r.is_flat() is True


def test_is_flat_returns_false_on_uneven_histogram():
    """A histogram with one bin below limit*mean is not flat."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    r = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
    )
    r.ensemble._reached_energy_window = True
    r.ensemble._histogram = {0: 100, 1: 1000, 2: 900}
    assert r.is_flat() is False


def test_default_ensemble_cls_is_coordinated():
    """The default ensemble_cls is CoordinatedWangLandauEnsemble."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble
    from mchammer_pt.wl_replica import WangLandauReplica
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    r = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
    )
    assert isinstance(r.ensemble, CoordinatedWangLandauEnsemble)


def test_non_coordinated_ensemble_cls_rejected():
    """Passing the plain WangLandauEnsemble raises TypeError."""
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.ensembles import WangLandauEnsemble

    from mchammer_pt.wl_replica import WangLandauReplica
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    with pytest.raises(TypeError, match="CoordinatedWangLandauEnsemble"):
        WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=0.1,
            energy_limit_left=e0 - 100.0,
            energy_limit_right=e0 + 100.0,
            random_seed=0,
            ensemble_cls=WangLandauEnsemble,
        )


# ---------------------------------------------------------------------------
# Slot surface tests (Task 2)
# ---------------------------------------------------------------------------


def test_walker_states_initialised_as_one_default_snapshot(wl_replica_factory):
    """walker_states starts as a 1-tuple with default WalkerPostBlockState."""
    from mchammer_pt.wl_coordinator import WalkerPostBlockState

    replica = wl_replica_factory()
    assert isinstance(replica.walker_states, tuple)
    assert len(replica.walker_states) == 1
    state = replica.walker_states[0]
    assert isinstance(state, WalkerPostBlockState)
    assert state.step == 0
    assert state.reached_energy_window is False


def test_advance_refreshes_walker_states(wl_replica_factory):
    """advance(n) populates walker_states from live ensemble state."""
    replica = wl_replica_factory()
    replica.advance(50)
    state = replica.walker_states[0]
    assert state.step == int(replica.ensemble.step)
    assert state.fill_factor == float(replica.ensemble._fill_factor)


def test_apply_plan_halve_only_halves_fill_factor(wl_replica_factory):
    """plan.halve=True halves _fill_factor and resets histogram."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan

    replica = wl_replica_factory()
    replica.advance(500)
    f_before = float(replica.ensemble._fill_factor)
    replica.apply_plan(CoordinatorPlan(
        halve=True, merged_entropy=None, switch_to_phase=None
    ))
    assert replica.ensemble._fill_factor == f_before / 2.0


def test_apply_plan_merged_entropy_writes_to_ensemble(wl_replica_factory):
    """plan.merged_entropy writes to self._ensemble._entropy."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan

    replica = wl_replica_factory()
    replica.advance(50)
    plan = CoordinatorPlan(
        halve=False,
        merged_entropy={0: 1.5, 1: 2.5},
        switch_to_phase=None,
    )
    replica.apply_plan(plan)
    assert replica.ensemble._entropy == {0: 1.5, 1: 2.5}


def test_apply_plan_switch_to_phase_flips_and_recomputes_fill_factor(
    wl_replica_factory,
):
    """plan.switch_to_phase='1_over_t' flips _phase and sets _fill_factor=1/t."""
    from mchammer_pt.wl_coordinator import CoordinatorPlan

    replica = wl_replica_factory(schedule="1_over_t")
    replica.advance(200)
    # Force a window_entry_step so the 1/t branch can compute t.
    e = replica.ensemble
    if e._window_entry_step is None:
        e._window_entry_step = 0
    step = int(e.step)
    entry = int(e._window_entry_step)
    expected_t = step - entry + 1
    replica.apply_plan(CoordinatorPlan(
        halve=False, merged_entropy=None, switch_to_phase="1_over_t"
    ))
    assert replica.ensemble._phase == "1_over_t"
    assert replica.ensemble._fill_factor == 1.0 / expected_t


def test_reroll_exchange_idx_is_noop(wl_replica_factory):
    """Single-walker reroll has no observable effect."""
    replica = wl_replica_factory()
    replica.advance(50)
    f_before = float(replica.ensemble._fill_factor)
    replica.reroll_exchange_idx()
    assert replica.ensemble._fill_factor == f_before


def test_replica_satisfies_wang_landau_slot_protocol(wl_replica_factory):
    """WangLandauReplica satisfies the runtime-checkable Protocol."""
    from mchammer_pt.wl_replica import WangLandauSlot

    replica = wl_replica_factory()
    assert isinstance(replica, WangLandauSlot)


def test_set_occupations_seeds_new_bin_into_histogram_and_entropy():
    """set_occupations records the new bin if it wasn't already tracked.

    REWL exchanges and process-pool transports go through
    set_occupations, so this seeding also covers exchange arrivals.
    """
    replica = _make_wl_replica()
    e = replica.ensemble

    # Construct alternative occupations that produce a different energy
    # (and therefore a different bin) but stay inside the window.
    occ = replica.current_occupations()
    # Flip the species of the first two distinct sites to change energy.
    species_a, species_b = int(occ[0]), None
    for s in occ[1:]:
        if int(s) != species_a:
            species_b = int(s)
            break
    assert species_b is not None, "fixture should have two species"
    new_occ = occ.copy()
    # Find one site of each species and swap them.
    idx_a = int(np.where(new_occ == species_a)[0][0])
    idx_b = int(np.where(new_occ == species_b)[0][0])
    new_occ[idx_a], new_occ[idx_b] = species_b, species_a

    # Compute the bin this configuration lands in.
    new_potential = float(e.calculator.calculate_total(occupations=new_occ))
    new_bin = e._get_bin_index(new_potential)

    # Sanity: make sure we're actually moving to a fresh bin.
    if new_bin in e._histogram:
        pytest.skip("swap didn't change bin; fixture-dependent")

    replica.set_occupations(new_occ)

    assert new_bin in e._histogram
    assert e._histogram[new_bin] == 0
    assert new_bin in e._entropy
    assert e._entropy[new_bin] == 0.0


def test_set_occupations_preserves_existing_count_for_known_bin():
    """If the new bin is already in _histogram, its count is unchanged.

    setdefault semantics: only initialise when missing.
    """
    replica = _make_wl_replica()
    e = replica.ensemble
    bin_init = e._get_bin_index(e._potential)

    # Pretend the starting bin has been visited a thousand times.
    e._histogram[bin_init] = 1000
    e._entropy[bin_init] = 7.5

    # set_occupations with the SAME initial occupations: new_bin == bin_init.
    occ = replica.current_occupations()
    replica.set_occupations(occ)

    assert e._histogram[bin_init] == 1000
    assert e._entropy[bin_init] == 7.5


def test_restore_state_seeds_restored_bin_when_last_state_histogram_is_empty():
    """If the saved _last_state has an empty histogram, restore_state
    still records the restored bin via the seed.

    Covers the case where a checkpoint was written before any
    _update_entropy call (zero-step restart) — without the seed the
    restored bin would be invisible to the flatness gate.
    """
    src = _make_wl_replica()
    src.refresh_last_state()
    container = src.data_container()
    # Force the saved state to look like a pre-step checkpoint:
    # occupations recorded but no entropy/histogram visits yet.
    container._last_state["histogram"] = {}
    container._last_state["entropy"] = {}

    dst = _make_wl_replica()
    dst.ensemble._histogram.clear()
    dst.ensemble._entropy.clear()

    dst.restore_state(container)

    restored_bin = dst.ensemble._get_bin_index(dst.ensemble._potential)
    assert restored_bin in dst.ensemble._histogram
    assert dst.ensemble._histogram[restored_bin] == 0
    assert restored_bin in dst.ensemble._entropy
    assert dst.ensemble._entropy[restored_bin] == 0.0
