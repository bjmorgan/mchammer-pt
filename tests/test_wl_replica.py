"""Unit tests for WangLandauReplica."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


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
    e = replica.ensemble
    if hasattr(e, "_phase"):
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
