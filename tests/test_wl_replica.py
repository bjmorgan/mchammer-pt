"""Unit tests for WangLandauReplica."""

from __future__ import annotations

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
    """Two replicas with the same seed advance to the same state when each runs alone."""
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
