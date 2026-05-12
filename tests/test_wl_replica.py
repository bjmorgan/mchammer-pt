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
