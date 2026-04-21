"""Shared pytest fixtures for the mchammer-pt test suite."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace


@pytest.fixture(scope="session")
def toy_cluster_space() -> ClusterSpace:
    """Cu/Au FCC cluster space with a 3.5 A pair cutoff."""
    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    return ClusterSpace(
        structure=primitive,
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )


@pytest.fixture(scope="session")
def toy_ce(toy_cluster_space: ClusterSpace) -> ClusterExpansion:
    """Cluster expansion with deterministic random ECIs on `toy_cluster_space`."""
    rng = np.random.default_rng(0)
    parameters = rng.normal(loc=0.0, scale=0.05, size=len(toy_cluster_space))
    parameters[0] = -1.0  # zerolet dominates so energies are in a physical range
    return ClusterExpansion(cluster_space=toy_cluster_space, parameters=parameters)


@pytest.fixture
def toy_atoms() -> Atoms:
    """3x3x3 cubic-FCC supercell, half-decorated with Au.

    Cell side is 12 A, safely larger than twice the 3.5 A pair cutoff,
    so the CE evaluates without cluster self-interaction.
    """
    atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(1)
    n_au = len(atoms) // 2
    au_indices = rng.choice(len(atoms), size=n_au, replace=False)
    symbols = np.array(atoms.get_chemical_symbols())
    symbols[au_indices] = "Au"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms
