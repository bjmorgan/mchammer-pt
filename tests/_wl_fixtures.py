"""Test fixtures for Wang-Landau replica/orchestrator tests.

A toy 2D Ising-flavoured cluster expansion small enough that the
density of states converges in a handful of seconds, and whose
energy range is known analytically so window construction is
deterministic.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace


def make_wl_cluster_space() -> ClusterSpace:
    """4x4x1 single-layer FCC, two-species, 1.1 A pair cutoff."""
    primitive = bulk("Au", "fcc", a=4.0, cubic=True)
    return ClusterSpace(
        structure=primitive,
        cutoffs=[3.5],
        chemical_symbols=["Ag", "Au"],
    )


def make_wl_ce() -> ClusterExpansion:
    """CE with antiferromagnetic-like pair ECI for a clear DOS shape."""
    cs = make_wl_cluster_space()
    parameters = np.zeros(len(cs))
    parameters[0] = 0.0
    parameters[1] = 0.5  # singlet
    parameters[2] = 1.0  # nearest-neighbour pair
    return ClusterExpansion(cluster_space=cs, parameters=parameters)


def make_wl_atoms(n_au: int = 8) -> Atoms:
    """2x2x2 supercell with `n_au` Au atoms and the rest Ag."""
    atoms = bulk("Au", "fcc", a=4.0, cubic=True).repeat((2, 2, 2))
    symbols = ["Ag"] * len(atoms)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(atoms), size=n_au, replace=False)
    for i in indices:
        symbols[i] = "Au"
    atoms.set_chemical_symbols(symbols)
    return atoms
