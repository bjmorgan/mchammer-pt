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
    """FCC conventional cell, two-species (Ag/Au), 3.5 A pair cutoff."""
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


def make_serial_wl_pool_mixed():
    """SerialWangLandauPool with walkers_per_window=[1, 2].

    Window 0 holds a bare WangLandauReplica (W=1); window 1 holds a
    WangLandauWindowGroup with two walkers (W=2). Total M=3 walkers.

    data_container_file is None to satisfy the current W>1 guard.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=[1, 2],
        data_container_file=None,
    )
    return pt._pool
