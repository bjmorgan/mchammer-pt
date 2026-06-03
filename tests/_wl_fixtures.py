"""Test fixtures for Wang-Landau replica/orchestrator tests.

A toy 2D Ising-flavoured cluster expansion small enough that the
density of states converges in a handful of seconds, and whose
energy range is known analytically so window construction is
deterministic.
"""

from __future__ import annotations

from pathlib import Path

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


def distinct_in_window_pair(
    ce: ClusterExpansion,
) -> tuple[Atoms, Atoms, float, float]:
    """Two same-composition configs with well-separated energies.

    Builds ``a`` from :func:`make_wl_atoms` and ``b`` by swapping the
    first Ag and first Au atom, then returns ``(a, b, ea, eb)`` with the
    energies computed under ``ce``. Used by the per-walker REWL tests to
    give each walker a distinguishable in-window start. The energy-gap
    assertion guards the callers' window sizing; if it ever fails, swap
    a nearest-neighbour Ag/Au pair to widen the gap.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    a = make_wl_atoms()
    b = a.copy()
    symbols = list(b.get_chemical_symbols())
    i_ag = symbols.index("Ag")
    i_au = symbols.index("Au")
    symbols[i_ag], symbols[i_au] = symbols[i_au], symbols[i_ag]
    b.set_chemical_symbols(symbols)

    def energy(at: Atoms) -> float:
        return float(
            ClusterExpansionCalculator(at, ce).calculate_total(
                occupations=at.numbers
            )
        )

    ea, eb = energy(a), energy(b)
    assert abs(ea - eb) > 0.5, (ea, eb)
    return a, b, ea, eb


def make_process_wl_pool_w2(tmp_path: Path):
    """ProcessWangLandauPool (in-process workers) with 2 windows x W=2.

    Both windows share the same energy range; W=2 walkers per window
    gives M=4 total walkers. ``data_container_file`` is not used for
    in-process pools, so the W>1 + file guard in ``process_pool`` is
    irrelevant here.

    Args:
        tmp_path: pytest-managed temp directory. The caller must pass
            its own ``tmp_path`` (or a sub-directory thereof) so the
            CE artefacts the in-process pool writes are cleaned up
            after the test.

    Returns a :class:`ProcessWangLandauPool` backed by
    :class:`InProcessWorkerConn` instances. Callers are responsible for
    calling ``pool.shutdown()`` after use.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from tests._in_process_pool import make_in_process_wl_pool

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    lo, hi = e0 - 100.0, e0 + 100.0
    tmp_path.mkdir(parents=True, exist_ok=True)
    return make_in_process_wl_pool(
        tmp_path,
        windows=[(lo, hi), (lo, hi)],
        seeds=[0, 1],
        n_walkers_per_window=2,
    )


def make_serial_wl_pool_mixed():
    """SerialWangLandauPool with walkers_per_window=[1, 2].

    Window 0 holds a bare WangLandauReplica (W=1); window 1 holds a
    WangLandauWindowGroup with two walkers (W=2). Total M=3 walkers.
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
