"""Minimal CanonicalParallelTempering run on a Cu/Au CE.

Run from the repo root:

    python examples/01_basic_canonical.py

The example builds a tiny Cu/Au cluster expansion on the fly, sets up
four replicas on a short temperature ladder, runs 100 cycles, and
prints per-pair swap acceptance rates. It completes in a few seconds
on a laptop and is self-contained — no external CE file needed.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace

from mchammer_pt import CanonicalParallelTempering


def build_toy_ce() -> ClusterExpansion:
    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    cs = ClusterSpace(structure=primitive, cutoffs=[3.5], chemical_symbols=["Cu", "Au"])
    rng = np.random.default_rng(0)
    parameters = rng.normal(loc=0.0, scale=0.05, size=len(cs))
    parameters[0] = -1.0
    return ClusterExpansion(cluster_space=cs, parameters=parameters)


def build_atoms():
    atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(1)
    au_indices = rng.choice(len(atoms), size=len(atoms) // 2, replace=False)
    symbols = np.array(atoms.get_chemical_symbols())
    symbols[au_indices] = "Au"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def main() -> None:
    ce = build_toy_ce()
    atoms = build_atoms()
    pt = CanonicalParallelTempering(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[200.0, 400.0, 800.0, 1600.0],
        block_size=200,
        random_seed=0,
    )
    history = pt.run(n_cycles=100)
    print("Final per-replica energies (eV):")
    for T, E in zip(pt.temperatures, history.energies_per_cycle[-1], strict=True):
        print(f"  T = {T:>6.1f} K   E = {E:.4f}")
    print("Per-pair swap acceptance rates:")
    rates = history.swap_accepted / np.maximum(history.swap_attempted, 1)
    for i, rate in enumerate(rates):
        print(f"  pair ({i}, {i + 1}): {rate:.2%}")


if __name__ == "__main__":
    main()
