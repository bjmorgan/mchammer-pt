"""Parallel (multiprocess) PT run via the ProcessPool factory.

Demonstrates the `CanonicalParallelTempering.process_pool` classmethod:
one call constructs a process-parallel orchestrator, spawning per-
replica seeds, writing the cluster expansion to a managed temp
directory, and constructing a ProcessPool at the same temperature
ladder as the orchestrator. Used as a context manager, worker
processes are joined on exit.

Run from the repo root:

    python examples/03_parallel_workers.py
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
    params = rng.normal(scale=0.05, size=len(cs))
    params[0] = -1.0
    return ClusterExpansion(cluster_space=cs, parameters=params)


def main() -> None:
    ce = build_toy_ce()
    atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(1)
    au_indices = rng.choice(len(atoms), size=len(atoms) // 2, replace=False)
    symbols = np.array(atoms.get_chemical_symbols())
    symbols[au_indices] = "Au"
    atoms.set_chemical_symbols(symbols.tolist())

    with CanonicalParallelTempering.process_pool(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[200.0, 400.0, 800.0, 1600.0],
        block_size=200,
        random_seed=0,
    ) as pt:
        history = pt.run(n_cycles=30)

    print(
        f"Ran {history.energies_per_cycle.shape[0] - 1} cycles across "
        f"{history.energies_per_cycle.shape[1]} processes."
    )


if __name__ == "__main__":
    main()
