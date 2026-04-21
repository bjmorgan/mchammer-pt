"""Parallel (multiprocess) PT run.

Demonstrates spawning one worker process per replica. The CE is
written to a temp file, the ProcessBackend reads it back inside each
worker, and only occupation vectors cross the process boundary during
the run.

Run from the repo root:

    python examples/03_parallel_workers.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace

from mchammer_pt import CanonicalParallelTempering
from mchammer_pt.parallel.processes import ProcessBackend


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

    temperatures = [200.0, 400.0, 800.0, 1600.0]

    with tempfile.TemporaryDirectory() as tmpdir:
        ce_path = Path(tmpdir) / "toy.ce"
        ce.write(str(ce_path))
        seeds = [
            int(np.random.SeedSequence(0).spawn(5)[i].generate_state(1)[0])
            for i in range(len(temperatures))
        ]
        backend = ProcessBackend(
            ce_path=ce_path,
            initial_atoms=atoms,
            temperatures=temperatures,
            seeds=seeds,
        )
        try:
            pt = CanonicalParallelTempering(
                cluster_expansion=ce,
                atoms=atoms,
                temperatures=temperatures,
                block_size=200,
                random_seed=0,
                backend=backend,
            )
            history = pt.run(n_cycles=30)
        finally:
            backend.shutdown()

    print(
        f"Ran {history.energies_per_cycle.shape[0] - 1} cycles across "
        f"{history.energies_per_cycle.shape[1]} processes."
    )


if __name__ == "__main__":
    main()
