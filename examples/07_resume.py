"""Checkpoint and resume on a short PT run.

Run from the repo root:

    python examples/07_resume.py

Demonstrates `CheckpointWriter` for periodic mid-run saves and
`CanonicalParallelTempering.resume` for picking up where a previous
run left off. The bit-identical contract holds: running A then
resuming and running B produces, after `ExchangeHistory.concatenate`,
the same trajectory as a single run of total length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace

from mchammer_pt import (
    CanonicalParallelTempering,
    ExchangeHistory,
)


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

    path = Path("resume_demo.h5")

    # Run A: 25 cycles. Periodic checkpoint every 10 cycles + final.
    pt_a = CanonicalParallelTempering(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[200.0, 400.0, 800.0, 1600.0],
        block_size=100,
        random_seed=0,
    )
    pt_a.attach_checkpoint_writer(path, interval=10)
    history_a = pt_a.run(n_cycles=25)
    print(f"Run A: {history_a.energies_per_cycle.shape[0] - 1} cycles done.")

    # Resume from the checkpoint and run another 25 cycles.
    pt_b = CanonicalParallelTempering.resume(path, cluster_expansion=ce)
    history_b = pt_b.run(n_cycles=25)
    print(f"Run B: {history_b.energies_per_cycle.shape[0] - 1} cycles done.")

    combined = ExchangeHistory.concatenate(history_a, history_b)
    print(
        f"Combined: {combined.energies_per_cycle.shape[0] - 1} cycles, "
        f"swap_attempted={combined.swap_attempted.tolist()}, "
        f"swap_accepted={combined.swap_accepted.tolist()}"
    )

    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
