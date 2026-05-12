"""Live progress monitoring on a long PT run.

Run from the repo root:

    python examples/06_progress_monitoring.py

Demonstrates the `ProgressPrinter` cycle callback. Each emitted line
goes to stderr by default (so it composes with shell redirection
without extra plumbing) and carries a wall-clock timestamp, the
cycle counter, completion fraction, elapsed and ETA, plus
cumulative per-pair swap-acceptance rates — enough signal to spot
a bad temperature ladder while a multi-hour run is still going,
rather than after it finishes.

The example is short and runs in a few seconds; bump `n_cycles` and
shrink `interval` to see what a real run's stderr looks like.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace

from mchammer_pt import CanonicalParallelTempering, ProgressPrinter


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

    pt = CanonicalParallelTempering(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[200.0, 400.0, 800.0, 1600.0],
        block_size=100,
        random_seed=0,
    )

    # Emit a progress line every 20 cycles (and one at the final cycle).
    # Reusing one printer across multiple `pt.run(...)` calls is safe:
    # the elapsed/ETA clock resets at the start of each run.
    pt.attach_cycle_callback(ProgressPrinter(interval=20))

    pt.run(n_cycles=100)


if __name__ == "__main__":
    main()
