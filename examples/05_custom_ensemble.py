"""Parallel tempering with a custom CanonicalEnsemble subclass.

Demonstrates the `ensemble_cls` and `ensemble_kwargs` parameters of
`CanonicalParallelTempering`. Any `mchammer.CanonicalEnsemble` subclass
can ride the parallel-tempering machinery — `_do_trial_step`,
`_acceptance_condition`, or any other internal hook can be overridden
in the subclass without touching `mchammer-pt`.

This example overrides `_do_trial_step` to record a per-replica trial
counter on top of `mchammer`'s standard canonical swap. The override
itself is not physically meaningful; it is a stand-in for real custom
move sets (cluster moves, row/chain translations, constrained swaps)
that subclasses use to escape kinetic traps single-site swaps cannot
cross. The pattern is the same in every case: subclass
`CanonicalEnsemble`, override the hook, pass the class via
``ensemble_cls=``.

Run from the repo root:

    python examples/05_custom_ensemble.py

The example uses the in-process `SerialPool` (the default) so the
counter on each replica's ensemble is reachable for inspection at the
end of the run. For multiprocess runs via
`CanonicalParallelTempering.process_pool(...)` the same `ensemble_cls=`
kwarg works, but the subclass must live in an importable ``.py``
module (top-level classes in ``python script.py`` work; classes
defined in a Jupyter cell or REPL do not — `ProcessPool` rejects this
case up-front).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from mchammer_pt import CanonicalParallelTempering


class CountingCanonicalEnsemble(CanonicalEnsemble):
    """Canonical MC subclass that counts its own trial steps.

    The override of `_do_trial_step` is the extension point named in
    issue #6 — real subclasses use it to add custom MC moves. Here it
    just increments a counter on top of the parent's canonical swap,
    so the example stays self-contained and pedagogical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_trials = 0

    def _do_trial_step(self) -> int:
        self.n_trials += 1
        return super()._do_trial_step()


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
    block_size = 200
    n_cycles = 50

    pt = CanonicalParallelTempering(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[200.0, 400.0, 800.0, 1600.0],
        block_size=block_size,
        random_seed=0,
        ensemble_cls=CountingCanonicalEnsemble,
    )
    pt.run(n_cycles=n_cycles)

    # Each replica advanced n_cycles * block_size trial steps; the
    # counter on the per-replica ensemble pins that the override
    # actually fired inside the PT machinery.
    expected = n_cycles * block_size
    print(f"Expected trial-step count per replica: {expected}")
    for i, replica in enumerate(pt.pool._replicas):  # type: ignore[attr-defined]
        ensemble = replica._ensemble
        assert isinstance(ensemble, CountingCanonicalEnsemble)
        T = pt.temperatures[i]
        print(
            f"  replica {i}  T = {T:>6.1f} K   "
            f"n_trials = {ensemble.n_trials}"
        )


if __name__ == "__main__":
    main()
