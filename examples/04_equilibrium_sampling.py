"""Equilibrium sampling via burn-in discard.

All replicas in :class:`~mchammer_pt.CanonicalParallelTempering` start
from the same configuration. This is the standard parallel-tempering
cold-start: low-temperature replicas equilibrate by inheriting
already-thermalised configurations from high temperatures via accepted
exchanges, much faster than they would on their own. The resulting
trajectory splits into a burn-in regime (equilibration) and a
collection regime (equilibrium sampling) — for equilibrium statistics
you discard the former and average over the latter.

This example:

1. prints the first several rows of the coldest-temperature trajectory
   so the cold-start descent is visible,
2. computes per-temperature means with and without the burn-in slice,
3. shows the per-pair swap-acceptance rates as a ladder-quality check.

Run from the repo root:

    python examples/04_equilibrium_sampling.py
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
    """Ordered half-Au / half-Cu start: Au occupies the first half of
    the bulk supercell's sites and Cu the second half. Far from the
    disordered equilibrium at any of the temperatures used here, so
    the cold-start drop in energy is visible in the first cycle."""
    atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))
    symbols = ["Au"] * (len(atoms) // 2) + ["Cu"] * (len(atoms) - len(atoms) // 2)
    atoms.set_chemical_symbols(symbols)
    return atoms


def main() -> None:
    ce = build_toy_ce()
    atoms = build_atoms()

    n_warmup = 20
    n_collect = 180
    temperatures = [200.0, 400.0, 800.0, 1600.0]

    pt = CanonicalParallelTempering(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=temperatures,
        block_size=100,
        random_seed=0,
    )
    history = pt.run(n_cycles=n_warmup + n_collect)

    # ``history.energies_per_cycle`` has shape (n_cycles + 1, n_replicas).
    # Row 0 is the pre-run snapshot and is shared across all columns
    # (every replica starts from the same ``atoms``). Rows 1..n_cycles
    # are cycle endpoints. Column k is the energy sample stream seen at
    # ``temperatures[k]`` — the configuration in that column changes on
    # accepted exchanges, but the column is temperature-indexed.

    # 1. Show the cold-start descent at the coldest temperature.
    print(
        f"First 10 cycle-end energies at T = {temperatures[0]:.0f} K "
        "(row 0 is pre-run):"
    )
    for i, E in enumerate(history.energies_per_cycle[:10, 0]):
        marker = "  <- pre-run" if i == 0 else ""
        print(f"  cycle {i:>2}  E = {E:.4f} eV{marker}")
    print()

    # 2. Per-temperature mean energy, with and without the burn-in slice.
    #    ``naive`` drops only the pre-run row; ``equilibrium`` additionally
    #    drops the warm-up cycles. For a fast-equilibrating toy system the
    #    two will be close, but the pattern is the same at any scale:
    #    inspect the trajectory, pick a warm-up length that covers the
    #    transient, slice past it.
    naive = history.energies_per_cycle[1:, :]
    equilibrium = history.energies_per_cycle[1 + n_warmup:, :]

    print(
        f"Ran {n_warmup + n_collect} cycles "
        f"(burn-in {n_warmup}, collection {n_collect}).\n"
    )
    print("Per-temperature mean energy (eV):")
    print(f"  {'T (K)':>8}  {'naive':>12}  {'equilibrium':>14}  {'shift':>10}")
    for k, T in enumerate(temperatures):
        m_naive = naive[:, k].mean()
        m_eq = equilibrium[:, k].mean()
        print(
            f"  {T:>8.1f}  {m_naive:>12.4f}  {m_eq:>14.4f}  {m_eq - m_naive:>+10.4f}"
        )

    # 3. Swap acceptance per pair — an independent ladder-quality check.
    #    If any pair's rate drops to single percent, widen that gap in
    #    the temperature ladder; if they all sit above ~70 % the ladder
    #    is probably denser than it needs to be.
    print("\nPer-pair swap acceptance rates:")
    rates = history.swap_accepted / np.maximum(history.swap_attempted, 1)
    for i, rate in enumerate(rates):
        print(f"  pair ({i}, {i + 1}): {rate:.2%}")

    # Per-temperature energy histograms (canonical PT tuning diagnostic)
    # are a one-liner on the same equilibrium slice. If adjacent columns
    # do not overlap visibly, widen the temperature ladder:
    #
    #     import matplotlib.pyplot as plt
    #     for k, T in enumerate(temperatures):
    #         plt.hist(equilibrium[:, k], bins=30, alpha=0.4, label=f"{T} K")
    #     plt.xlabel("Energy (eV)")
    #     plt.legend()
    #     plt.show()


if __name__ == "__main__":
    main()
