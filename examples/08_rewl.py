"""Replica-exchange Wang-Landau (REWL) on a toy 2D Ising model.

Run from the repo root:

    python examples/08_rewl.py

Builds a 4x4 single-layer Ising cluster expansion with a known
energy range, partitions the range into overlapping windows, seeds
each window with a random configuration whose energy falls inside
it, and runs REWL until all replicas converge. Stitches the
per-window ln g(E) curves via icet's `get_density_of_states_wl`
and prints the resulting density of states.

Completes in under a minute on a laptop.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.data_containers.wang_landau_data_container import (
    get_density_of_states_wl,
)

from mchammer_pt import ProgressPrinter, WangLandauParallelTempering


def build_ising_ce() -> tuple[ClusterExpansion, Atoms]:
    """4x4 single-layer AFM Ising: J=2 nearest-neighbour pair ECI."""
    primitive = Atoms(
        "Au", positions=[[0.0, 0.0, 0.0]], cell=[1, 1, 10], pbc=True
    )
    cs = ClusterSpace(
        structure=primitive,
        cutoffs=[1.1],
        chemical_symbols=["Ag", "Au"],
    )
    ce = ClusterExpansion(cluster_space=cs, parameters=[0, 0, 2])
    prototype = primitive.repeat((4, 4, 1))
    return ce, prototype


def find_in_window_config(
    prototype: Atoms,
    ce: ClusterExpansion,
    window: tuple[float | None, float | None],
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> Atoms:
    """Random-search for a configuration whose energy falls in `window`.

    Draws a random number of Au atoms at random sites until the
    resulting energy lies inside the requested window.
    """
    n_sites = len(prototype)
    calc = ClusterExpansionCalculator(prototype.copy(), ce)
    lo, hi = window
    for _ in range(max_tries):
        n_au = int(rng.integers(0, n_sites + 1))
        symbols = ["Ag"] * n_sites
        for i in rng.choice(n_sites, size=n_au, replace=False):
            symbols[i] = "Au"
        atoms = prototype.copy()
        atoms.set_chemical_symbols(symbols)
        e = float(calc.calculate_total(occupations=atoms.numbers))
        if (lo is None or e >= lo) and (hi is None or e <= hi):
            return atoms
    raise RuntimeError(
        f"could not find config in {window} after {max_tries} tries"
    )


def main() -> None:
    ce, prototype = build_ising_ce()
    energy_spacing = 4.0

    # Two overlapping windows covering negative and positive halves
    # of the energy range [-32, 32]. The outer edges are unbounded
    # (None) so the walkers can explore to the extremes; the overlap
    # around zero lets icet stitch the curves.
    windows: list[tuple[float | None, float | None]] = [
        (None, 12.0),
        (-12.0, None),
    ]

    # Seed each window with a configuration whose energy falls inside it.
    rng = np.random.default_rng(0)
    atoms_per_window = [
        find_in_window_config(prototype, ce, w, rng) for w in windows
    ]
    for i, (atoms, w) in enumerate(zip(atoms_per_window, windows, strict=True)):
        calc = ClusterExpansionCalculator(atoms.copy(), ce)
        e = float(calc.calculate_total(occupations=atoms.numbers))
        print(f"  window {i}: {w}  seed energy = {e:.1f}")

    # Build and run the REWL orchestrator. `trial_move='flip'` allows
    # composition-changing single-site moves so the walkers explore
    # the full energy range (the default 'swap' preserves composition).
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=atoms_per_window,
        windows=windows,
        energy_spacing=energy_spacing,
        block_size=len(prototype) * 200,
        random_seed=42,
        ensemble_kwargs={
            "fill_factor_limit": 1e-4,
            "flatness_limit": 0.7,
            "trial_move": "flip",
        },
    )
    pt.attach_cycle_callback(ProgressPrinter(interval=100))
    pt.run(n_cycles=5000)
    print(f"\nConverged after {pt.cycles_in_segment} cycles.")

    # Stitch the per-window ln g(E) into a single density of states.
    pt.pool.snapshot_for_checkpoint()
    dcs = {i: r.data_container() for i, r in enumerate(pt.pool.replicas)}
    df, errors = get_density_of_states_wl(dcs)

    print("\nStitched density of states:")
    print(f"  {'energy':>8s}  {'ln g(E)':>10s}")
    for _, row in df.iterrows():
        print(f"  {row['energy']:8.1f}  {row['entropy']:10.4f}")

    if errors:
        print(f"\nStitching errors: {errors}")


if __name__ == "__main__":
    main()
