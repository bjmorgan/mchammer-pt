"""Frozen-g observable measurement on top of REWL.

Run from the repo root:

    python examples/10_observable_measurement.py

First converges a tiny REWL density of states (as in ``08_rewl.py``), then
runs a FROZEN-G MEASUREMENT pass: ``g(E)`` is held fixed while the walkers
sample flat-in-energy and exchange, and a forwarded observer is evaluated
every ``interval`` steps and accumulated per energy bin. The per-walker
moments are then stitched into the microcanonical average ``<O>(E)`` and
reweighted to the canonical ``<O>(T)`` and the Binder cumulant ``U(T)`` --
here via the same console scripts a production workflow would use.

The observable is the Au fraction (a composition order parameter); in
production a downstream package supplies the real observer.

Completes in a couple of minutes on a laptop.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.observers.base_observer import BaseObserver

from mchammer_pt import WangLandauParallelTempering
from mchammer_pt.cli.reweight_observables import main as reweight_observables_cli
from mchammer_pt.cli.stitch import main as stitch_cli
from mchammer_pt.cli.stitch_observables import main as stitch_observables_cli


def build_ising_ce() -> tuple[ClusterExpansion, Atoms]:
    """4x4 single-layer AFM Ising: J=2 nearest-neighbour pair ECI."""
    primitive = Atoms("Au", positions=[[0.0, 0.0, 0.0]], cell=[1, 1, 10], pbc=True)
    cs = ClusterSpace(
        structure=primitive, cutoffs=[1.1], chemical_symbols=["Ag", "Au"]
    )
    ce = ClusterExpansion(cluster_space=cs, parameters=[0, 0, 2])
    return ce, primitive.repeat((4, 4, 1))


def find_in_window_config(
    prototype: Atoms,
    ce: ClusterExpansion,
    window: tuple[float | None, float | None],
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> Atoms:
    """Random-search for a configuration whose energy falls in ``window``."""
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
    raise RuntimeError(f"no config in {window} after {max_tries} tries")


class AuFractionObserver(BaseObserver):
    """Fraction of sites occupied by Au (atomic number 79)."""

    def __init__(self, interval: int = 10, tag: str = "au_fraction") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)

    def get_observable(self, structure: Atoms) -> float:
        return float(np.mean(structure.get_atomic_numbers() == 79))


def main() -> None:
    ce, prototype = build_ising_ce()
    energy_spacing = 4.0
    windows: list[tuple[float | None, float | None]] = [(None, 12.0), (-12.0, None)]
    rng = np.random.default_rng(0)
    atoms_per_window = [find_in_window_config(prototype, ce, w, rng) for w in windows]

    common = {
        "cluster_expansion": ce,
        "atoms": atoms_per_window,
        "windows": windows,
        "energy_spacing": energy_spacing,
        "block_size": len(prototype) * 200,
        "random_seed": 42,
        "ensemble_kwargs": {
            "fill_factor_limit": 1e-4,
            "flatness_limit": 0.7,
            "trial_move": "flip",
        },
    }
    tmp = Path(tempfile.mkdtemp())

    # 1. Converge g(E) -- the standard REWL run -- and checkpoint it.
    pt = WangLandauParallelTempering(**common)
    pt.run(n_cycles=5000)
    converged = tmp / "converged.h5"
    pt.save_checkpoint(converged)
    print(f"Converged g(E) after {pt.cycles_in_segment} cycles.")

    # 2. Frozen-g measurement pass: load the converged checkpoint (g frozen,
    #    coordinator off, exchanges on), attach the observer, and sample.
    meas = WangLandauParallelTempering.measure_from_checkpoint(
        converged, cluster_expansion=ce
    )
    meas.record_observable(AuFractionObserver(interval=10))
    meas.run(n_cycles=2000)
    measure_ckpt = tmp / "measure.h5"
    meas.save_checkpoint(measure_ckpt)
    print("Recorded <O>(E) moments during the frozen-g pass.")

    # 3. Post-process with the console scripts (the production workflow).
    #    The measurement checkpoint carries the frozen g(E) too, so a single
    #    file feeds both the DOS stitch and the observable stitch.
    dos_csv = tmp / "dos.csv"
    obs_dir = tmp / "observables"
    canonical = tmp / "au_fraction_canonical.csv"
    stitch_cli([str(measure_ckpt), "-o", str(dos_csv)])
    stitch_observables_cli([str(measure_ckpt), "-o", str(obs_dir)])
    reweight_observables_cli(
        [
            str(obs_dir / "au_fraction.csv"),
            str(dos_csv),
            "-o",
            str(canonical),
            "--T-min", "100", "--T-max", "1000", "--T-step", "50",
        ]
    )

    out = pd.read_csv(canonical)
    print("\nCanonical Au fraction and Binder cumulant:")
    print(f"  {'T (K)':>7s}  {'<O>(T)':>8s}  {'U(T)':>8s}")
    for _, row in out.iloc[:: max(1, len(out) // 10)].iterrows():
        print(
            f"  {row['T_K']:7.0f}  {row['au_fraction_mean']:8.4f}  "
            f"{row['au_fraction_binder']:8.4f}"
        )


if __name__ == "__main__":
    main()
