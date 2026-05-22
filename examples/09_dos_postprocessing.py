"""DOS post-processing on REWL output: stitch + canonical reweight.

Run from the repo root:

    python examples/09_dos_postprocessing.py

Builds the same 4x4 Ising fixture as ``08_rewl.py``, runs REWL to
convergence, stitches the per-window walker-merged entropies into a
single density of states via
``mchammer_pt.analysis.dos.stitch_entropy``, and evaluates canonical
thermodynamics across a temperature grid via
``mchammer_pt.analysis.dos.reweight_canonical_from_dos``.

For a command-line workflow, the ``mchammer-pt-stitch`` and
``mchammer-pt-reweight`` console scripts (installed with the package)
compose the same pipeline. ``mchammer-pt-stitch`` reads an mchammer-pt
checkpoint HDF5 (the artefact written by ``data_container_file=`` /
``save_checkpoint`` / ``CheckpointWriter``) by default, or
``WangLandauDataContainer`` files directly with ``--containers``:

    mchammer-pt-stitch run.h5 -o dos.csv
    mchammer-pt-stitch --containers window_*.dc -o dos.csv
    mchammer-pt-reweight dos.csv --T-min 50 --T-max 2000 --T-step 10

Completes in under a minute on a laptop.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace
from mchammer.calculators import ClusterExpansionCalculator

from mchammer_pt import WangLandauParallelTempering, WangLandauProgressPrinter
from mchammer_pt.analysis.dos import reweight_canonical_from_dos, stitch_entropy


def build_ising_ce() -> tuple[ClusterExpansion, Atoms]:
    """4x4 single-layer AFM Ising: J=2 nearest-neighbour pair ECI."""
    primitive = Atoms(
        "Au", positions=[[0.0, 0.0, 0.0]], cell=[1, 1, 10], pbc=True
    )
    cs = ClusterSpace(
        structure=primitive, cutoffs=[1.1], chemical_symbols=["Ag", "Au"],
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
    """Random-search for a configuration whose energy falls in `window`."""
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

    windows: list[tuple[float | None, float | None]] = [
        (None, 12.0),
        (-12.0, None),
    ]

    rng = np.random.default_rng(0)
    atoms_per_window = [
        find_in_window_config(prototype, ce, w, rng) for w in windows
    ]

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
    pt.attach_cycle_callback(WangLandauProgressPrinter(pt.pool, interval=100))
    pt.run(n_cycles=5000)
    print(f"\nConverged after {pt.cycles_in_segment} cycles.")

    # snapshot_for_checkpoint flushes live ensemble state into each
    # data container's _last_state. WindowResult.get_entropy reads
    # entropy and fill_factor_history from _last_state, so this call is
    # required for the per-window merge to see current values.
    pt.pool.snapshot_for_checkpoint()
    per_window = [r.get_entropy() for r in pt.results()]
    stitched, errors = stitch_entropy(per_window, energy_spacing)
    print(
        f"\nStitched DOS: {len(stitched)} bins; "
        f"max overlap std = {max(errors.values()):.3g}"
    )

    Ts = np.linspace(50.0, 2000.0, 196)
    canonical = reweight_canonical_from_dos(stitched, Ts)
    i_peak = int(np.argmax(canonical["Cv"].to_numpy()))
    T_peak = float(canonical["T_K"].iloc[i_peak])
    cv_peak = float(canonical["Cv"].iloc[i_peak])
    print(
        f"Canonical reweight: {len(canonical)} temperatures; "
        f"Cv peak {cv_peak:.3g} eV/K at T = {T_peak:.1f} K"
    )


if __name__ == "__main__":
    main()
