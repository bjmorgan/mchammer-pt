"""Integration test: W=2 multi-walker REWL checkpoint and resume.

End-to-end behaviour pin: small 2D Ising W=2 multi-walker REWL run
that saves a checkpoint mid-run, resumes into a fresh orchestrator,
and continues to completion. Asserts structural correctness only —
W > 1 same-pool resume is not bit-identical because run()'s finally
merges per-walker entropies destructively. The relaxed contract is
documented in test_wl_pt_resume_w2_round_trips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]
from mchammer.calculators import (  # type: ignore[import-untyped]
    ClusterExpansionCalculator,
)


def _build_4x4_ising() -> tuple[ClusterExpansion, Atoms]:
    """Build the 4x4 AFM 2D Ising cluster expansion and prototype.

    Single-layer FCC with J=2 nearest-neighbour pair ECI on a
    [Ag, Au] binary, repeated 4x4 in-plane. Identical to the
    construction used in ``test_rewl_2d_ising.py``.
    """
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


def _find_in_window_config(
    prototype: Atoms,
    ce: ClusterExpansion,
    energy_range: tuple[float | None, float | None],
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> Atoms:
    """Random-search a configuration whose energy falls in ``energy_range``.

    Draws a random number of Au atoms (rest Ag) at random sites until
    the resulting energy lies in the requested window. Used to seed
    each REWL replica with a configuration its
    ``_reached_energy_window`` check accepts up front.
    """
    species_a, species_b = "Ag", "Au"
    n_sites = len(prototype)
    calc = ClusterExpansionCalculator(prototype.copy(), ce)
    lo, hi = energy_range
    for _ in range(max_tries):
        n_b = int(rng.integers(0, n_sites + 1))
        symbols = [species_a] * n_sites
        indices = rng.choice(n_sites, size=n_b, replace=False)
        for i in indices:
            symbols[i] = species_b
        atoms = prototype.copy()
        atoms.set_chemical_symbols(symbols)
        e = float(calc.calculate_total(occupations=atoms.numbers))
        if (lo is None or e >= lo) and (hi is None or e <= hi):
            return atoms
    raise RuntimeError(
        f"could not find config in {energy_range} after {max_tries} tries"
    )


def test_rewl_w2_checkpoint_resume_serial_pool(tmp_path: Path) -> None:
    """Small 2D Ising W=2 serial-pool checkpoint and resume.

    Runs 10 cycles, saves, resumes, runs 10 more. Asserts structural
    correctness only — pool shape, finite energies, non-None entropies.
    W > 1 resume is not bit-identical; see test_wl_pt_resume_w2_round_trips
    for the unit-level contract.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, prototype = _build_4x4_ising()
    rng = np.random.default_rng(0)

    # Two overlapping windows covering the low-energy and high-energy
    # halves of the 4x4 Ising range [-32, 32]. The overlap [-4, 4]
    # is one energy-spacing bin wide at energy_spacing=4, giving both
    # windows a shared region without confining walkers too tightly.
    windows: list[tuple[float | None, float | None]] = [
        (None, 4.0),
        (-4.0, None),
    ]
    atoms_per_window = [
        _find_in_window_config(prototype, ce, w, rng) for w in windows
    ]

    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=atoms_per_window,
        windows=windows,
        energy_spacing=4.0,
        block_size=len(prototype) * 10,
        random_seed=0,
        n_walkers_per_window=2,
        flatness_mode="pooled",
        merge_cadence="at_halve",
        ensemble_kwargs={"trial_move": "flip"},
    )
    pt.run(n_cycles=10)

    path = tmp_path / "rewl_w2.h5"
    pt.save_checkpoint(path)

    resumed = WangLandauParallelTempering.resume(
        path,
        cluster_expansion=ce,
        ensemble_kwargs={"trial_move": "flip"},
    )

    # Pool structure: 2 WangLandauWindowGroup slots, each with 2 walkers.
    assert len(resumed.pool) == 2
    for slot in resumed.pool.replicas:
        assert isinstance(slot, WangLandauWindowGroup)
        assert len(slot.walker_states) == 2

    history = resumed.run(n_cycles=10)

    # Structural correctness: shape, finite energies, non-None entropies.
    assert history.energies_per_cycle.shape == (11, 2)
    assert np.all(np.isfinite(resumed.pool.current_energies()))
    for wr in resumed.results():
        assert wr.get_entropy() is not None
