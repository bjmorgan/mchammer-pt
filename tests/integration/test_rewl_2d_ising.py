"""End-to-end REWL correctness gate: overlap consistency.

Two adjacent windows that overlap should agree on ln g(E) within the
overlap region, up to a single additive constant (each window's WL
entropy is rebased periodically by icet). After alignment on a common
anchor bin, the two curves must match within a tolerance set by the
fill_factor_limit. This is the prerequisite the stitching procedure
exploits; testing it directly does not require a stitching helper.

A regression in `WangLandauParallelTempering._log_prob_ratio` (e.g. a
sign error) would bias replica i toward one half of the overlap and
replica j toward the other, and this test would diverge. Marked `slow`
because it spins up two full WL ensembles and runs them to convergence.

Run manually:

    pytest tests/integration/test_rewl_2d_ising.py -v -m slow
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]
from mchammer.calculators import (  # type: ignore[import-untyped]
    ClusterExpansionCalculator,
)


def _find_in_window_config(
    prototype: Atoms,
    ce: ClusterExpansion,
    species_pair: tuple[str, str],
    energy_range: tuple[float, float],
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> Atoms:
    """Random-search a configuration whose energy falls in `energy_range`.

    Draws a random number of `species_pair[1]` atoms (the rest `[0]`)
    and assigns them to random sites until the resulting energy lies
    in the requested range, or `max_tries` is exhausted.
    """
    species_a, species_b = species_pair
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
        if lo <= e <= hi:
            return atoms
    raise RuntimeError(
        f"could not find config in {energy_range} after {max_tries} tries"
    )


@pytest.mark.slow
def test_rewl_overlap_consistency():
    """Two overlapping windows agree on ln g(E) within the overlap.

    Catches sign errors and other systematic biases in
    `_log_prob_ratio`: a wrong sign would push replica 0's entropy
    one direction and replica 1's the other across the overlap.
    """
    from mchammer.ensembles import (  # type: ignore[import-untyped]
        WangLandauEnsemble,
    )

    from mchammer_pt.wl import WangLandauParallelTempering

    # 4x4 2D Ising on a single-layer FCC, AFM nearest-neighbour J=2.
    # Same setup as icet's WangLandauEnsemble docstring example.
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
    rng = np.random.default_rng(0)

    # Two overlapping windows. The 4x4 AFM Ising with J=2 has
    # reachable energies on a grid of spacing 4 (each pair-flip costs
    # one J-bond = 4 in our units), spanning [-32, 32]. We use
    # energy_spacing=4.0 to match that natural grid: any finer
    # produces large stretches of unreachable bins that the WL
    # histogram cannot flatten over.
    energy_spacing = 4.0
    windows = [(-32.0, 8.0), (-8.0, 32.0)]  # overlap is [-8, 8], 5 bins
    overlap_lo, overlap_hi = -8.0, 8.0

    # Per-window starting configurations whose energies fall in each window.
    atoms_per_window = [
        _find_in_window_config(prototype, ce, ("Ag", "Au"), w, rng)
        for w in windows
    ]

    # Use the base WangLandauEnsemble (halving-phase only). Its
    # convergence requires histogram flatness over the whole window
    # before each f halving, so both replicas must visit the overlap
    # by the time they converge — which is exactly the regime in
    # which overlap consistency is meaningful. The 1/t variant
    # declares convergence on a step-count clock independent of
    # coverage and is unsuitable for this gate.
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=atoms_per_window,
        windows=windows,
        energy_spacing=energy_spacing,
        block_size=len(prototype) * 200,
        random_seed=42,
        ensemble_cls=WangLandauEnsemble,
        ensemble_kwargs={"fill_factor_limit": 1e-3, "flatness_limit": 0.7},
    )
    pt.run(n_cycles=2000)
    assert pt.cycles_completed < 2000, (
        f"failed to converge within 2000 cycles "
        f"(got {pt.cycles_completed}); raise the cap or tighten the system"
    )

    # Extract overlap-region entropies from each replica.
    e0 = pt.pool.replicas[0].ensemble._entropy
    e1 = pt.pool.replicas[1].ensemble._entropy

    bin_lo = int(round(overlap_lo / energy_spacing))
    bin_hi = int(round(overlap_hi / energy_spacing))
    overlap_bins = list(range(bin_lo, bin_hi + 1))

    # Use only bins visited by BOTH replicas (each has a strict
    # overlap region but the histogram-flatness condition may not
    # fill every bin uniformly at this fill_factor_limit).
    shared_bins = [b for b in overlap_bins if b in e0 and b in e1]
    assert len(shared_bins) >= 4, (
        f"too few shared overlap bins: {shared_bins}. "
        f"Likely a setup problem (replicas did not visit the overlap)."
    )

    # Align on the shared bin with the lowest visit imbalance — any
    # shared bin works; pick the lowest energy in the shared set as
    # a deterministic anchor.
    anchor = min(shared_bins)
    delta0 = np.array([e0[b] - e0[anchor] for b in shared_bins])
    delta1 = np.array([e1[b] - e1[anchor] for b in shared_bins])
    max_disagreement = float(np.max(np.abs(delta0 - delta1)))

    # Tolerance: fill_factor_limit=1e-3 means each entropy update
    # adds at most ~1e-3, so noise in the converged estimate is
    # several times that per bin. A tolerance of 0.5 natural-log
    # units is generous; a wrong-sign bug would diverge by many
    # entropy units across the overlap and the test would fail
    # loudly.
    assert max_disagreement < 0.5, (
        f"REWL overlap-consistency violated: replica 0 and replica 1 "
        f"ln g(E) disagree by up to {max_disagreement:.3f} natural-log "
        f"units across the overlap region (anchor=bin {anchor}, "
        f"shared_bins={shared_bins}). "
        f"delta0={delta0.tolist()}, delta1={delta1.tolist()}. "
        f"A wrong sign in _log_prob_ratio would produce this."
    )
