"""End-to-end REWL correctness gate: recovers analytic 4x4 Ising DOS.

Runs REWL with multiple overlapping windows on the 4x4 2D Ising
system, stitches per-window ln g(E) curves via icet's
`get_density_of_states_wl`, and compares against the analytic DOS
computed by brute-force enumeration of all 2^16 = 65536 spin
configurations. After aligning the stitched curve to the exact one
by subtracting the mean residual (ln g is defined up to an
additive constant), the residuals must lie within a tolerance set
by the WL fill_factor_limit.

A regression in `_log_prob_ratio` (e.g. a sign error), in the WL
ensemble state-refresh on swap (`_potential`,
`_reached_energy_window`), or in how the orchestrator routes data
containers, would all push the stitched curve off the exact one
and this test would fail.

Marked `slow` because it converges two-to-four full WL ensembles.

Run manually:

    pytest tests/integration/test_rewl_2d_ising.py -v -m slow
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from ase import Atoms
from ase.symbols import symbols2numbers
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]
from mchammer.calculators import (  # type: ignore[import-untyped]
    ClusterExpansionCalculator,
)
from mchammer.data_containers.wang_landau_data_container import (  # type: ignore[import-untyped]
    get_density_of_states_wl,
)


def _build_4x4_ising() -> tuple[ClusterExpansion, Atoms]:
    """Build the 4x4 AFM 2D Ising cluster expansion and prototype.

    Same setup as icet's `WangLandauEnsemble` docstring example: a
    single-layer FCC with `J=2` nearest-neighbour pair ECI on a
    [Ag, Au] binary, repeated 4x4 in-plane.
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


def _exact_dos(
    ce: ClusterExpansion,
    prototype: Atoms,
    energy_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force enumeration of the 4x4 Ising DOS.

    Iterates over all 2^n_sites occupation vectors, evaluates the CE
    energy of each, and bins by `energy_spacing`. Returns
    `(energies, log_g)` arrays in ascending-energy order.
    """
    calc = ClusterExpansionCalculator(prototype.copy(), ce)
    n_sites = len(prototype)
    species_options = symbols2numbers(["Ag", "Au"])
    counts: dict[int, int] = {}
    for occupations in product(species_options, repeat=n_sites):
        e = float(calc.calculate_total(occupations=np.array(occupations)))
        bin_idx = int(round(e / energy_spacing))
        counts[bin_idx] = counts.get(bin_idx, 0) + 1
    bins = np.array(sorted(counts.keys()), dtype=int)
    energies = bins * energy_spacing
    log_g = np.log(np.array([counts[b] for b in bins], dtype=float))
    return energies, log_g


def _find_in_window_config(
    prototype: Atoms,
    ce: ClusterExpansion,
    energy_range: tuple[float | None, float | None],
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> Atoms:
    """Random-search a configuration whose energy falls in `energy_range`.

    Draws a random number of `Au` atoms (rest `Ag`) at random sites
    until the resulting energy lies in the requested window. Used to
    seed each REWL replica with a configuration its
    `_reached_energy_window` check accepts up front.
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


@pytest.mark.slow
def test_rewl_recovers_analytic_4x4_ising_dos() -> None:
    """Stitched REWL ln g(E) matches the exact 4x4 Ising DOS."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce, prototype = _build_4x4_ising()
    energy_spacing = 4.0

    # Exact DOS by brute enumeration (cheap: 2^16 ~ 65k configurations).
    exact_energies, exact_log_g = _exact_dos(ce, prototype, energy_spacing)

    # Two overlapping windows covering the negative-half and
    # positive-half of the energy range with a generous overlap
    # around zero. The physical energy range for the 4x4 AFM Ising
    # is [-32, 32]; `None` edges leave the outer sides unbounded so
    # icet's `WangLandauEnsemble` accepts any energy below the right
    # edge (window 0) or above the left edge (window 1). Overlap
    # [-12, 12] is 7 bins wide on the energy_spacing=4 grid.
    windows: list[tuple[float | None, float | None]] = [
        (None, 12.0),
        (-12.0, None),
    ]  # overlap [-12, 12]

    rng = np.random.default_rng(0)
    atoms_per_window = [
        _find_in_window_config(prototype, ce, w, rng) for w in windows
    ]

    # Use the base WangLandauEnsemble (halving-phase only). Its
    # convergence requires histogram flatness over the whole window
    # before each f halving, so both replicas must visit the overlap
    # by the time they converge. The 1/t variant declares convergence
    # on a step-count clock independent of coverage and is unsuitable
    # for a shape-comparison gate at this wall-time budget.
    #
    # `trial_move='flip'` (single-site composition-changing moves) is
    # needed here: the exact DOS we compare against enumerates *all*
    # 2^16 configurations across every composition, so the WL walkers
    # must also explore across compositions. The default 'swap' move
    # preserves composition and would lock each replica into the
    # handful of energies reachable from its seed composition.
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=atoms_per_window,
        windows=windows,
        energy_spacing=energy_spacing,
        block_size=len(prototype) * 200,
        random_seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={
            "fill_factor_limit": 1e-5,
            "flatness_limit": 0.7,
            "trial_move": "flip",
        },
    )
    pt.run(n_cycles=5000)
    assert pt.cycles_in_segment < 5000, (
        f"failed to converge within 5000 cycles "
        f"(got {pt.cycles_in_segment}); raise the cap or relax "
        f"fill_factor_limit (currently 1e-5)"
    )

    # Refresh each container's `_last_state` so the live container
    # carries the WL fields (`fill_factor`, `fill_factor_history`,
    # `entropy`, ...) that `get_density_of_states_wl` reads. mchammer
    # normally populates these inside `write_data_container`; the
    # `snapshot_for_checkpoint` path on `WangLandauReplica` does the
    # same refresh in place. Stitch via icet's combiner.
    pt.pool.snapshot_for_checkpoint()
    dcs = {i: r.data_container() for i, r in enumerate(pt.pool.replicas)}
    stitched_df, _stitch_errors = get_density_of_states_wl(dcs)

    # Map stitched (energy, entropy) to a dict for bin-aligned compare.
    stitched_log_g = dict(zip(
        np.round(
            stitched_df["energy"].to_numpy() / energy_spacing
        ).astype(int),
        stitched_df["entropy"].to_numpy(),
        strict=True,
    ))
    exact_bins = np.round(exact_energies / energy_spacing).astype(int)
    exact_log_g_by_bin = dict(zip(exact_bins, exact_log_g, strict=True))

    shared_bins = sorted(set(stitched_log_g) & set(exact_log_g_by_bin))
    assert len(shared_bins) >= 5, (
        f"too few shared bins between stitched and exact DOS: "
        f"{shared_bins}"
    )

    stitched = np.array([stitched_log_g[b] for b in shared_bins])
    exact = np.array([exact_log_g_by_bin[b] for b in shared_bins])

    # ln g is only defined up to an additive constant, so subtract
    # the mean residual before comparing the shapes.
    residual = stitched - exact
    residual_centred = residual - residual.mean()
    max_dev = float(np.max(np.abs(residual_centred)))
    std_dev = float(np.std(residual_centred))

    # Tolerances chosen so a sign-error or stale-_potential bug
    # would fail loudly (those produce per-bin discrepancies of
    # many ln-units); fill_factor_limit=1e-5 gives a converged-
    # estimate noise of ~few * 1e-5 per bin, so the std-dev of
    # the residual across the full range should be well under
    # 0.3. The max-abs bound allows a single noisy bin to drift
    # up to 1.0 ln-units.
    assert std_dev < 0.3, (
        f"REWL-recovered DOS shape diverges from analytic 4x4 Ising "
        f"DOS: residual std = {std_dev:.3f} ln-units across "
        f"{len(shared_bins)} bins. residual_centred = "
        f"{residual_centred.tolist()}"
    )
    assert max_dev < 1.0, (
        f"REWL-recovered DOS has a bin with large deviation from "
        f"analytic: max |residual| = {max_dev:.3f} ln-units. "
        f"residual_centred = {residual_centred.tolist()}"
    )
