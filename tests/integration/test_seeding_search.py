"""End-to-end (slow) tests for seed_window_configs on the toy CE."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import PairSwap

from mchammer_pt import (
    SeedSearchParams,
    WangLandauParallelTempering,
    seed_window_configs,
)
from tests._wl_fixtures import make_wl_atoms, make_wl_ce

pytestmark = pytest.mark.slow

_N_AU = 16  # half-filled 32-site supercell; fixed composition


def _energy(ce, atoms: Atoms) -> float:
    return float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )


def _random_fill_factory(template: Atoms):
    """Return a seed -> Atoms factory producing random _N_AU-Au configs."""
    n_sites = len(template)

    def random_fill(seed: int) -> Atoms:
        rng = np.random.default_rng(seed)
        atoms = template.copy()
        symbols = ["Ag"] * n_sites
        for i in rng.choice(n_sites, size=_N_AU, replace=False):
            symbols[i] = "Au"
        atoms.set_chemical_symbols(symbols)
        return atoms

    return random_fill


def _scenario():
    """Build a toy CE, a low anchor, a fill factory, and three windows."""
    ce = make_wl_ce()
    template = make_wl_atoms(n_au=_N_AU)
    random_fill = _random_fill_factory(template)

    # Sample energies to size windows and find a low anchor.
    energies = []
    configs = []
    for s in range(60):
        a = random_fill(s)
        energies.append(_energy(ce, a))
        configs.append(a)
    energies = np.array(energies)
    e_lo, e_hi = float(energies.min()), float(energies.max())
    bottom_anchor = configs[int(energies.argmin())]

    span = e_hi - e_lo
    # Three overlapping windows across the sampled range.
    windows = [
        (e_lo - 0.1, e_lo + 0.45 * span),
        (e_lo + 0.30 * span, e_lo + 0.70 * span),
        (e_lo + 0.55 * span, e_hi + 0.1),
    ]
    return ce, bottom_anchor, random_fill, windows


def test_fills_all_windows_distinct_and_in_band():
    ce, bottom_anchor, random_fill, windows = _scenario()
    counts = [2, 2, 2]  # K>1 everywhere, including the low window
    params = SeedSearchParams(
        walk_sweeps=40, max_walks_per_window=12, n_workers=2
    )

    result = seed_window_configs(
        cluster_expansion=ce,
        moves=[(PairSwap(sublattice_index=0), 1.0)],
        windows=windows,
        counts=counts,
        energy_spacing=0.1,
        bottom_anchor=bottom_anchor,
        random_fill=random_fill,
        random_seed=7,
        params=params,
    )

    assert [len(w) for w in result] == counts
    for (lo, hi), per_window in zip(windows, result, strict=True):
        keys = set()
        for atoms in per_window:
            e = _energy(ce, atoms)
            assert lo <= e <= hi
            keys.add(atoms.numbers.tobytes())
        assert len(keys) == len(per_window)  # distinct within window

    # The result drops straight into a process-pool REWL run.
    pt = WangLandauParallelTempering.process_pool(
        cluster_expansion=ce,
        atoms=result,
        windows=windows,
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=counts,
    )
    try:
        pt.run(1)
    finally:
        pt._pool.shutdown()


def test_unfillable_window_raises_naming_it():
    ce, bottom_anchor, random_fill, windows = _scenario()
    # Append a window above any achievable energy: cannot be filled.
    windows = list(windows) + [(1.0e6, 1.0e6 + 1.0)]
    counts = [1, 1, 1, 1]
    params = SeedSearchParams(
        walk_sweeps=20, max_walks_per_window=2, n_workers=2
    )

    with pytest.raises(RuntimeError, match=r"3"):  # names window index 3
        seed_window_configs(
            cluster_expansion=ce,
            moves=[(PairSwap(sublattice_index=0), 1.0)],
            windows=windows,
            counts=counts,
            energy_spacing=0.1,
            bottom_anchor=bottom_anchor,
            random_fill=random_fill,
            random_seed=7,
            params=params,
        )
