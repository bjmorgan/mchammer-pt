"""Fast validation tests for seed_window_configs (no pool is spawned).

The bad-configuration check raises during input validation, before any
cluster-expansion write or worker pool is created, so these run in the
fast suite.
"""

from __future__ import annotations

import pytest
from mchammer_moves import PairSwap

from mchammer_pt import SeedSearchParams, seed_window_configs
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def test_rejects_max_walks_below_max_count():
    ce = make_wl_ce()
    atoms = make_wl_atoms(n_au=16)
    with pytest.raises(ValueError, match="max_walks_per_window"):
        seed_window_configs(
            cluster_expansion=ce,
            moves=[(PairSwap(sublattice_index=0), 1.0)],
            windows=[(-10.0, 10.0)],
            counts=[3],
            energy_spacing=0.1,
            bottom_anchor=atoms,
            random_fill=lambda s: atoms,
            random_seed=0,
            params=SeedSearchParams(max_walks_per_window=2, n_workers=1),
        )
