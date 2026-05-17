"""Tests for ``mchammer_pt.contrib`` glue classes.

Each test skips when its third-party dependency is unavailable; the
module itself imports cleanly in either case.
"""

from __future__ import annotations

import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce

mchammer_moves = pytest.importorskip("mchammer_moves")


def test_coordinated_custom_wl_ensemble_mro_resolves_disjoint_overrides():
    """Coordinator's ``_update_entropy`` wins; custom moves' ``_do_trial_step`` wins."""
    from mchammer_moves import CustomWangLandauEnsemble  # noqa: F401

    from mchammer_pt.contrib import CoordinatedCustomWangLandauEnsemble
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    assert issubclass(
        CoordinatedCustomWangLandauEnsemble, CoordinatedWangLandauEnsemble
    )
    assert issubclass(
        CoordinatedCustomWangLandauEnsemble, CustomWangLandauEnsemble
    )
    update_owner = CoordinatedCustomWangLandauEnsemble._update_entropy.__qualname__
    assert "CoordinatedWangLandauEnsemble" in update_owner
    trial_owner = CoordinatedCustomWangLandauEnsemble._do_trial_step.__qualname__
    assert "CustomWangLandauEnsemble" in trial_owner


def test_coordinated_custom_wl_ensemble_runs_under_orchestrator():
    """Short serial run with PairSwap reaches results() with finite entropy."""
    import numpy as np
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer_moves import PairSwap

    from mchammer_pt.contrib import CoordinatedCustomWangLandauEnsemble
    from mchammer_pt.wl import WangLandauParallelTempering

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(e0 - 100.0, e0), (e0, e0 + 100.0)],
        energy_spacing=0.1,
        block_size=100,
        random_seed=0,
        ensemble_cls=CoordinatedCustomWangLandauEnsemble,
        ensemble_kwargs={"moves": [(PairSwap(), 1.0)]},
    )
    pt.run(n_cycles=5)
    results = pt.results()
    assert len(results) == 2
    for r in results:
        df = r.get_entropy()
        assert df is not None
        assert len(df) > 0
        assert np.all(np.isfinite(df["entropy"].to_numpy()))
