"""Unit tests for WangLandauWindowGroup."""

from __future__ import annotations

import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _make_replicas(n: int = 2):
    """N WangLandauReplica instances sharing a wide window."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    return [
        WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=0.1,
            energy_limit_left=e0 - 100.0,
            energy_limit_right=e0 + 100.0,
            random_seed=i,
        )
        for i in range(n)
    ]


def test_sync_fill_factors_force_halves_lagging_replica():
    """A replica with fewer halvings is force-halved to match the most-halved one."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    # Replica 0: 2 halvings already done.
    replicas[0].ensemble._fill_factor = 0.25
    replicas[0].ensemble._fill_factor_history = [1.0, 0.5]
    replicas[0].ensemble._histogram.clear()
    # Replica 1: only 1 halving.
    replicas[1].ensemble._fill_factor = 0.5
    replicas[1].ensemble._fill_factor_history = [1.0]
    replicas[1].ensemble._histogram = {0: 10, 1: 8}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._sync_fill_factors()

    assert replicas[1].ensemble._fill_factor == pytest.approx(0.25)
    assert len(replicas[1].ensemble._fill_factor_history) == 2
    assert replicas[1].ensemble._fill_factor_history[1] == pytest.approx(0.5)
    assert replicas[1].ensemble._histogram == {}


def test_sync_fill_factors_noop_when_already_in_sync():
    """No mutation when all replicas have the same halving count."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._fill_factor = 0.5
        r.ensemble._fill_factor_history = [1.0]

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._sync_fill_factors()

    for r in replicas:
        assert r.ensemble._fill_factor == pytest.approx(0.5)
        assert len(r.ensemble._fill_factor_history) == 1


def test_merge_entropies_averages_all_bins():
    """Entropy is averaged bin-wise; unvisited bins treat as 0.0."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    replicas[0].ensemble._entropy = {0: 2.0, 1: 4.0}
    replicas[1].ensemble._entropy = {0: 6.0, 2: 8.0}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._merge_entropies()

    # bin 0: (2.0 + 6.0) / 2 = 4.0
    # bin 1: (4.0 + 0.0) / 2 = 2.0  (replica 1 never visited bin 1)
    # bin 2: (0.0 + 8.0) / 2 = 4.0  (replica 0 never visited bin 2)
    assert replicas[0].ensemble._entropy[0] == pytest.approx(4.0)
    assert replicas[0].ensemble._entropy[1] == pytest.approx(2.0)
    assert replicas[0].ensemble._entropy[2] == pytest.approx(4.0)
    assert replicas[1].ensemble._entropy == replicas[0].ensemble._entropy


def test_merge_entropies_noop_on_empty_entropies():
    """No KeyError or crash when all entropy dicts are empty."""
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    replicas = _make_replicas(2)
    for r in replicas:
        r.ensemble._entropy = {}

    group = WangLandauWindowGroup(replicas, random_seed=0)
    group._merge_entropies()

    for r in replicas:
        assert r.ensemble._entropy == {}
