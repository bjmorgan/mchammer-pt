"""Tests for WangLandauParallelTempering."""

from __future__ import annotations

import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _initial_energy():
    from mchammer.calculators import ClusterExpansionCalculator
    return float(
        ClusterExpansionCalculator(make_wl_atoms(), make_wl_ce())
        .calculate_total(occupations=make_wl_atoms().numbers)
    )


def test_wl_pt_constructs_with_two_windows():
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )
    assert len(pt.pool) == 2
    assert pt.pool.windows == [(None, e0 + 50.0), (e0 - 50.0, None)]
    assert pt.pool.energy_spacing == 0.1


def test_wl_pt_rejects_single_atoms_no_broadcast():
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    with pytest.raises(TypeError, match="sequence of Atoms"):
        WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=make_wl_atoms(),
            windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
            energy_spacing=0.1,
            block_size=10,
            random_seed=0,
        )


def test_wl_pt_log_prob_ratio_identical_states_gives_zero():
    """When both replicas hold identical configurations, log_r = 0 by symmetry."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )
    pool = pt.pool
    bin0 = pool.replicas[0].ensemble._get_bin_index(pool.current_energy(0))
    # Asymmetric entropies; identical configurations -> the four lookups
    # collapse to two within-bin values which subtract to zero per replica.
    pool.replicas[0].ensemble._entropy = {bin0: 3.0}
    pool.replicas[1].ensemble._entropy = {bin0: 5.0}
    log_r = pt._log_prob_ratio(0, 1)
    # (g_i_Ej - g_i_Ei) + (g_j_Ei - g_j_Ej) = (3 - 3) + (5 - 5) = 0
    assert log_r == pytest.approx(0.0)


def test_wl_pt_log_prob_ratio_uses_cross_bin_entropies(monkeypatch):
    """Distinct bins on the two replicas exercise the full four-term formula."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )
    pool = pt.pool
    e_i = pool.current_energy(0)
    bin_i = pool.replicas[0].ensemble._get_bin_index(e_i)
    e_j_fake = e_i + 1.0  # 10 bins above e_i (spacing=0.1)
    bin_j = pool.replicas[0].ensemble._get_bin_index(e_j_fake)
    assert bin_j != bin_i

    pool.replicas[0].ensemble._entropy = {bin_i: 1.0, bin_j: 2.0}
    pool.replicas[1].ensemble._entropy = {bin_i: 4.0, bin_j: 7.0}

    def fake_current_energy(idx):
        return e_i if idx == 0 else e_j_fake

    monkeypatch.setattr(pool, "current_energy", fake_current_energy)
    log_r = pt._log_prob_ratio(0, 1)
    # (g_i(E_j) - g_i(E_i)) + (g_j(E_i) - g_j(E_j))
    # = (2 - 1) + (4 - 7) = 1 + (-3) = -2
    assert log_r == pytest.approx(-2.0)


def test_wl_pt_run_returns_history_with_expected_shape():
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=20,
        random_seed=0,
    )
    history = pt.run(n_cycles=5)
    assert history.energies_per_cycle.shape == (6, 2)
    assert history.replica_labels_per_cycle.shape == (6, 2)
    assert history.swap_attempted.shape == (1,)


def test_wl_pt_run_stops_on_all_converged():
    """If every replica reports converged, the loop terminates early."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=1,
        random_seed=0,
    )
    # Force convergence after the first cycle by setting `_converged`
    # on each underlying ensemble.
    for r in pt.pool.replicas:
        r.ensemble._converged = True
    pt.run(n_cycles=10)
    # Run should bail out after the first cycle's converged_flags check.
    assert pt.cycles_completed == 1
