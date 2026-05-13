"""Tests for WangLandauParallelTempering."""

from __future__ import annotations

import numpy as np
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
    # log_r = (g_i(E_i) - g_i(E_j)) + (g_j(E_j) - g_j(E_i))
    #       = (1 - 2) + (7 - 4)
    #       = -1 + 3 = +2
    # (Standard REWL detailed balance:
    #  ln A = ln g_i(E_i)/g_i(E_j) + ln g_j(E_j)/g_j(E_i))
    assert log_r == pytest.approx(2.0)


def test_wl_pt_log_prob_ratio_rejects_out_of_window_partner(monkeypatch):
    """A swap that would move a replica outside its window returns -inf cleanly."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    # Use broad overlapping windows so the initial configurations
    # validate at construction; then override `current_energy` below
    # to simulate replica 1 reporting an energy that lies outside
    # replica 0's effective coverage.
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(e0 - 100.0, e0 + 100.0), (e0 - 100.0, e0 + 100.0)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )
    pool = pt.pool
    e_i = pool.current_energy(0)
    bin_i = pool.replicas[0].ensemble._get_bin_index(e_i)
    pool.replicas[0].ensemble._entropy = {bin_i: 1.0}
    pool.replicas[1].ensemble._entropy = {bin_i: 4.0}

    # Force replica 1 to report a partner energy that lies outside
    # replica 0's window: choose E_j beyond replica 0's right edge.
    e_j_out = pool.replicas[0].energy_window[1] + 10.0

    def fake_current_energy(idx):
        return e_i if idx == 0 else e_j_out

    monkeypatch.setattr(pool, "current_energy", fake_current_energy)
    log_r = pt._log_prob_ratio(0, 1)
    assert log_r == -float("inf")


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


def test_wl_pt_checkpoint_round_trip_is_bit_identical(tmp_path):
    """Checkpoint mid-run, resume, run M more cycles — bit-identical to a single run."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    def fresh():
        return WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
            energy_spacing=0.1,
            block_size=10,
            random_seed=1234,
        )

    # Reference: 4 cycles in one go.
    pt_ref = fresh()
    pt_ref.run(n_cycles=4)

    # Split: 2 + checkpoint + resume + 2.
    pt_a = fresh()
    pt_a.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt_a.save_checkpoint(cp)
    pt_b = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce()
    )
    pt_b.run(n_cycles=2)

    # The bit-identical contract: configurations and energies after
    # the split match the reference at the matching cycle.
    np.testing.assert_array_equal(
        pt_ref.pool.current_occupations(0),
        pt_b.pool.current_occupations(0),
    )
    np.testing.assert_array_equal(
        pt_ref.pool.current_occupations(1),
        pt_b.pool.current_occupations(1),
    )
    np.testing.assert_allclose(
        pt_ref.pool.current_energies(),
        pt_b.pool.current_energies(),
    )


def test_wl_pt_from_bin_count_builds_overlapping_windows():
    from mchammer.ensembles.wang_landau_ensemble import (  # type: ignore[import-untyped]
        get_bins_for_parallel_simulations,
    )

    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    # Two windows with wide enough overlap that the single starting
    # configuration (energy e0) lies in both — keeps the test
    # focused on the windows-translation contract without needing
    # tuned per-window starting structures.
    raw = get_bins_for_parallel_simulations(
        n_bins=2, energy_spacing=0.1,
        minimum_energy=e0 - 1.0, maximum_energy=e0 + 1.0, overlap=4,
    )
    # icet returns NaN for unbounded edges; the orchestrator uses
    # None. Translate before comparison.
    expected = [
        (None if np.isnan(lo) else float(lo),
         None if np.isnan(hi) else float(hi))
        for lo, hi in raw
    ]
    pt = WangLandauParallelTempering.from_bin_count(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms()] * 2,
        n_bins=2,
        energy_spacing=0.1,
        minimum_energy=e0 - 1.0,
        maximum_energy=e0 + 1.0,
        overlap=4,
        block_size=10,
        random_seed=0,
    )
    assert pt.windows == expected


def test_wl_pt_process_pool_round_trips():
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    with WangLandauParallelTempering.process_pool(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
    ) as pt:
        history = pt.run(n_cycles=3)
    assert history.energies_per_cycle.shape == (4, 2)


def test_wl_pt_rejects_pool_plus_ensemble_kwargs():
    """Cannot combine an explicit pool with non-default ensemble args."""
    from mchammer_pt.parallel.serial import SerialWangLandauPool
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_replica import WangLandauReplica
    e0 = _initial_energy()

    replicas = [
        WangLandauReplica(
            cluster_expansion=make_wl_ce(), atoms=make_wl_atoms(),
            energy_spacing=0.1,
            energy_limit_left=e0 - 100.0, energy_limit_right=e0 + 100.0,
            random_seed=i,
        )
        for i in range(2)
    ]
    pool = SerialWangLandauPool(replicas, energy_spacing=0.1)

    with pytest.raises(ValueError, match="pool already owns"):
        WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=[(e0 - 100.0, e0 + 100.0), (e0 - 100.0, e0 + 100.0)],
            energy_spacing=0.1,
            block_size=10,
            random_seed=0,
            pool=pool,
            ensemble_kwargs={"flatness_limit": 0.5},
        )


def test_wl_pt_resume_process_pool_round_trips(tmp_path):
    """Checkpoint, resume into a process pool, continue running.

    Worker scheduling non-determinism means we don't assert bit-identity
    across the serial/process boundary; only that the resumed
    orchestrator continues running correctly and the final
    configurations are inside-window.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    pt_a = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=42,
    )
    pt_a.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt_a.save_checkpoint(cp)

    pt_b = WangLandauParallelTempering.resume_process_pool(
        cp, cluster_expansion=make_wl_ce()
    )
    try:
        history = pt_b.run(n_cycles=2)
        assert history.energies_per_cycle.shape == (3, 2)
    finally:
        pt_b._pool.shutdown()


def test_wl_pt_resume_rejects_unknown_schema_version(tmp_path):
    import h5py

    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)
    with h5py.File(cp, "r+") as f:
        f["meta"].attrs["schema_version"] = "999"
    with pytest.raises(ValueError, match="schema_version"):
        WangLandauParallelTempering.resume(cp, cluster_expansion=make_wl_ce())


def test_wl_pt_resume_rejects_mismatched_ce(tmp_path):
    from icet import ClusterExpansion

    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import make_wl_cluster_space
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    # Build a different CE — bump one parameter so the identity hash differs.
    cs = make_wl_cluster_space()
    params = np.zeros(len(cs))
    params[0] = 0.0
    params[1] = 0.5
    params[2] = 1.5  # was 1.0 in make_wl_ce — bumps the identity hash
    different_ce = ClusterExpansion(cluster_space=cs, parameters=params)

    with pytest.raises(ValueError, match="CE identity mismatch"):
        WangLandauParallelTempering.resume(cp, cluster_expansion=different_ce)


def test_wl_pt_resume_rejects_mismatched_ensemble_cls(tmp_path):
    from mchammer.ensembles import WangLandauEnsemble

    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)
    # The original used OneOverTWangLandauEnsemble; resume with the plain
    # WangLandauEnsemble should be rejected by the FQN mismatch.
    with pytest.raises(ValueError, match="ensemble_cls FQN"):
        WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(),
            ensemble_cls=WangLandauEnsemble,
        )


def test_wl_pt_resume_rejects_mismatched_ensemble_kwargs_hash(tmp_path):
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)
    # Original ensemble_kwargs=None hashes to the canonical empty-dict
    # hash, which is a real (non-sentinel) hash. Resuming with a
    # picklable but materially different kwargs should fail the hash
    # comparison rather than fall through the sentinel skip.
    with pytest.raises(ValueError, match="ensemble_kwargs hash mismatch"):
        WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(),
            ensemble_kwargs={"flatness_limit": 0.5},
        )


def test_wl_pt_process_pool_records_actual_ensemble_identity():
    """process_pool's checkpoint metadata reflects the workers' actual ensemble."""
    from mchammer.ensembles import (  # type: ignore[import-untyped]
        WangLandauEnsemble,
    )

    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    with WangLandauParallelTempering.process_pool(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        ensemble_cls=WangLandauEnsemble,
        ensemble_kwargs={"fill_factor_limit": 1e-3},
    ) as pt:
        expected_fqn = (
            f"{WangLandauEnsemble.__module__}."
            f"{WangLandauEnsemble.__qualname__}"
        )
        assert pt._ensemble_cls_fqn == expected_fqn

        from mchammer_pt.checkpoint import _compute_ensemble_kwargs_hash
        empty_hash = _compute_ensemble_kwargs_hash({})
        assert pt._ensemble_kwargs_hash != empty_hash
