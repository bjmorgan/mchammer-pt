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


def test_wl_pt_run_finalises_pool_for_reporting_at_exit(monkeypatch):
    """``run`` calls ``pool.finalise_for_reporting`` exactly once at exit.

    Downstream consumers of the per-window data containers expect a
    consistent per-window entropy estimate regardless of the final
    block's halve state, so the orchestrator must merge walker
    entropies once before returning.
    """
    from unittest.mock import MagicMock

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
    mock = MagicMock()
    monkeypatch.setattr(pt.pool, "finalise_for_reporting", mock)
    pt.run(n_cycles=2)
    assert mock.call_count == 1


def test_wl_pt_run_finalises_pool_for_reporting_on_early_convergence(monkeypatch):
    """``run`` calls ``pool.finalise_for_reporting`` on the early-exit path.

    Convergence-triggered loop termination must still leave the
    per-window data containers in the merged state expected by
    ``results()`` and ``WindowResult`` consumers.
    """
    from unittest.mock import MagicMock

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
    for r in pt.pool.replicas:
        r.ensemble._converged = True
    mock = MagicMock()
    monkeypatch.setattr(pt.pool, "finalise_for_reporting", mock)
    pt.run(n_cycles=10)
    assert mock.call_count == 1


def test_wl_pt_run_finalises_pool_for_reporting_on_exception(monkeypatch):
    """``run`` finalises the pool even when a cycle raises.

    A mid-run failure (``KeyboardInterrupt`` from a notebook user, or
    an icet exception from a sweep) must not leave per-walker entropies
    unreconciled in the data containers — otherwise ``pt.results()``
    silently returns divergent values.
    """
    from unittest.mock import MagicMock

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
    mock_finalise = MagicMock()
    monkeypatch.setattr(pt.pool, "finalise_for_reporting", mock_finalise)

    def boom(n_steps):
        raise RuntimeError("synthetic test exception")

    monkeypatch.setattr(pt.pool, "advance_all", boom)

    with pytest.raises(RuntimeError, match="synthetic test exception"):
        pt.run(n_cycles=2)
    assert mock_finalise.call_count == 1


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
    assert pt.cycles_in_segment == 1


def test_wl_pt_early_convergence_passes_effective_n_to_callbacks():
    """Cycle callbacks receive effective_n = c + 1 on the convergence cycle."""
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
    for r in pt.pool.replicas:
        r.ensemble._converged = True

    received: list[tuple[int, int]] = []

    class Recorder:
        def on_cycle_end(self, cycle, n_cycles, history):
            received.append((cycle, n_cycles))

    pt.attach_cycle_callback(Recorder())
    pt.run(n_cycles=10)
    # Should converge after cycle 0; callback receives (0, 1) not (0, 10).
    assert len(received) == 1
    assert received[0] == (0, 1)


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


def test_wl_pt_resume_wraps_replicas_in_window_groups(tmp_path):
    """Resumed serial pool slots are WangLandauWindowGroup, not bare WangLandauReplica.

    Bare replica slots would never halve under the default
    CoordinatedWangLandauEnsemble (the coordinator drives halving),
    silently producing wrong WL output post-resume.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    e0 = _initial_energy()
    pt_a = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        data_container_file=str(tmp_path / "wl_ckpt.hdf5"),
    )
    pt_a.run(n_cycles=2)
    pt_a.save_checkpoint(tmp_path / "wl_ckpt.hdf5")

    pt_b = WangLandauParallelTempering.resume(
        tmp_path / "wl_ckpt.hdf5",
        cluster_expansion=make_wl_ce(),
    )
    for slot in pt_b.pool.replicas:
        assert isinstance(slot, WangLandauWindowGroup)
        assert len(slot._replicas) == 1


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


def test_wl_pt_checkpoint_preserves_flatness_mode_and_merge_cadence(tmp_path):
    """Checkpoint round-trip preserves non-default flatness_mode and merge_cadence."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="never",
    )
    pt.run(n_cycles=1)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    pt2 = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce()
    )
    assert pt2._flatness_mode == "per_walker"
    assert pt2._merge_cadence == "never"


def test_wl_pt_resume_falls_back_to_defaults_when_meta_lacks_new_keys(tmp_path):
    """Older checkpoints without the new keys resume with the default values."""
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        flatness_mode="per_walker",
        merge_cadence="never",
    )
    pt.run(n_cycles=1)
    # Simulate a pre-W1 checkpoint by omitting the two new keys from
    # the meta dict at save time. The reader's `.get(..., default)`
    # path must then resurface the defaults — not the saved values.
    original_meta = pt._checkpoint_meta()
    legacy_meta = {
        k: v for k, v in original_meta.items()
        if k not in ("flatness_mode", "merge_cadence")
    }
    pt._checkpoint_meta = lambda: legacy_meta  # type: ignore[method-assign]
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    pt2 = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce()
    )
    assert pt2._flatness_mode == "pooled"
    assert pt2._merge_cadence == "at_halve"


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


def test_wl_pt_resume_rejects_legacy_schema_2(tmp_path):
    """A schema-2 checkpoint (v0.7.0 era) cannot resume as a schema-3 reader."""
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
        f["meta"].attrs["schema_version"] = "2"
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
    """Resume with a different `ensemble_cls` than was used to save fails."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._ensemble_fixtures import TaggedWangLandauEnsemble

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        ensemble_cls=TaggedWangLandauEnsemble,
        ensemble_kwargs={"tag": "mismatch-test"},
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)
    # Original used TaggedWangLandauEnsemble; resume with the
    # default (base WangLandauEnsemble) should be rejected by the
    # FQN mismatch.
    with pytest.raises(ValueError, match="ensemble_cls FQN"):
        WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(),
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
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    e0 = _initial_energy()
    with WangLandauParallelTempering.process_pool(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={"fill_factor_limit": 1e-3},
    ) as pt:
        expected_fqn = (
            f"{CoordinatedWangLandauEnsemble.__module__}."
            f"{CoordinatedWangLandauEnsemble.__qualname__}"
        )
        assert pt._ensemble_cls_fqn == expected_fqn

        from mchammer_pt.checkpoint import _compute_ensemble_kwargs_hash
        empty_hash = _compute_ensemble_kwargs_hash({})
        assert pt._ensemble_kwargs_hash != empty_hash


def test_wl_pt_n_walkers_1_wraps_in_window_groups():
    """n_walkers_per_window=1 wraps each replica in a single-walker window group."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        n_walkers_per_window=1,
    )
    assert len(pt.pool) == 2
    for slot in pt.pool.replicas:
        assert isinstance(slot, WangLandauWindowGroup)
        assert len(slot._replicas) == 1


def test_wl_pt_n_walkers_2_creates_window_groups():
    """n_walkers_per_window=2 creates WangLandauWindowGroup slots with 2 replicas."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        n_walkers_per_window=2,
    )
    assert len(pt.pool) == 2
    assert isinstance(pt.pool.replicas[0], WangLandauWindowGroup)
    assert len(pt.pool.replicas[0]._replicas) == 2
    assert isinstance(pt.pool.replicas[1], WangLandauWindowGroup)
    assert len(pt.pool.replicas[1]._replicas) == 2


def test_wl_pt_n_walkers_2_rejects_data_container_file():
    """data_container_file with n_walkers_per_window > 1 raises NotImplementedError."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    with pytest.raises(NotImplementedError, match="checkpointing"):
        WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
            energy_spacing=0.1,
            block_size=5,
            random_seed=0,
            n_walkers_per_window=2,
            data_container_file="test.hdf5",
        )


def test_wl_pt_n_walkers_per_window_sequence_creates_window_groups():
    """n_walkers_per_window=[1, 2]: both windows wrap in WangLandauWindowGroup."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        n_walkers_per_window=[1, 2],
    )
    assert isinstance(pt.pool.replicas[0], WangLandauWindowGroup)
    assert len(pt.pool.replicas[0]._replicas) == 1
    assert isinstance(pt.pool.replicas[1], WangLandauWindowGroup)
    assert len(pt.pool.replicas[1]._replicas) == 2


def test_wl_pt_n_walkers_per_window_wrong_length_raises():
    """n_walkers_per_window sequence with wrong length raises ValueError."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    with pytest.raises(ValueError, match="n_walkers_per_window"):
        WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
            energy_spacing=0.1,
            block_size=5,
            random_seed=0,
            n_walkers_per_window=[2, 2, 2],
        )


def test_wl_pt_process_pool_accepts_n_walkers_per_window():
    """process_pool with n_walkers_per_window=2 returns a valid orchestrator."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering.process_pool(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        n_walkers_per_window=2,
    )
    try:
        assert isinstance(pt, WangLandauParallelTempering)
        assert len(pt.pool) == 2
        assert len(pt.pool._slots[0].workers) == 2
    finally:
        pt.pool.shutdown()


def test_wl_pt_n_walkers_2_run_returns_correct_history_shape():
    """A multi-walker run returns ExchangeHistory with the expected shape."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        n_walkers_per_window=2,
    )
    history = pt.run(n_cycles=3)
    # 2 windows, 3 cycles -> energies shape (4, 2), one pair -> swap_attempted (1,)
    assert history.energies_per_cycle.shape == (4, 2)
    assert history.swap_attempted.shape == (1,)


def test_wl_pt_process_pool_resume_round_trip_preserves_non_default_ensemble(
    tmp_path,
):
    """Full process_pool checkpoint/resume round-trip with non-default ensemble."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._ensemble_fixtures import TaggedWangLandauEnsemble

    e0 = _initial_energy()
    cp = tmp_path / "wl_ckpt.hdf5"

    with WangLandauParallelTempering.process_pool(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        ensemble_cls=TaggedWangLandauEnsemble,
        ensemble_kwargs={"tag": "round-trip-test"},
    ) as pt_a:
        pt_a.run(n_cycles=2)
        pt_a.save_checkpoint(cp)
        expected_fqn = pt_a._ensemble_cls_fqn
        expected_hash = pt_a._ensemble_kwargs_hash

    pt_b = WangLandauParallelTempering.resume_process_pool(
        cp,
        cluster_expansion=make_wl_ce(),
        ensemble_cls=TaggedWangLandauEnsemble,
        ensemble_kwargs={"tag": "round-trip-test"},
    )
    try:
        assert pt_b._ensemble_cls_fqn == expected_fqn
        assert pt_b._ensemble_kwargs_hash == expected_hash
        pt_b.run(n_cycles=1)
    finally:
        pt_b._pool.shutdown()


def test_wl_pt_results_returns_window_results():
    """results() returns one WindowResult per window with correct shape."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_result import WindowResult

    e0 = _initial_energy()
    windows = [
        (None, e0 + 50.0),
        (e0 - 50.0, e0 + 50.0),
        (e0 - 50.0, None),
    ]
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms() for _ in range(3)],
        windows=windows,
        energy_spacing=0.1,
        block_size=5,
        random_seed=42,
    )
    pt.run(n_cycles=2)
    results = pt.results()
    assert len(results) == 3
    for i, wr in enumerate(results):
        assert isinstance(wr, WindowResult)
        lo, hi = windows[i]
        expected_lo = float("-inf") if lo is None else lo
        expected_hi = float("inf") if hi is None else hi
        assert wr.energy_limit_left == expected_lo
        assert wr.energy_limit_right == expected_hi
        assert wr.energy_spacing == 0.1
        assert wr.n_walkers == 1


def test_wl_pt_results_multi_walker():
    """results() with W=2 returns WindowResults with 2 containers each."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_result import WindowResult

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=42,
        n_walkers_per_window=2,
    )
    pt.run(n_cycles=2)
    results = pt.results()
    assert len(results) == 2
    for wr in results:
        assert isinstance(wr, WindowResult)
        assert wr.n_walkers == 2
        assert len(wr.containers) == 2


def test_wl_pt_results_matches_data_containers_w1():
    """For W=1, results() entropy matches the underlying container."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=42,
    )
    pt.run(n_cycles=3)
    results = pt.results()
    for wr in results:
        wr_entropy = wr.get_entropy()
        if wr_entropy is not None and len(wr_entropy) > 0:
            # Single container, so merge is identity.
            assert "energy" in wr_entropy.columns
            assert "entropy" in wr_entropy.columns
            assert wr_entropy["entropy"].min() == 0.0  # min-shifted


def test_wl_pt_process_pool_rejects_multi_walker_with_checkpoint():
    """process_pool() rejects n_walkers > 1 with data_container_file before spawning."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    with pytest.raises(NotImplementedError, match="checkpointing"):
        WangLandauParallelTempering.process_pool(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
            energy_spacing=0.1,
            block_size=5,
            random_seed=0,
            n_walkers_per_window=2,
            data_container_file="test.hdf5",
        )


def test_wl_pt_serial_w1_slot_is_window_group():
    """Serial path always wraps replicas in WangLandauWindowGroup, even W=1."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

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
        block_size=5,
        random_seed=0,
        n_walkers_per_window=1,
        ensemble_cls=CoordinatedWangLandauEnsemble,
    )
    for slot in pt._pool.replicas:
        assert isinstance(slot, WangLandauWindowGroup)


def test_wl_pt_flatness_mode_default_pooled():
    """Default flatness_mode on the orchestrator is 'pooled'."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

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
        block_size=5,
        random_seed=0,
        ensemble_cls=CoordinatedWangLandauEnsemble,
    )
    assert pt._pool.replicas[0]._flatness_mode == "pooled"
    assert pt._pool.replicas[0]._merge_cadence == "at_halve"


def test_wl_pt_flatness_mode_per_walker_propagates():
    """flatness_mode='per_walker' reaches the window group."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

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
        block_size=5,
        random_seed=0,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        flatness_mode="per_walker",
    )
    assert pt._pool.replicas[0]._flatness_mode == "per_walker"


@pytest.mark.parametrize(
    "bad", ["per-walker", "Pooled", " pooled", "always", ""]
)
def test_wl_pt_flatness_mode_rejects_typos(bad):
    """Invalid flatness_mode values raise ValueError at construction."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    with pytest.raises(ValueError, match="flatness_mode"):
        WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[atoms, atoms],
            windows=[(e0 - 100.0, e0), (e0, e0 + 100.0)],
            energy_spacing=0.1,
            block_size=5,
            random_seed=0,
            ensemble_cls=CoordinatedWangLandauEnsemble,
            flatness_mode=bad,
        )


@pytest.mark.parametrize("flatness_mode", ["pooled", "per_walker"])
def test_wl_pt_w2_short_run_converges(flatness_mode):
    """W=2 short serial run produces valid WindowResult under both flatness modes."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

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
        block_size=200,
        random_seed=0,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        n_walkers_per_window=2,
        flatness_mode=flatness_mode,
    )
    pt.run(n_cycles=20)
    results = pt.results()
    assert len(results) == 2
    for r in results:
        df = r.get_entropy()
        assert df is not None
        assert len(df) > 0
        assert df["entropy"].notna().all()
        hist = r.get_histogram()
        assert hist is not None
        assert (hist["histogram"] > 0).any()


def test_wl_pt_w2_one_over_t_collective_phase():
    """W=2 with schedule='1_over_t': all walkers agree on phase post-run."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

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
        block_size=200,
        random_seed=0,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={
            "schedule": "1_over_t",
            "flatness_check_interval": 100,
        },
        n_walkers_per_window=2,
    )
    pt.run(n_cycles=20)
    for slot in pt._pool.replicas:
        phases = {r.ensemble._phase for r in slot._replicas}
        assert len(phases) == 1
        assert phases.pop() in {"halving", "1_over_t"}


def test_wl_pt_w1_unified_path_produces_finite_results():
    """W=1 through the unified WindowGroup coordinator produces valid output."""
    import numpy as np
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

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
        ensemble_cls=CoordinatedWangLandauEnsemble,
        n_walkers_per_window=1,
    )
    pt.run(n_cycles=10)
    results = pt.results()
    assert len(results) == 2
    for r in results:
        df = r.get_entropy()
        assert df is not None
        assert len(df) > 0
        assert np.all(np.isfinite(df["entropy"].to_numpy()))
