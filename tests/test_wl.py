"""Tests for WangLandauParallelTempering."""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.wl import WangLandauParallelTempering
from tests._wl_fixtures import (
    distinct_in_window_pair,
    make_wl_atoms,
    make_wl_ce,
)


def _initial_energy():
    from mchammer.calculators import ClusterExpansionCalculator
    return float(
        ClusterExpansionCalculator(make_wl_atoms(), make_wl_ce())
        .calculate_total(occupations=make_wl_atoms().numbers)
    )


def _make_serial_wl_pt(**overrides):
    """Smallest valid two-window serial WangLandauParallelTempering.

    Keyword ``overrides`` are merged into the constructor kwargs, so a test
    can flip ``one_over_t_gate`` / ``bp_stall_multiple`` / ``ensemble_kwargs``
    / ``block_size`` without restating the rest.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    lo, hi = e0 - 50.0, e0 + 50.0
    kwargs = dict(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )
    kwargs.update(overrides)
    return WangLandauParallelTempering(**kwargs)


class TestOrchestratorOneOverTGate:
    def test_constructor_stores_and_forwards(self) -> None:
        pt = _make_serial_wl_pt(
            one_over_t_gate="flatness",
            bp_stall_multiple=3.0,
            ensemble_kwargs={"schedule": "1_over_t"},
        )
        assert pt._one_over_t_gate == "flatness"
        assert pt._bp_stall_multiple == 3.0
        # forwarded to the default serial pool
        assert pt._pool._one_over_t_gate == "flatness"
        assert pt._pool._bp_stall_multiple == 3.0

    def test_constructor_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError, match="one_over_t_gate"):
            _make_serial_wl_pt(one_over_t_gate="bogus")
        with pytest.raises(ValueError, match="bp_stall_multiple"):
            _make_serial_wl_pt(bp_stall_multiple=-1.0)

    def test_checkpoint_meta_round_trips_policy(self) -> None:
        pt = _make_serial_wl_pt(
            one_over_t_gate="flatness",
            bp_stall_multiple=3.0,
            ensemble_kwargs={"schedule": "1_over_t"},
        )
        meta = pt._checkpoint_meta()
        assert meta["one_over_t_gate"] == "flatness"
        assert meta["bp_stall_multiple"] == 3.0

    def test_flatness_gate_without_one_over_t_schedule_raises(self) -> None:
        # The flatness gate is inert under the (default) halving schedule;
        # selecting it without schedule="1_over_t" is a silent no-op, so
        # construction must reject it.
        with pytest.raises(ValueError, match="1/t schedule"):
            _make_serial_wl_pt(one_over_t_gate="flatness")

    def test_explicit_pool_policy_is_source_of_truth(self) -> None:
        # With an explicit pool=, the orchestrator (and its checkpoint meta)
        # must reflect the POOL's policy, not the constructor's default args.
        # Otherwise a run governed by the pool checkpoints a different policy
        # and resume silently diverges.
        from mchammer.calculators import ClusterExpansionCalculator

        from mchammer_pt.parallel.serial import SerialWangLandauPool
        from mchammer_pt.wl import WangLandauParallelTempering
        from mchammer_pt.wl_replica import WangLandauReplica

        ce, atoms = make_wl_ce(), make_wl_atoms()
        e0 = float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )
        lo, hi = e0 - 50.0, e0 + 50.0
        replicas = [
            WangLandauReplica(
                cluster_expansion=ce, atoms=atoms, energy_spacing=0.1,
                energy_limit_left=lo, energy_limit_right=hi, random_seed=s,
                ensemble_kwargs={"schedule": "1_over_t"},
            )
            for s in (0, 1)
        ]
        pool = SerialWangLandauPool(
            replicas, energy_spacing=0.1,
            flatness_mode="per_walker", merge_cadence="never",
            one_over_t_gate="flatness", bp_stall_multiple=2.5,
        )
        # The orchestrator receives the pool but only DEFAULT policy args.
        pt = WangLandauParallelTempering(
            cluster_expansion=ce, atoms=[atoms, atoms],
            windows=[(lo, hi), (lo, hi)], energy_spacing=0.1,
            block_size=10, random_seed=0, pool=pool,
        )
        assert pt._one_over_t_gate == "flatness"
        assert pt._bp_stall_multiple == 2.5
        assert pt._flatness_mode == "per_walker"
        assert pt._merge_cadence == "never"
        meta = pt._checkpoint_meta()
        assert meta["one_over_t_gate"] == "flatness"
        assert meta["bp_stall_multiple"] == 2.5
        assert meta["flatness_mode"] == "per_walker"
        assert meta["merge_cadence"] == "never"


class TestOrchestratorOneOverTEntry:
    def test_constructor_stores_and_forwards_to_replicas(self) -> None:
        pt = _make_serial_wl_pt(
            one_over_t_entry="f_continuous",
            ensemble_kwargs={"schedule": "1_over_t"},
        )
        assert pt._one_over_t_entry == "f_continuous"
        for slot in pt._pool.replicas:
            assert slot.one_over_t_entry == "f_continuous"

    def test_default_is_window_clock(self) -> None:
        pt = _make_serial_wl_pt()
        assert pt._one_over_t_entry == "window_clock"
        assert pt._checkpoint_meta()["one_over_t_entry"] == "window_clock"

    def test_constructor_rejects_bad_value(self) -> None:
        with pytest.raises(ValueError, match="one_over_t_entry"):
            _make_serial_wl_pt(one_over_t_entry="bogus")

    def test_f_continuous_without_one_over_t_schedule_raises(self) -> None:
        with pytest.raises(ValueError, match="1/t schedule"):
            _make_serial_wl_pt(one_over_t_entry="f_continuous")

    def test_checkpoint_meta_round_trips_policy(self) -> None:
        pt = _make_serial_wl_pt(
            one_over_t_entry="f_continuous",
            ensemble_kwargs={"schedule": "1_over_t"},
        )
        assert pt._checkpoint_meta()["one_over_t_entry"] == "f_continuous"


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
    bin0 = pool.replicas[0].ensemble._get_bin_index(pool.walker_energy(0, 0))
    # Asymmetric entropies; identical configurations -> the four lookups
    # collapse to two within-bin values which subtract to zero per replica.
    pool.replicas[0].ensemble._entropy = {bin0: 3.0}
    pool.replicas[1].ensemble._entropy = {bin0: 5.0}
    log_r = pt._log_prob_ratio(0, 0, 1, 0)
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
    e_i = pool.walker_energy(0, 0)
    bin_i = pool.replicas[0].ensemble._get_bin_index(e_i)
    e_j_fake = e_i + 1.0  # 10 bins above e_i (spacing=0.1)
    bin_j = pool.replicas[0].ensemble._get_bin_index(e_j_fake)
    assert bin_j != bin_i

    pool.replicas[0].ensemble._entropy = {bin_i: 1.0, bin_j: 2.0}
    pool.replicas[1].ensemble._entropy = {bin_i: 4.0, bin_j: 7.0}

    def fake_walker_energy(idx, walker):
        return e_i if idx == 0 else e_j_fake

    monkeypatch.setattr(pool, "walker_energy", fake_walker_energy)
    log_r = pt._log_prob_ratio(0, 0, 1, 0)
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
    e_i = pool.walker_energy(0, 0)
    bin_i = pool.replicas[0].ensemble._get_bin_index(e_i)
    pool.replicas[0].ensemble._entropy = {bin_i: 1.0}
    pool.replicas[1].ensemble._entropy = {bin_i: 4.0}

    # Force replica 1 to report a partner energy that lies outside
    # replica 0's window: choose E_j beyond replica 0's right edge.
    e_j_out = pool.replicas[0].energy_window[1] + 10.0

    def fake_walker_energy(idx, walker):
        return e_i if idx == 0 else e_j_out

    monkeypatch.setattr(pool, "walker_energy", fake_walker_energy)
    log_r = pt._log_prob_ratio(0, 0, 1, 0)
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


def test_wl_pt_run_skips_finalise_when_pool_shuts_down_on_exception(monkeypatch):
    """If the pool shuts down on exception, ``run`` does not call finalise.

    ``ProcessWangLandauPool.advance_all`` shuts the pool down on worker
    errors before propagating. The original exception is what the user
    wants to see — not the secondary ``RuntimeError("pool is shut
    down")`` that ``finalise_for_reporting`` would produce on a closed
    pool. Gated on the new ``pool.is_open`` property.
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

    def boom(n_steps):
        # Simulate ProcessWangLandauPool.advance_all: shut the pool down
        # before propagating the worker error. ``shutdown`` itself is a
        # no-op on SerialWangLandauPool, so we also flip ``is_open`` via
        # a monkeypatched property to mirror process-pool semantics.
        pt.pool.shutdown()
        raise RuntimeError("worker died")

    monkeypatch.setattr(pt.pool, "advance_all", boom)
    monkeypatch.setattr(
        type(pt.pool), "is_open", property(lambda self: False)
    )
    finalise_mock = MagicMock()
    monkeypatch.setattr(pt.pool, "finalise_for_reporting", finalise_mock)

    with pytest.raises(RuntimeError, match="worker died"):
        pt.run(n_cycles=2)
    finalise_mock.assert_not_called()


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


def test_wl_pt_resume_slots_are_bare_replicas(tmp_path):
    """Resumed serial pool slots are bare WangLandauReplica."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_replica import WangLandauReplica

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
        assert isinstance(slot, WangLandauReplica)


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
        recency_visits_per_bin=250,
    )
    pt_a.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt_a.save_checkpoint(cp)

    pt_b = WangLandauParallelTempering.resume_process_pool(
        cp, cluster_expansion=make_wl_ce()
    )
    try:
        # recency_visits_per_bin is read from meta and threaded into the
        # process pool reconstruction without the caller re-specifying it.
        assert pt_b._recency_visits_per_bin == 250
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


def test_resume_threads_recency_visits_per_bin_from_metadata(tmp_path):
    """Resume reads recency_visits_per_bin from meta, not the caller.

    The EWMA per-bin weights are not persisted; the diagnostic
    re-accumulates from empty state on resume. For that re-accumulation
    to use the same timescale it had before the interruption, resume
    must thread the saved ``recency_visits_per_bin`` into the
    reconstructed ensembles and orchestrator without the caller
    re-specifying it.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        recency_visits_per_bin=250,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    resumed = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce()
    )
    for slot in resumed.pool.replicas:
        assert slot.ensemble._recency_visits_per_bin == 250
    assert resumed._recency_visits_per_bin == 250


def test_resume_does_not_persist_recency_ewma_state(tmp_path):
    """The EWMA recency state re-accumulates from empty on resume.

    Unlike ``_visited_bins`` and the entropy/histogram (which are
    persisted and restored), the per-bin EWMA weights are deliberately
    not checkpointed. After a resume the state must be empty so the
    diagnostic re-accumulates from scratch; ``recency_flatness`` reads
    ``None`` until the first post-resume visit. This guards against a
    future change that mistakenly persists ``_recent_weight`` by
    analogy with the restored state.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        recency_visits_per_bin=250,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    resumed = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce()
    )
    # No MC has run since resume, so the EWMA state is genuinely empty.
    for slot in resumed.pool.replicas:
        assert slot.ensemble._recent_weight == {}
        assert slot.ensemble._recent_last_step == {}
    # recency_flatness is None until the first post-resume visit.
    for s in resumed.pool.per_window_stats():
        assert s["recency_flatness"] is None


def test_resume_defaults_recency_visits_per_bin_for_pre_feature_checkpoint(
    tmp_path,
):
    """A checkpoint lacking the meta key resumes with the 1000 default.

    Checkpoints written before ``recency_visits_per_bin`` was recorded
    have no such entry in ``/meta``. Resume must fall back to
    ``meta.get(..., 1000)`` rather than raising a ``KeyError``. The
    meta is stored as HDF5 attributes on the ``meta`` group (see
    ``history.write_hdf5``), so deleting the attribute simulates such
    a file.
    """
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
        recency_visits_per_bin=250,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    with h5py.File(cp, "r+") as f:
        assert "recency_visits_per_bin" in f["meta"].attrs
        del f["meta"].attrs["recency_visits_per_bin"]

    resumed = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce()
    )
    assert resumed._recency_visits_per_bin == 1000
    for slot in resumed.pool.replicas:
        assert slot.ensemble._recency_visits_per_bin == 1000


def test_resume_rejects_non_integer_recency_visits_per_bin_metadata(tmp_path):
    """A non-integer recency timescale in meta fails loudly on resume.

    Resume routes the saved ``recency_visits_per_bin`` through the same
    strict validator used at construction, so a corrupted or hand-edited
    checkpoint carrying a non-integer timescale raises rather than
    silently truncating (as a bare ``int(2.5)`` would).
    """
    import h5py
    import pytest

    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()

    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        recency_visits_per_bin=250,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    with h5py.File(cp, "r+") as f:
        f["meta"].attrs["recency_visits_per_bin"] = 2.5

    with pytest.raises(ValueError, match="positive integer"):
        WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce()
        )


def test_recency_flatness_populates_from_real_mc_steps():
    """Real MC steps drive recency_flatness to a non-None value in [0, 1].

    The other recency tests call ``_record_recency_visit`` by hand; this
    one exercises the production wiring end-to-end (``_update_entropy`` ->
    ``_record_recency_visit`` -> ``recency_effective_weights`` ->
    ``per_window_stats``). A window mid-exploration can legitimately
    report 0.0 (a known bin with no recent visit), so the contract is
    non-None and bounded, not strictly positive.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(e0 - 100.0, e0), (e0, e0 + 100.0)],
        energy_spacing=0.1,
        block_size=100,
        random_seed=0,
    )
    pt.run(n_cycles=10)
    stats = pt.pool.per_window_stats()
    rf_values = [s["recency_flatness"] for s in stats]
    assert any(v is not None for v in rf_values)
    for v in rf_values:
        assert v is None or 0.0 <= v <= 1.0


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
    with pytest.raises(ValueError, match="0.9.0"):
        WangLandauParallelTempering.resume(cp, cluster_expansion=make_wl_ce())


def test_wl_pt_resume_rejects_v3_schema(tmp_path):
    """v5 readers refuse v3 files with a message pointing at 0.9.0."""
    import h5py

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
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)
    with h5py.File(cp, "r+") as f:
        f["meta"].attrs["schema_version"] = "3"
    with pytest.raises(ValueError, match="0.9.0"):
        WangLandauParallelTempering.resume(cp, cluster_expansion=make_wl_ce())


def test_wl_pt_resume_rejects_schema4_multi_walker(tmp_path):
    """A schema-4 file with a window_groups subgroup is an old multi-walker
    checkpoint; its window-indexed labels are incompatible, so resume must
    reject it with a clear message rather than silently mis-restoring."""
    import h5py
    import numpy as np

    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    # Downgrade to the old multi-walker layout: schema "4" plus the
    # exchange_idx dataset the format used to write per window group.
    with h5py.File(cp, "r+") as f:
        f["meta"].attrs["schema_version"] = "4"
        for g in ("0", "1"):
            f[f"orchestrator/window_groups/{g}"].create_dataset(
                "exchange_idx", data=np.int32(0)
            )

    with pytest.raises(ValueError, match="multi-walker|regenerate"):
        WangLandauParallelTempering.resume(cp, cluster_expansion=make_wl_ce())


def test_wl_pt_resume_accepts_schema4_single_walker(tmp_path):
    """A single-walker schema-4 checkpoint has no window_groups subgroup; its
    labels are width n_windows == N_w, so it remains loadable under schema 5."""
    import h5py

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_replica import WangLandauReplica

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=1,
    )
    pt.run(n_cycles=2)
    cp = tmp_path / "wl.hdf5"
    pt.save_checkpoint(cp)

    # A single-walker run writes no window_groups; relabel it as schema 4.
    with h5py.File(cp, "r+") as f:
        assert "window_groups" not in f["orchestrator"]
        f["meta"].attrs["schema_version"] = "4"

    resumed = WangLandauParallelTempering.resume(
        cp, cluster_expansion=make_wl_ce(),
    )
    assert len(resumed.pool) == 2
    for slot in resumed.pool.replicas:
        assert isinstance(slot, WangLandauReplica)
    assert resumed._replica_labels.tolist() == pt._replica_labels.tolist()


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


def test_wl_pt_n_walkers_1_slots_are_bare_replicas():
    """n_walkers_per_window=1 places bare WangLandauReplica instances as slots."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_replica import WangLandauReplica

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
        assert isinstance(slot, WangLandauReplica)


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



def test_wl_pt_n_walkers_per_window_sequence_mixed_slot_types():
    """n_walkers_per_window=[1, 2]: W=1 window is a bare replica, W=2 is a group."""
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_replica import WangLandauReplica
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
    assert isinstance(pt.pool.replicas[0], WangLandauReplica)
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


def test_wl_pt_threads_recency_visits_per_bin_to_bare_replicas():
    """recency_visits_per_bin reaches every W=1 replica's ensemble."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        recency_visits_per_bin=250,
    )
    for slot in pt.pool.replicas:
        assert slot.ensemble._recency_visits_per_bin == 250


def test_wl_pt_threads_recency_visits_per_bin_to_window_groups():
    """recency_visits_per_bin reaches every walker in a W>1 window group."""
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
        recency_visits_per_bin=250,
    )
    for slot in pt.pool.replicas:
        for walker in slot._replicas:
            assert walker.ensemble._recency_visits_per_bin == 250


def test_wl_pt_recency_visits_per_bin_validated_at_orchestrator():
    """A non-positive recency_visits_per_bin raises at construction."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    with pytest.raises(ValueError, match="recency_visits_per_bin"):
        WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
            energy_spacing=0.1,
            block_size=5,
            random_seed=0,
            recency_visits_per_bin=0,
        )


def test_wl_pt_checkpoint_meta_records_recency_visits_per_bin():
    """recency_visits_per_bin is captured in the checkpoint metadata."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        recency_visits_per_bin=250,
    )
    assert pt._checkpoint_meta()["recency_visits_per_bin"] == 250


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


def test_wl_pt_w2_matching_attempts_two_per_active_cycle():
    """A W=2 boundary attempts a full 2x2 matching: 2 swaps per active cycle.

    Two two-walker windows give n_carriers = 4. Boundary 0 is active on
    even cycles only (cycles 0 and 2 over a 3-cycle run); each active
    cycle proposes min(W_i, W_j) = 2 disjoint walker pairs. So
    swap_attempted[0] = 2 active cycles x 2 pairs = 4, and the label
    width equals n_carriers = 4.
    """
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(e0 - 100.0, e0 + 100.0), (e0 - 100.0, e0 + 100.0)],
        energy_spacing=0.1,
        block_size=5,
        random_seed=0,
        n_walkers_per_window=2,
    )
    history = pt.run(n_cycles=3)
    assert pt.pool.n_carriers() == 4
    assert history.replica_labels_per_cycle.shape[1] == 4
    # Cycles 0 and 2 are even (pair 0 active); cycle 1 is odd (no pairs).
    # Each active cycle proposes a 2x2 matching = 2 attempts.
    assert history.swap_attempted[0] == 4


def test_wl_pt_single_walker_matching_one_attempt_per_active_cycle():
    """A W=1 boundary attempts exactly one swap per active cycle.

    With single-walker windows the matching collapses to the single
    pair (0, 0): one attempt per active cycle and a label width equal
    to len(pool), matching the pre-matching behaviour byte-for-byte.
    """
    from mchammer_pt.wl import WangLandauParallelTempering

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
    history = pt.run(n_cycles=3)
    assert pt.pool.n_carriers() == len(pt.pool)
    assert history.replica_labels_per_cycle.shape[1] == len(pt.pool)
    # Boundary 0 active on cycles 0 and 2: one attempt each.
    assert history.swap_attempted[0] == 2


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


def test_wl_pt_process_pool_accepts_multi_walker_with_checkpoint(tmp_path):
    """process_pool() accepts n_walkers > 1 with data_container_file."""
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
        data_container_file=tmp_path / "ckpt.hdf5",
    )
    try:
        assert pt is not None
    finally:
        pt.pool.shutdown()


def test_wl_pt_serial_w1_slot_is_bare_replica():
    """Serial path uses bare WangLandauReplica slots for W=1."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble
    from mchammer_pt.wl_replica import WangLandauReplica

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
        assert isinstance(slot, WangLandauReplica)


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
    assert pt._pool._flatness_mode == "pooled"
    assert pt._pool._merge_cadence == "at_halve"


def test_wl_pt_flatness_mode_per_walker_propagates():
    """flatness_mode='per_walker' is stored on the pool."""
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
    assert pt._pool._flatness_mode == "per_walker"


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


def test_wl_pt_v4_checkpoint_has_walkers_per_window_in_meta(tmp_path):
    """Schema v4: /meta carries walkers_per_window as an int array."""
    import h5py

    from mchammer_pt.wl import WangLandauParallelTempering
    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=1,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == "5"
        wpw = np.asarray(f["meta"].attrs["walkers_per_window"])
        assert wpw.tolist() == [1, 1]  # two windows in the simple fixture


def test_walker_seeds_helper_is_deterministic_and_matches_constructor():
    """The extracted helper produces the same per-walker / per-group /
    master seeds as the existing WangLandauParallelTempering constructor."""
    from mchammer_pt.wl import _spawn_wl_seeds

    walker_seeds, group_seeds, master_seed = _spawn_wl_seeds(
        random_seed=42,
        walkers_per_window=[1, 2, 1],
    )
    # Stability: repeating the call yields identical seeds.
    again = _spawn_wl_seeds(42, [1, 2, 1])
    assert (walker_seeds, group_seeds, master_seed) == again
    # Shape:
    assert len(walker_seeds) == 3  # one list per window
    assert [len(ws) for ws in walker_seeds] == [1, 2, 1]
    assert len(group_seeds) == 3
    assert isinstance(master_seed, int)


def test_wl_pt_resume_w2_round_trips(tmp_path):
    """Save mid-run with W=2, resume, and continue running without errors.

    W>1 resume does not guarantee bit-identical continuation: ``run()``
    calls ``finalise_for_reporting`` at exit, which merges per-walker
    entropies in each window group. The resumed walkers therefore start
    from the merged entropy rather than the individual entropies they
    held at the split point, so the MC acceptance chain diverges from
    an uninterrupted run. The test instead checks that:

    - ``resume`` reconstructs the correct pool structure (W=2 slots).
    - The resumed run completes without errors.
    - All per-window results return non-None entropy.
    - Current energies are finite after resuming.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    # Schema 5: window-group subgroups carry no exchange_idx.
    import h5py

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == "5"
        for g in ("0", "1"):
            assert "exchange_idx" not in f[f"orchestrator/window_groups/{g}"]

    resumed = WangLandauParallelTempering.resume(
        path, cluster_expansion=make_wl_ce(),
    )

    # Pool structure: 2 WangLandauWindowGroup slots, each with 2 walkers.
    assert len(resumed.pool) == 2
    for slot in resumed.pool.replicas:
        assert isinstance(slot, WangLandauWindowGroup)
        assert len(slot.walker_states) == 2

    # Walker-indexed replica labels (width N_w = 4) round-trip exactly.
    assert resumed._replica_labels.tolist() == pt._replica_labels.tolist()
    assert len(resumed._replica_labels) == 4

    history = resumed.run(n_cycles=2)
    assert history.energies_per_cycle.shape == (3, 2)
    assert np.all(np.isfinite(resumed.pool.current_energies()))
    for wr in resumed.results():
        assert wr.get_entropy() is not None


def test_wl_pt_resume_rejects_walker_count_mismatch(tmp_path):
    """Truncated /replicas/ relative to walkers_per_window must raise.

    Constructs a real W=2 checkpoint, then corrupts the file by lying
    about walkers_per_window so it claims more walkers than the
    /replicas/ group contains. resume must reject before reconstruction.
    """
    import h5py

    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=1,  # 2 replicas total
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)
    # Lie about the walker count without adjusting /replicas/.
    with h5py.File(path, "r+") as f:
        f["meta"].attrs["walkers_per_window"] = np.array([2, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="walker-count mismatch"):
        WangLandauParallelTempering.resume(
            path, cluster_expansion=make_wl_ce(),
        )


def test_wl_pt_round_trip_twice_preserves_continuation(tmp_path):
    """save -> resume -> run -> save -> resume keeps the orchestrator
    consistent across two save/resume cycles.

    Pins identity-copy-through: nothing accumulates spurious state, and
    the second resume reconstructs the orchestrator with the same pool
    shape, finite energies, and non-None entropies as the first.
    """
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
    )
    pt.run(n_cycles=2)
    cp1 = tmp_path / "ckpt-1.h5"
    pt.save_checkpoint(cp1)

    with pytest.warns(UserWarning, match="not bit-identical"):
        resumed_1 = WangLandauParallelTempering.resume(
            cp1, cluster_expansion=make_wl_ce(),
        )
    resumed_1.run(n_cycles=2)
    cp2 = tmp_path / "ckpt-2.h5"
    resumed_1.save_checkpoint(cp2)

    with pytest.warns(UserWarning, match="not bit-identical"):
        resumed_2 = WangLandauParallelTempering.resume(
            cp2, cluster_expansion=make_wl_ce(),
        )
    history = resumed_2.run(n_cycles=2)
    assert history.energies_per_cycle.shape == (3, 2)
    assert np.all(np.isfinite(resumed_2.pool.current_energies()))
    assert len(resumed_2.pool) == 2
    for wr in resumed_2.results():
        assert wr.get_entropy() is not None


def test_wl_pt_resume_emits_warning_for_multi_walker_windows(tmp_path):
    """W>1 resume emits UserWarning naming the affected windows."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=[1, 2],
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with pytest.warns(UserWarning, match=r"windows \[1\].*not bit-identical"):
        WangLandauParallelTempering.resume(
            path, cluster_expansion=make_wl_ce(),
        )


def test_wl_pt_resume_does_not_warn_for_all_w1(tmp_path):
    """All-W=1 resume is bit-identical; no warning expected."""
    import warnings as _warnings

    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=1,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        WangLandauParallelTempering.resume(
            path, cluster_expansion=make_wl_ce(),
        )


def test_wl_pt_resume_process_pool_w2_round_trips(tmp_path):
    """W=2 process pool resume reconstructs structure and continues without errors.

    Same relaxed contract as the serial-pool W=2 resume test
    (test_wl_pt_resume_w2_round_trips): finalise_for_reporting in run()'s
    finally destroys pre-merge per-walker entropy, so resumed runs are
    structurally correct but not bit-identical to uninterrupted runs.
    """
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering.process_pool(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
    )
    try:
        pt.run(n_cycles=2)
        path = tmp_path / "ckpt.h5"
        pt.save_checkpoint(path)
    finally:
        pt.pool.shutdown()

    resumed = WangLandauParallelTempering.resume_process_pool(
        path, cluster_expansion=make_wl_ce(),
    )
    try:
        assert len(resumed.pool) == 2
        history = resumed.run(n_cycles=2)
        assert history.energies_per_cycle.shape == (3, 2)
        assert np.all(np.isfinite(resumed.pool.current_energies()))
        for wr in resumed.results():
            assert wr.get_entropy() is not None
    finally:
        resumed.pool.shutdown()


def test_wl_pt_constructor_accepts_w2_with_data_container_file(tmp_path):
    """The constructor accepts W>1 with data_container_file."""
    from mchammer_pt.wl import WangLandauParallelTempering

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
        data_container_file=tmp_path / "ckpt.h5",
    )
    assert pt is not None


def test_wl_pt_checkpoint_round_trip_mixed_walkers_per_window(tmp_path):
    """walkers_per_window=[1, 2, 1] round-trips through save/resume.

    Asserts structural correctness only — per the relaxed W>1 same-pool
    contract documented in test_wl_pt_resume_w2_round_trips. The
    heterogeneous slot layout exercises the bare-replica / WindowGroup
    dispatch path in both snapshot and restore.
    """
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms(), make_wl_atoms()],
        windows=[
            (None, e0 + 50.0),
            (e0 - 50.0, e0 + 50.0),
            (e0 - 50.0, None),
        ],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=[1, 2, 1],
    )
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    resumed = WangLandauParallelTempering.resume(
        path, cluster_expansion=make_wl_ce(),
    )

    # Pool structure: 3 slots — bare/group/bare.
    assert len(resumed.pool) == 3
    assert isinstance(resumed.pool.replicas[0], WangLandauReplica)
    assert isinstance(resumed.pool.replicas[1], WangLandauWindowGroup)
    assert len(resumed.pool.replicas[1].walker_states) == 2
    assert isinstance(resumed.pool.replicas[2], WangLandauReplica)

    history = resumed.run(n_cycles=2)
    assert history.energies_per_cycle.shape == (3, 3)
    assert np.all(np.isfinite(resumed.pool.current_energies()))
    for wr in resumed.results():
        assert wr.get_entropy() is not None


def _distinct_in_window_pair():
    """Return (ce, a, b, ea, eb) for the serial per-walker tests.

    Thin wrapper over :func:`tests._wl_fixtures.distinct_in_window_pair`
    that also returns the in-memory CE the energies were computed under.
    """
    ce = make_wl_ce()
    a, b, ea, eb = distinct_in_window_pair(ce)
    return ce, a, b, ea, eb


def test_wl_pt_serial_per_walker_structures_reach_replicas():
    ce, a, b, ea, eb = _distinct_in_window_pair()
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[[a, b], a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=[2, 1],
    )
    group = pt._pool._replicas[0]  # WangLandauWindowGroup (W=2)
    assert group._replicas[0].current_energy() == pytest.approx(ea)
    assert group._replicas[1].current_energy() == pytest.approx(eb)


def test_wl_pt_serial_broadcast_yields_independent_walkers():
    ce, a, b, ea, eb = _distinct_in_window_pair()
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[a, a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=[2, 1],
    )
    group = pt._pool._replicas[0]
    r0, r1 = group._replicas[0], group._replicas[1]
    assert r0 is not r1
    np.testing.assert_array_equal(
        r0.current_occupations(), r1.current_occupations()
    )
    # Mutate walker 0 to b's (in-window) configuration; walker 1 unchanged.
    r0.set_occupations(b.numbers)
    assert not np.array_equal(
        r0.current_occupations(), r1.current_occupations()
    )


def test_wl_pt_serial_per_walker_length_mismatch_raises():
    ce, a, _b, ea, _eb = _distinct_in_window_pair()
    lo, hi = ea - 1.0, ea + 1.0
    with pytest.raises(ValueError, match=r"window 0 has 2 walkers"):
        WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[[a], a],
            windows=[(lo, hi), (lo, hi)],
            energy_spacing=0.1,
            block_size=10,
            random_seed=0,
            n_walkers_per_window=[2, 1],
        )


def test_wl_pt_serial_per_walker_out_of_window_raises():
    ce, a, b, ea, eb = _distinct_in_window_pair()
    # Narrow window brackets a only; b is outside.
    lo, hi = ea - 0.5, ea + 0.5
    assert not (lo < eb < hi)
    with pytest.raises(ValueError, match="outside window"):
        WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[[a, b], a],
            windows=[(lo, hi), (lo, hi)],
            energy_spacing=0.1,
            block_size=10,
            random_seed=0,
            n_walkers_per_window=[2, 1],
        )


def test_wl_pt_serial_per_walker_start_checkpoints_and_resumes(tmp_path):
    ce, a, b, ea, eb = _distinct_in_window_pair()
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    ckpt = tmp_path / "rewl.h5"
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[[a, b], a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=[2, 1],
    )
    pt.run(3)
    pt.save_checkpoint(ckpt)

    with pytest.warns(UserWarning):
        pt_b = WangLandauParallelTempering.resume(ckpt, cluster_expansion=ce)
    # Structurally resumes: one WindowResult per window, walker counts
    # preserved (window 0 has 2 walkers, window 1 has 1).
    results = pt_b.results()
    assert len(results) == 2
    assert len(results[0].containers) == 2
    assert len(results[1].containers) == 1


def test_wl_pt_stores_and_forwards_dos_snapshot_ratio():
    """The orchestrator stores dos_snapshot_ratio and forwards it to replicas."""
    e0 = _initial_energy()
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        dos_snapshot_ratio=4.0,
        data_container_file=None,
    )
    assert pt._dos_snapshot_ratio == 4.0
    # Each W=1 window's slot is a bare WangLandauReplica; .ensemble is its
    # CoordinatedWangLandauEnsemble. The serial pool exposes slots via the
    # public `replicas` property.
    for slot in pt.pool.replicas:
        assert slot.ensemble._dos_snapshot_ratio == 4.0


def test_wl_pt_checkpoint_meta_records_dos_snapshot_ratio():
    """dos_snapshot_ratio is written into checkpoint metadata."""
    e0 = _initial_energy()
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        dos_snapshot_ratio=4.0,
        data_container_file=None,
    )
    meta = pt._checkpoint_meta()
    assert meta["dos_snapshot_ratio"] == 4.0


def test_wl_pt_checkpoint_meta_encodes_disabled_as_nan():
    """dos_snapshot_ratio=None round-trips through the float-only meta as NaN."""
    import math

    e0 = _initial_energy()
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        dos_snapshot_ratio=None,
        data_container_file=None,
    )
    meta = pt._checkpoint_meta()
    assert isinstance(meta["dos_snapshot_ratio"], float)
    assert math.isnan(meta["dos_snapshot_ratio"])


def test_resume_threads_dos_snapshot_ratio_from_metadata(tmp_path):
    """A serial resume reads dos_snapshot_ratio back from checkpoint meta."""
    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = _initial_energy()
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        dos_snapshot_ratio=4.0,
        data_container_file=None,
    )
    pt.run(n_cycles=1)
    ckpt = tmp_path / "rewl_state.h5"
    pt.save_checkpoint(ckpt)

    resumed = WangLandauParallelTempering.resume(
        ckpt,
        cluster_expansion=ce,
    )
    assert resumed._dos_snapshot_ratio == 4.0
    for slot in resumed.pool.replicas:
        assert slot.ensemble._dos_snapshot_ratio == 4.0


def test_resume_threads_disabled_dos_snapshot_ratio_from_metadata(tmp_path):
    """A run with snapshotting disabled stays disabled across resume (NaN->None)."""
    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = _initial_energy()
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        dos_snapshot_ratio=None,
        data_container_file=None,
    )
    pt.run(n_cycles=1)
    ckpt = tmp_path / "rewl_state.h5"
    pt.save_checkpoint(ckpt)

    resumed = WangLandauParallelTempering.resume(
        ckpt,
        cluster_expansion=ce,
    )
    assert resumed._dos_snapshot_ratio is None


def test_resume_round_trips_non_empty_1_over_t_snapshot_store(tmp_path):
    """A non-empty 1/t snapshot store survives a real HDF5 checkpoint and
    resume, and get_entropy reads it back below the last halving.

    Exercises the full persistence chain on non-empty snapshot content --
    refresh_last_state -> HDF5 serialise -> HDF5 read -> integer-key
    coercion -> restore_state -> get_entropy union scan. The in-memory
    round-trip unit tests never hit the real string-key-to-int-key
    coercion against the serialiser, so a regression there would silently
    drop every 1/t snapshot on resume with no failing test.
    """
    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = _initial_energy()
    lo, hi = e0 - 100.0, e0 + 100.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        dos_snapshot_ratio=2.0,
        data_container_file=None,
    )
    pt.run(n_cycles=1)

    # Seed a non-empty 1/t snapshot store on each window's walker, with the
    # live fill factor below the snapshot rungs so get_entropy's guard
    # passes for a below-rung limit.
    seeded_ff = {400: 1.0 / 64, 500: 1.0 / 128}
    seeded_entropy = {400: {0: 8.0, 1: 16.0}, 500: {0: 10.0, 1: 20.0}}
    for replica in pt.pool.replicas:
        replica.ensemble._fill_factor_snapshots = dict(seeded_ff)
        replica.ensemble._entropy_snapshots = {
            step: dict(entropy) for step, entropy in seeded_entropy.items()
        }
        replica.ensemble._fill_factor = 1.0 / 128

    ckpt = tmp_path / "rewl_state.h5"
    pt.save_checkpoint(ckpt)

    resumed = WangLandauParallelTempering.resume(ckpt, cluster_expansion=ce)

    # The store survives the real HDF5 round-trip with integer keys
    # restored (string keys would fail these equality checks).
    for replica in resumed.pool.replicas:
        assert replica.ensemble._fill_factor_snapshots == seeded_ff
        assert replica.ensemble._entropy_snapshots == seeded_entropy

    # And get_entropy reads a snapshot back at a below-last-halving limit
    # (1/64), where without the persisted store it would return None.
    for window in resumed.results():
        df = window.get_entropy(fill_factor_limit=1.0 / 64)
        assert df is not None
        assert not df.empty


class TestResumeOneOverTGate:
    def test_resume_round_trips_policy(self, tmp_path) -> None:
        from mchammer_pt.wl import WangLandauParallelTempering

        ekw = {"schedule": "1_over_t"}
        pt = _make_serial_wl_pt(
            one_over_t_gate="flatness", bp_stall_multiple=3.0, ensemble_kwargs=ekw,
        )
        pt.run(n_cycles=2)
        cp = tmp_path / "wl_policy.hdf5"
        pt.save_checkpoint(cp)
        resumed = WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(), ensemble_kwargs=ekw,
        )
        assert resumed._one_over_t_gate == "flatness"
        assert resumed._bp_stall_multiple == 3.0

    def test_resume_preserves_stall_state(self, tmp_path) -> None:
        from mchammer.calculators import ClusterExpansionCalculator

        from mchammer_pt.wl import WangLandauParallelTempering

        ce, atoms = make_wl_ce(), make_wl_atoms()
        e0 = float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )
        # Narrow windows + coarse spacing -> few bins -> visit-once halves fast.
        lo, hi = e0 - 3.0, e0 + 3.0
        ekw = {"schedule": "1_over_t"}
        pt_a = WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[atoms, atoms],
            windows=[(lo, hi), (lo, hi)],
            energy_spacing=0.5,
            block_size=500,
            random_seed=0,
            ensemble_kwargs=ekw,
        )
        pt_a.run(n_cycles=10)
        assert any(x is not None for x in pt_a._pool._last_halve_step), (
            "fixture did not halve; narrow the window or raise n_cycles"
        )
        cp = tmp_path / "wl_stall.hdf5"
        pt_a.save_checkpoint(cp)
        resumed = WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(), ensemble_kwargs=ekw,
        )
        assert resumed._pool._last_halve_step == pt_a._pool._last_halve_step
        assert (
            resumed._pool._first_halve_duration
            == pt_a._pool._first_halve_duration
        )


class TestResumeOneOverTEntry:
    def test_resume_round_trips_policy(self, tmp_path) -> None:
        from mchammer_pt.wl import WangLandauParallelTempering

        ekw = {"schedule": "1_over_t"}
        pt = _make_serial_wl_pt(
            one_over_t_entry="f_continuous", ensemble_kwargs=ekw,
        )
        pt.run(n_cycles=2)
        cp = tmp_path / "wl_entry_policy.hdf5"
        pt.save_checkpoint(cp)
        resumed = WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(), ensemble_kwargs=ekw,
        )
        assert resumed._one_over_t_entry == "f_continuous"
        for slot in resumed._pool.replicas:
            assert slot.one_over_t_entry == "f_continuous"

    def test_resume_defaults_missing_key_to_window_clock(
        self, tmp_path, monkeypatch
    ) -> None:
        """A checkpoint written before ``one_over_t_entry`` was
        recorded has no such meta key; resume must fall back to
        ``window_clock``."""
        from mchammer_pt.history import read_hdf5 as real_read_hdf5
        from mchammer_pt.wl import WangLandauParallelTempering

        pt = _make_serial_wl_pt()
        pt.run(n_cycles=2)
        cp = tmp_path / "wl_pre_feature.hdf5"
        pt.save_checkpoint(cp)

        def read_without_entry_key(path):
            histories, containers, meta = real_read_hdf5(path)
            meta = dict(meta)
            meta.pop("one_over_t_entry", None)
            return histories, containers, meta

        monkeypatch.setattr(
            "mchammer_pt.history.read_hdf5", read_without_entry_key
        )
        resumed = WangLandauParallelTempering.resume(
            cp, cluster_expansion=make_wl_ce(),
        )
        assert resumed._one_over_t_entry == "window_clock"


def test_resume_process_pool_preserves_stall_state(tmp_path) -> None:
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    lo, hi = e0 - 3.0, e0 + 3.0
    ekw = {"schedule": "1_over_t"}
    cp = tmp_path / "wl_proc_stall.hdf5"
    pt_a = WangLandauParallelTempering.process_pool(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.5,
        block_size=500,
        random_seed=0,
        ensemble_kwargs=ekw,
    )
    try:
        pt_a.run(n_cycles=10)
        live = [s.last_halve_step for s in pt_a._pool._slots]
        live_t1 = [s.first_halve_duration for s in pt_a._pool._slots]
        assert any(x is not None for x in live), (
            "fixture did not halve; narrow the window or raise n_cycles"
        )
        pt_a.save_checkpoint(cp)
    finally:
        pt_a._pool.shutdown()

    resumed = WangLandauParallelTempering.resume_process_pool(
        cp, cluster_expansion=make_wl_ce(), ensemble_kwargs=ekw,
    )
    try:
        assert [s.last_halve_step for s in resumed._pool._slots] == live
        assert [s.first_halve_duration for s in resumed._pool._slots] == live_t1
    finally:
        resumed._pool.shutdown()


class TestOneOverTGateIntegration:
    def test_default_run_unchanged(self) -> None:
        # random_seed is fixed inside _make_serial_wl_pt, so the only
        # difference is the explicit visit_once kwarg vs the default.
        pt_a = _make_serial_wl_pt(ensemble_kwargs={"schedule": "1_over_t"})
        pt_b = _make_serial_wl_pt(
            ensemble_kwargs={"schedule": "1_over_t"},
            one_over_t_gate="visit_once",
        )
        pt_a.run(n_cycles=5)
        pt_b.run(n_cycles=5)
        ent_a = dict(pt_a._pool._replicas[0].ensemble._entropy)
        ent_b = dict(pt_b._pool._replicas[0].ensemble._entropy)
        assert ent_a == ent_b

    def test_serial_process_view_parity(self) -> None:
        import numpy as np

        from mchammer_pt.parallel.processes import (
            ProcessWangLandauWindow,
            _view_of,
        )
        from mchammer_pt.wl_coordinator import (
            WalkerPostBlockState,
            decide_block_actions,
        )

        # A stalled, un-flat, halved-once walker state (trips the escape).
        state = WalkerPostBlockState(
            halving_criterion_met=False,
            fill_factor=0.25,
            entropy={0: 1.0, 1: 2.0},
            step=5000,
            window_entry_step=0,
            histogram={0: 1, 1: 100},
            reached_energy_window=True,
            current_energy=0.0,
        )

        # Serial slot: a real W=1 replica under the flatness variant.
        pt = _make_serial_wl_pt(
            one_over_t_gate="flatness",
            ensemble_kwargs={"schedule": "1_over_t"},
        )
        serial_pool = pt._pool
        serial_slot = serial_pool._replicas[0]
        serial_slot.walker_states = (state,)
        serial_pool._last_halve_step[0] = 1000
        serial_pool._first_halve_duration[0] = 100
        v_serial = serial_pool._view_of(0, serial_slot)

        # Process window: identical state.
        win = ProcessWangLandauWindow(
            workers=[],
            rng=np.random.default_rng(0),
            schedule="1_over_t",
            one_over_t_gate="flatness",
        )
        win.walker_states = [state]
        win.last_halve_step = 1000
        win.first_halve_duration = 100
        v_proc = _view_of(win)

        assert v_serial == v_proc
        assert decide_block_actions(v_serial) == decide_block_actions(v_proc)
        # And the shared decision is the stuck-window escape.
        assert decide_block_actions(v_serial).halve is False
        assert decide_block_actions(v_serial).switch_to_phase == "1_over_t"

    def test_in_process_flatness_run_halves_and_tracks(self, tmp_path) -> None:
        # End-to-end exercise of ProcessWangLandauPool.advance_all under the
        # flatness variant: a narrow window halves under the flatness gate,
        # and the process backend tracks the halve (last_halve_step set, with
        # a consistent first-stage duration). A real, non-vacuous assertion
        # about the new path -- not just a type check.
        from mchammer.calculators import ClusterExpansionCalculator

        from tests._in_process_pool import make_in_process_wl_pool

        ce, atoms = make_wl_ce(), make_wl_atoms()
        e0 = float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )
        # Narrow window (few bins at the fixture's 0.1 spacing) so the
        # flatness gate is reachable within the step budget.
        lo, hi = e0 - 1.0, e0 + 1.0
        pool = make_in_process_wl_pool(
            tmp_path,
            windows=[(lo, hi), (lo, hi)],
            seeds=[0, 1],
            ensemble_kwargs={"schedule": "1_over_t"},
            one_over_t_gate="flatness",
            bp_stall_multiple=3.0,
        )
        try:
            for _ in range(20):
                pool.advance_all(500)
            assert any(s.last_halve_step is not None for s in pool._slots), (
                "no window halved under the flatness gate; narrow the window "
                "or raise the step budget"
            )
            for slot in pool._slots:
                assert slot._one_over_t_gate == "flatness"
                assert slot._bp_stall_multiple == 3.0
                # A first-stage duration is only set once a halve is recorded.
                if slot.first_halve_duration is not None:
                    assert slot.last_halve_step is not None
        finally:
            pool.shutdown()
