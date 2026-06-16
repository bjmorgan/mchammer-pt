"""Tests for record_observable plumbing and measure_from_checkpoint.

Covers:
- SerialWangLandauPool.record_observable installs recorder on every walker
  (single-replica slots and multi-walker WangLandauWindowGroup slots).
- ProcessWangLandauPool.record_observable installs recorder on every worker
  (exercised via in-process workers to avoid real subprocess spawn).
- WangLandauParallelTempering.record_observable forwards to the pool and
  every replica carries a recorder for the given tag.
- measure_from_checkpoint loads a converged checkpoint with frozen_g=True
  on all ensembles and frozen_measurement=True on the pool.
- A real measurement run via measure_from_checkpoint does not mutate
  _entropy on any ensemble (g frozen end-to-end through exchanges).
"""

from __future__ import annotations

import pytest
from mchammer.observers.base_observer import BaseObserver

# ---------------------------------------------------------------------------
# Minimal picklable test observer (module-level so ProcessPool can import it).
# ---------------------------------------------------------------------------


class _ConstantObs(BaseObserver):
    """Returns a fixed constant regardless of structure.  Picklable."""

    def __init__(self, tag: str = "const", interval: int = 1) -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)

    def get_observable(self, structure: object) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_energy():
    from mchammer.calculators import ClusterExpansionCalculator

    from tests._wl_fixtures import make_wl_atoms, make_wl_ce  # noqa: PLC0415

    atoms = make_wl_atoms()
    return float(
        ClusterExpansionCalculator(atoms, make_wl_ce()).calculate_total(
            occupations=atoms.numbers
        )
    )


def _make_two_window_serial_pt(**overrides):
    """Two-window serial WangLandauParallelTempering on the toy CE."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

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


# ---------------------------------------------------------------------------
# Task 4.1 — record_observable plumbing
# ---------------------------------------------------------------------------


class TestSerialPoolRecordObservable:
    """SerialWangLandauPool.record_observable."""

    def test_record_observable_installs_recorder_on_single_walker_slot(self):
        """Single-walker window: the replica gains a recorder for the tag."""
        from tests._wl_fixtures import make_serial_wl_pool_mixed

        pool = make_serial_wl_pool_mixed()
        obs = _ConstantObs(tag="test_rec")
        pool.record_observable(obs)

        # Window 0 is a bare WangLandauReplica (W=1).
        replica_0 = pool._replicas[0]
        assert "test_rec" in replica_0.ensemble._recorders

    def test_record_observable_installs_recorder_on_every_walker_in_group(self):
        """Multi-walker window: every walker in the group gains a recorder."""
        from tests._wl_fixtures import make_serial_wl_pool_mixed

        pool = make_serial_wl_pool_mixed()
        obs = _ConstantObs(tag="grp_rec")
        pool.record_observable(obs)

        # Window 1 is a WangLandauWindowGroup with 2 walkers.
        group = pool._replicas[1]
        for replica in group._replicas:
            assert "grp_rec" in replica.ensemble._recorders

    def test_record_observable_each_walker_gets_independent_copy(self):
        """Each walker's recorder is an independent deserialised copy."""
        from mchammer_pt.wl_window_group import WangLandauWindowGroup
        from tests._wl_fixtures import make_serial_wl_pool_mixed

        pool = make_serial_wl_pool_mixed()
        obs = _ConstantObs(tag="ind_rec")
        pool.record_observable(obs)

        group = pool._replicas[1]
        assert isinstance(group, WangLandauWindowGroup)
        rec_0 = group._replicas[0].ensemble._recorders["ind_rec"]
        rec_1 = group._replicas[1].ensemble._recorders["ind_rec"]
        assert rec_0 is not rec_1

    def test_record_observable_all_replicas_default(self):
        """Default replicas='all' reaches both windows."""
        from tests._wl_fixtures import make_serial_wl_pool_mixed

        pool = make_serial_wl_pool_mixed()
        obs = _ConstantObs(tag="all_rec")
        pool.record_observable(obs)

        # Window 0: bare replica.
        assert "all_rec" in pool._replicas[0].ensemble._recorders
        # Window 1: every walker in the group.
        for r in pool._replicas[1]._replicas:
            assert "all_rec" in r.ensemble._recorders

    def test_record_observable_selected_replica_only(self):
        """replicas=[0] installs only on window 0."""
        from tests._wl_fixtures import make_serial_wl_pool_mixed

        pool = make_serial_wl_pool_mixed()
        obs = _ConstantObs(tag="sel_rec")
        pool.record_observable(obs, replicas=[0])

        assert "sel_rec" in pool._replicas[0].ensemble._recorders
        for r in pool._replicas[1]._replicas:
            assert "sel_rec" not in r.ensemble._recorders


class TestProcessPoolRecordObservable:
    """ProcessWangLandauPool.record_observable via in-process workers."""

    def test_record_observable_installs_recorder_on_worker(self, tmp_path):
        """ATTACH_RECORDER command installs a recorder on the worker's replica."""
        from tests._wl_fixtures import make_process_wl_pool_w2

        pool = make_process_wl_pool_w2(tmp_path)
        try:
            obs = _ConstantObs(tag="proc_rec")
            pool.record_observable(obs)

            # Verify via GET_RECORDERS or inspect worker state indirectly
            # through get_recorders if it exists, or via a direct internal probe.
            # We use the in-process connection's underlying worker replica.
            for slot in pool._slots:
                for _, conn in slot.workers:
                    from tests._in_process_worker import InProcessWorkerConn
                    if isinstance(conn, InProcessWorkerConn):
                        assert "proc_rec" in conn._worker._replica.ensemble._recorders
        finally:
            pool.shutdown()

    def test_record_observable_all_slots_and_walkers(self, tmp_path):
        """With 2 windows x W=2, all 4 workers receive the recorder."""
        from tests._wl_fixtures import make_process_wl_pool_w2

        pool = make_process_wl_pool_w2(tmp_path)
        try:
            obs = _ConstantObs(tag="all_proc_rec")
            pool.record_observable(obs)

            count = 0
            for slot in pool._slots:
                for _, conn in slot.workers:
                    from tests._in_process_worker import InProcessWorkerConn
                    if isinstance(conn, InProcessWorkerConn):
                        recorders = conn._worker._replica.ensemble._recorders
                        assert "all_proc_rec" in recorders
                        count += 1
            assert count == 4
        finally:
            pool.shutdown()


class TestOrchestatorRecordObservable:
    """WangLandauParallelTempering.record_observable forwards to the pool."""

    def test_record_observable_reaches_all_replicas(self):
        """Every window's ensemble(s) gain a recorder after the call."""
        pt = _make_two_window_serial_pt()
        obs = _ConstantObs(tag="orch_rec")
        pt.record_observable(obs)

        for slot in pt.pool._replicas:
            # Slot may be a bare WangLandauReplica or a WangLandauWindowGroup.
            from mchammer_pt.wl_window_group import WangLandauWindowGroup
            if isinstance(slot, WangLandauWindowGroup):
                for r in slot._replicas:
                    assert "orch_rec" in r.ensemble._recorders
            else:
                assert "orch_rec" in slot.ensemble._recorders

    def test_record_observable_selected_replica(self):
        """replicas=[0] reaches only window 0."""
        pt = _make_two_window_serial_pt()
        obs = _ConstantObs(tag="sel_orch_rec")
        pt.record_observable(obs, replicas=[0])

        slot_0 = pt.pool._replicas[0]
        assert "sel_orch_rec" in slot_0.ensemble._recorders

        slot_1 = pt.pool._replicas[1]
        from mchammer_pt.wl_window_group import WangLandauWindowGroup
        if isinstance(slot_1, WangLandauWindowGroup):
            for r in slot_1._replicas:
                assert "sel_orch_rec" not in r.ensemble._recorders
        else:
            assert "sel_orch_rec" not in slot_1.ensemble._recorders

    def test_record_observable_on_non_observable_pool_raises(self):
        """Calling record_observable when pool does not support it raises TypeError."""
        from unittest.mock import MagicMock


        pt = _make_two_window_serial_pt()
        # Replace the pool with a minimal mock that does not support recording.
        bare_pool = MagicMock()
        bare_pool.__len__ = MagicMock(return_value=2)
        # Remove record_observable so isinstance check fails.
        del bare_pool.record_observable
        pt._pool = bare_pool

        with pytest.raises(TypeError, match="record_observable"):
            pt.record_observable(_ConstantObs(tag="x"))


# ---------------------------------------------------------------------------
# Task 4.2 — measure_from_checkpoint
# ---------------------------------------------------------------------------


class TestMeasureFromCheckpoint:
    """WangLandauParallelTempering.measure_from_checkpoint."""

    def _make_and_save_checkpoint(self, tmp_path):
        """Run a tiny REWL run and save a checkpoint; return path and CE."""
        from tests._wl_fixtures import make_wl_atoms, make_wl_ce

        ce = make_wl_ce()
        atoms = make_wl_atoms()
        e0 = _initial_energy()
        lo, hi = e0 - 50.0, e0 + 50.0
        pt = _make_two_window_serial_pt(
            cluster_expansion=ce,
            atoms=[atoms, atoms],
            windows=[(lo, hi), (lo, hi)],
        )
        pt.run(n_cycles=2)
        ckpt = tmp_path / "ckpt.h5"
        pt.save_checkpoint(ckpt)
        return ckpt, ce

    def test_measure_from_checkpoint_all_ensembles_have_frozen_g(self, tmp_path):
        """Every ensemble's _frozen_g is True after measure_from_checkpoint."""
        from mchammer_pt.wl import WangLandauParallelTempering

        ckpt, ce = self._make_and_save_checkpoint(tmp_path)
        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            ckpt, cluster_expansion=ce
        )
        from mchammer_pt.wl_window_group import WangLandauWindowGroup
        for slot in pt2.pool._replicas:
            if isinstance(slot, WangLandauWindowGroup):
                for r in slot._replicas:
                    assert r.ensemble._frozen_g is True, (
                        "expected _frozen_g=True after measure_from_checkpoint"
                    )
            else:
                assert slot.ensemble._frozen_g is True, (
                    "expected _frozen_g=True after measure_from_checkpoint"
                )

    def test_measure_from_checkpoint_pool_has_frozen_measurement(self, tmp_path):
        """Pool._frozen_measurement is True after measure_from_checkpoint."""
        from mchammer_pt.wl import WangLandauParallelTempering

        ckpt, ce = self._make_and_save_checkpoint(tmp_path)
        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            ckpt, cluster_expansion=ce
        )
        assert pt2.pool._frozen_measurement is True

    def test_measure_from_checkpoint_run_does_not_mutate_entropy(self, tmp_path):
        """A measurement run leaves _entropy bit-identical on every ensemble."""
        from mchammer_pt.wl import WangLandauParallelTempering
        from mchammer_pt.wl_window_group import WangLandauWindowGroup

        ckpt, ce = self._make_and_save_checkpoint(tmp_path)
        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            ckpt, cluster_expansion=ce
        )

        # Snapshot entropy before run.
        def collect_entropies(pt):
            entropies = []
            for slot in pt.pool._replicas:
                if isinstance(slot, WangLandauWindowGroup):
                    for r in slot._replicas:
                        entropies.append(dict(r.ensemble._entropy))
                else:
                    entropies.append(dict(slot.ensemble._entropy))
            return entropies

        before = collect_entropies(pt2)
        pt2.run(n_cycles=2)
        after = collect_entropies(pt2)

        for b, a in zip(before, after, strict=True):
            assert b == a, (
                "_entropy must be bit-identical after a frozen measurement run"
            )

    def test_measure_from_checkpoint_recorder_accumulates(self, tmp_path):
        """After attaching an observer and running, recorders accumulate moments."""
        from mchammer_pt.wl import WangLandauParallelTempering
        from mchammer_pt.wl_window_group import WangLandauWindowGroup

        ckpt, ce = self._make_and_save_checkpoint(tmp_path)
        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            ckpt, cluster_expansion=ce
        )
        obs = _ConstantObs(tag="meas")
        pt2.record_observable(obs)
        pt2.run(n_cycles=2)

        # At least one recorder should have accumulated samples in at least one bin.
        total_samples = 0
        for slot in pt2.pool._replicas:
            if isinstance(slot, WangLandauWindowGroup):
                replicas = slot._replicas
            else:
                replicas = [slot]
            for r in replicas:
                rec = r.ensemble._recorders.get("meas")
                if rec is not None:
                    total_samples += sum(rec._count.values())
        assert total_samples > 0, (
            "expected at least one moment to be accumulated during measurement run"
        )

    def test_measure_from_checkpoint_accepts_w2_pool(self, tmp_path):
        """measure_from_checkpoint correctly loads a W=2 checkpoint."""
        from mchammer_pt.wl import WangLandauParallelTempering
        from mchammer_pt.wl_window_group import WangLandauWindowGroup
        from tests._wl_fixtures import make_wl_atoms, make_wl_ce

        ce = make_wl_ce()
        atoms = make_wl_atoms()
        e0 = _initial_energy()
        lo, hi = e0 - 50.0, e0 + 50.0
        pt = WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[atoms, atoms],
            windows=[(lo, hi), (lo, hi)],
            energy_spacing=0.1,
            block_size=10,
            random_seed=0,
            n_walkers_per_window=2,
        )
        pt.run(n_cycles=2)
        ckpt = tmp_path / "w2.h5"
        pt.save_checkpoint(ckpt)

        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            ckpt, cluster_expansion=ce
        )
        assert pt2.pool._frozen_measurement is True
        for slot in pt2.pool._replicas:
            assert isinstance(slot, WangLandauWindowGroup)
            for r in slot._replicas:
                assert r.ensemble._frozen_g is True
