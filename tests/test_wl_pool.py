"""Tests for Wang-Landau pool implementations."""

from __future__ import annotations

import numpy as np
import pytest
from icet import ClusterExpansion

from mchammer_pt.parallel._comms import Reply, recv_reply, request
from mchammer_pt.parallel.processes import (
    ProcessWangLandauPool,
    _merge_per_window_stats,
)
from tests._in_process_worker import InProcessWorkerConn
from tests._wl_fixtures import (
    distinct_in_window_pair,
    make_wl_atoms,
    make_wl_ce,
)


def _make_wl_in_process_conn(
    ensemble_kwargs: dict | None = None,
    one_over_t_entry: str = "window_clock",
):
    """Build an :class:`InProcessWorkerConn` around a real WL replica.

    Uses the same seeds, energy limits and ensemble class as
    :func:`_spawn_wl_worker`, so an opcode test body can swap one
    helper for the other without further changes. Prefer this helper
    for tests that only need to exercise a worker handler; reach for
    :func:`_spawn_wl_worker` when the test must cross the real
    ``mp.Pipe`` pickle boundary.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble
    from mchammer_pt.wl_replica import WangLandauReplica

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    replica = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs=dict(ensemble_kwargs or {}),
        one_over_t_entry=one_over_t_entry,
    )
    return InProcessWorkerConn(replica)


def _spawn_wl_worker(tmp_path, ensemble_kwargs: dict | None = None):
    """Spawn a single _wl_worker and return (process, parent_conn).

    Caller is responsible for sending SHUTDOWN and joining.
    """
    import multiprocessing as mp

    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.parallel._builder import AtomsSpec, WLBuilder
    from mchammer_pt.parallel._worker import _wl_worker
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce, atoms = make_wl_ce(), make_wl_atoms()
    ce_path = tmp_path / "ce.ce"
    ce.write(str(ce_path))
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    builder = WLBuilder(
        ce_path=str(ce_path),
        atoms=AtomsSpec.from_atoms(atoms),
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs=dict(ensemble_kwargs or {}),
        recency_visits_per_bin=1000,
        dos_snapshot_ratio=2.0,
    )
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_wl_worker,
        args=(child_conn, builder),
        daemon=True,
    )
    process.start()
    child_conn.close()
    reply = parent_conn.recv()
    assert reply == Reply("OK", "STARTUP", None)
    return process, parent_conn


def _make_serial_wl_pool(n_replicas: int = 2):
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.parallel.serial import SerialWangLandauPool
    from mchammer_pt.wl_replica import WangLandauReplica

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers))
    replicas = [
        WangLandauReplica(
            cluster_expansion=ce, atoms=atoms,
            energy_spacing=0.1,
            energy_limit_left=e0 - 100.0 + i * 0.01,
            energy_limit_right=e0 + 100.0 + i * 0.01,
            random_seed=i,
        )
        for i in range(n_replicas)
    ]
    return SerialWangLandauPool(replicas, energy_spacing=0.1)


def test_serial_wl_pool_satisfies_protocol():
    from mchammer_pt.parallel.backend import ReplicaPool, WangLandauPool
    pool = _make_serial_wl_pool()
    assert isinstance(pool, ReplicaPool)
    assert isinstance(pool, WangLandauPool)


def test_serial_wl_pool_log_g_delegates_to_replicas():
    pool = _make_serial_wl_pool()
    e_i = pool.current_energy(0)
    assert pool.log_g(0, e_i) == 0.0
    assert pool.log_g(0, e_i + 1000.0) == -np.inf


def test_serial_wl_pool_converged_flags_initial_false():
    pool = _make_serial_wl_pool()
    flags = pool.converged_flags()
    assert flags.dtype == bool
    assert not flags.any()


def _wl_pool_factory_kwargs(tmp_path):
    """Common setup for ProcessWangLandauPool tests."""
    from mchammer.calculators import ClusterExpansionCalculator
    ce, atoms = make_wl_ce(), make_wl_atoms()
    ce_path = tmp_path / "ce.ce"
    ce.write(str(ce_path))
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    return ce_path, atoms, e0


def test_process_wl_pool_per_window_stats_returns_metrics(tmp_path):
    """per_window_stats() round-trips through WL_STATS opcode and returns
    fill_factor, halvings, histogram, bins_visited, bins_known,
    converged, phase, recency_flatness, and schedule for each window."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool
    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
    ) as pool:
        pool.advance_all(10)
        stats = pool.per_window_stats()

    assert len(stats) == 2
    for s in stats:
        assert set(s.keys()) == {
            "fill_factor", "halvings", "histogram",
            "bins_visited", "bins_filled", "bins_known",
            "converged", "phase", "recency_flatness", "schedule",
        }
        assert isinstance(s["fill_factor"], float)
        assert isinstance(s["halvings"], int)
        assert isinstance(s["histogram"], dict)
        assert isinstance(s["bins_visited"], int)
        assert isinstance(s["bins_filled"], int)
        assert isinstance(s["bins_known"], int)
        assert isinstance(s["converged"], bool)
        assert s["phase"] in {"halving", "1_over_t"}
        assert s["recency_flatness"] is None or isinstance(
            s["recency_flatness"], float
        )
        assert s["schedule"] in {"halving", "1_over_t"}
        assert s["fill_factor"] > 0.0


def test_process_wl_pool_propagates_flatness_limit_from_ensemble_kwargs(tmp_path):
    """``ensemble_kwargs={'flatness_limit': X}`` reaches the pooled-flatness gate.

    Before the fix, :class:`ProcessWangLandauWindow` hardcoded
    ``flatness_limit = 0.8``; a user passing
    ``ensemble_kwargs={'flatness_limit': 0.5}`` got the wrong
    pooled-flatness threshold on the process pool while the serial path
    used the configured 0.5.
    """
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms],
        windows=[(e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0],
        n_walkers_per_window=2,
        ensemble_kwargs={"flatness_limit": 0.5},
    ) as pool:
        assert pool._flatness_limit == 0.5
        assert pool._slots[0]._flatness_limit == 0.5


def test_wl_builder_threads_recency_visits_per_bin(tmp_path):
    """WLBuilder forwards recency_visits_per_bin to the built ensemble."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.parallel._builder import AtomsSpec, WLBuilder
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce, atoms = make_wl_ce(), make_wl_atoms()
    ce_path = tmp_path / "ce.ce"
    ce.write(str(ce_path))
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    builder = WLBuilder(
        ce_path=str(ce_path),
        atoms=AtomsSpec.from_atoms(atoms),
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={},
        recency_visits_per_bin=250,
        dos_snapshot_ratio=2.0,
    )
    replica = builder.build()
    assert replica.ensemble._recency_visits_per_bin == 250


def test_process_wl_pool_stores_recency_visits_per_bin(tmp_path):
    """ProcessWangLandauPool retains the recency_visits_per_bin it threads."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        recency_visits_per_bin=250,
    ) as pool:
        assert pool._recency_visits_per_bin == 250


def test_process_wl_pool_stores_dos_snapshot_ratio(tmp_path):
    """ProcessWangLandauPool retains the dos_snapshot_ratio it threads."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        dos_snapshot_ratio=4.0,
    ) as pool:
        assert pool._dos_snapshot_ratio == 4.0


def test_process_pool_rejects_nonpositive_recency_visits_per_bin(tmp_path):
    import pytest

    from mchammer_pt.parallel.processes import ProcessWangLandauPool
    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with pytest.raises(ValueError, match="recency_visits_per_bin"):
        ProcessWangLandauPool(
            ce_path=ce_path,
            initial_atoms=[atoms],
            windows=[(e0 - 50.0, e0 + 50.0)],
            energy_spacing=0.1,
            seeds=[0],
            recency_visits_per_bin=0,
        )


def test_merge_per_window_stats_single_walker_returns_payload_unchanged():
    """Single-walker slot: the per-walker stats dict is returned, with
    the internal ``visited_bins`` field (added by the worker for
    merge support) stripped from the output.
    """
    s = {
        "fill_factor": 0.5,
        "halvings": 1,
        "histogram": {0: 10, 1: 20},
        "bins_visited": 2,
        "bins_known": 2,
        "converged": False,
        "phase": "halving",
        "visited_bins": [0, 1],
    }
    out = _merge_per_window_stats([s], flatness_mode="pooled")
    assert set(out.keys()) == {
        "fill_factor", "halvings", "histogram",
        "bins_visited", "bins_known", "converged", "phase",
    }
    assert out["fill_factor"] == 0.5
    assert out["halvings"] == 1
    assert out["histogram"] == {0: 10, 1: 20}
    assert out["bins_visited"] == 2
    assert out["bins_known"] == 2
    assert out["converged"] is False


def test_serial_wl_pool_swap_configurations_refreshes_window_flag():
    """After a swap, each replica's _reached_energy_window is True."""
    pool = _make_serial_wl_pool(n_replicas=2)
    # Drive replicas to different occupations so swap is meaningful.
    r0, r1 = pool._replicas[0], pool._replicas[1]
    occ = r1.current_occupations().copy()
    occ[[0, -1]] = occ[[-1, 0]]
    r1.set_occupations(occ)

    pool.swap_configurations(0, 1)

    assert r0.ensemble._reached_energy_window is True
    assert r1.ensemble._reached_energy_window is True


def test_serial_wl_pool_satisfies_observable_protocol():
    from mchammer_pt.parallel.backend import WangLandauObservablePool
    pool = _make_serial_wl_pool()
    assert isinstance(pool, WangLandauObservablePool)


def test_serial_wl_pool_attach_observer_fires():
    """Observer attached to a SerialWangLandauPool fires inside advance(...)."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial_wl_pool(n_replicas=2)
    pool.attach_observer(StatefulCounter(interval=10), replicas=[0, 1])
    pool.advance_all(50)
    dcs = pool.data_containers()
    assert "counter" in dcs[0].data.columns
    assert "counter" in dcs[1].data.columns


def test_serial_wl_pool_attach_observer_gives_independent_copies():
    """Each replica gets its own deserialised counter, not a shared instance."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial_wl_pool(n_replicas=2)
    template = StatefulCounter(interval=10)
    pool.attach_observer(template, replicas="all")
    pool.advance_all(50)
    # The template never had `get_observable` called on it.
    assert template.n_calls == 0
    obs0 = pool.get_observers(0)
    obs1 = pool.get_observers(1)
    # Each replica's observer is a distinct object (pickle round-trip).
    assert obs0["counter"] is not obs1["counter"]


def test_serial_wl_pool_get_observers_round_trips():
    """get_observers returns a pickle-roundtripped copy keyed by tag."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial_wl_pool(n_replicas=2)
    pool.attach_observer(StatefulCounter(interval=10))
    snapshot = pool.get_observers(0)
    assert "counter" in snapshot
    assert isinstance(snapshot["counter"], StatefulCounter)


def test_process_wl_pool_attach_observer_fires(tmp_path):
    """attach_observer on the process pool reaches each worker."""
    from tests._in_process_pool import make_in_process_wl_pool
    from tests._observer_fixtures import StatefulCounter

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        seeds=[0, 1],
    ) as pool:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0, 1])
        snapshot = pool.get_observers(0)
        assert "counter" in snapshot
        assert isinstance(snapshot["counter"], StatefulCounter)


def test_process_wl_pool_current_energies_primed_before_advance(tmp_path):
    """current_energies() returns true initial energies before any advance_all.

    ``run()`` writes ``energies_per_cycle[0]`` from ``current_energies()``
    before the first block. The cache is primed at construction, so row 0
    records the real initial energies, not the 0.0 placeholder.
    """
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        seeds=[0, 1],
    ) as pool:
        live = np.array([
            float(request(pool._slots[i].workers[0][1], ("ENERGY",), i))
            for i in range(len(pool))
        ])
        primed = pool.current_energies()
        np.testing.assert_array_equal(primed, live)
        assert not np.all(primed == 0.0)


def test_process_wl_pool_apply_swaps_updates_energy_cache(tmp_path):
    """apply_swaps keeps the parent-side energy cache consistent with the swap.

    ``walker_energy`` / ``current_energies`` read the block-boundary cache
    with no IPC, so ``apply_swaps`` must swap the cached energies of the
    swapped walkers — otherwise the per-window energy trace would record a
    pre-swap energy and diverge from the serial backend (which reads the
    live potential).
    """
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        seeds=[0, 1],
    ) as pool:
        # Populate the parent-side cache from a block of MC; the two
        # single-walker windows diverge to distinct energies.
        pool.advance_all(50)
        e_cache_0 = pool.walker_energy(0, 0)
        e_cache_1 = pool.walker_energy(1, 0)
        assert e_cache_0 != e_cache_1, (
            "test setup did not produce distinct cached energies"
        )

        pool.apply_swaps([(0, 0, 1, 0)])

        # Cache (no IPC) now reflects the swapped energies, and the
        # per-window trace agrees.
        assert pool.walker_energy(0, 0) == e_cache_1
        assert pool.walker_energy(1, 0) == e_cache_0
        assert pool.current_energies()[0] == e_cache_1
        assert pool.current_energies()[1] == e_cache_0


def test_process_wl_pool_apply_swaps_cross_index_multi_swap_cache(tmp_path):
    """A batch of disjoint cross-index swaps keeps every walker's cached
    energy aligned with the configuration it now holds.

    The single-pair tests would miss a transposition or read-after-mutate
    bug in the cache fix-up; this drives two simultaneous ``a != b`` swaps
    and checks both the cache permutation and live-worker agreement.
    """
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        n_walkers_per_window=2,
        seeds=[0, 1],
    ) as pool:
        pool.advance_all(50)
        pre = {
            (i, w): pool.walker_energy(i, w)
            for i in range(len(pool))
            for w in range(pool.n_walkers(i))
        }
        # Two disjoint cross-index swaps covering all four walkers.
        swaps = [(0, 0, 1, 1), (0, 1, 1, 0)]
        pool.apply_swaps(swaps)

        # Cache fix-up: each swapped walker now caches its partner's energy
        # (exact permutation of the recorded pre-swap cache values).
        for i, a, j, b in swaps:
            assert pool.walker_energy(i, a) == pre[(j, b)]
            assert pool.walker_energy(j, b) == pre[(i, a)]

        # The cache agrees with the live worker (configs physically moved).
        # `approx` absorbs the running-total vs fresh-recompute float
        # difference between the MC potential and post-SET_OCC recompute.
        for i in range(len(pool)):
            for w in range(pool.n_walkers(i)):
                live = float(
                    request(pool._slots[i].workers[w][1], ("ENERGY",), i)
                )
                assert pool.walker_energy(i, w) == pytest.approx(live)


def test_process_wl_pool_apply_swaps_rejects_overlapping(tmp_path):
    """apply_swaps fails loudly if a walker appears in two swaps.

    The matching exchange guarantees disjoint swaps; an overlap would
    silently overwrite the occupations dict and move the wrong configs.
    """
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        n_walkers_per_window=2,
        seeds=[0, 1],
    ) as pool:
        pool.advance_all(10)
        with pytest.raises(ValueError, match="overlapping swaps"):
            # walker (0, 0) appears in both swaps.
            pool.apply_swaps([(0, 0, 1, 0), (0, 0, 1, 1)])


def test_process_wl_pool_swap_configurations_refreshes_worker_state(tmp_path):
    """After a swap, each worker's _potential reflects the new configuration."""
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        seeds=[0, 1],
    ) as pool:
        # Drive the two replicas to different occupation vectors.
        occ0 = pool.current_occupations(0).copy()
        occ1 = occ0.copy()
        occ1[[0, -1]] = occ1[[-1, 0]]  # swap first/last to differ

        # Direct SET_OCC on worker 1 so its configuration is distinct.
        _, conn1 = pool._slots[1].workers[0]
        request(conn1, ("SET_OCC", np.asarray(occ1, dtype=np.int64)), 1)

        e_before_0 = pool.current_energy(0)
        e_before_1 = pool.current_energy(1)
        assert e_before_0 != e_before_1, (
            "test setup did not produce distinct energies"
        )

        # Now swap. Each worker should refresh its cached _potential
        # to match the swapped-in configuration.
        pool.swap_configurations(0, 1)
        e_after_0 = pool.current_energy(0)
        e_after_1 = pool.current_energy(1)

        # The swap delivered configurations across; energies swap accordingly.
        assert e_after_0 == pytest.approx(e_before_1)
        assert e_after_1 == pytest.approx(e_before_0)

        # After the swap, both replicas should be flagged in-window
        # (their swapped-in energies were in the wide pool window).
        # Probing log_g at the current energy returns finite iff
        # _reached_energy_window is True and the bin is in-window.
        assert pool.log_g(0, e_after_0) != -float("inf")
        assert pool.log_g(1, e_after_1) != -float("inf")


def _make_wl_pool_with_groups(n_windows: int = 2, n_walkers: int = 2):
    """SerialWangLandauPool whose slots are WangLandauWindowGroup instances."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.parallel.serial import SerialWangLandauPool
    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    groups = [
        WangLandauWindowGroup(
            [
                WangLandauReplica(
                    cluster_expansion=ce,
                    atoms=atoms,
                    energy_spacing=0.1,
                    energy_limit_left=e0 - 100.0,
                    energy_limit_right=e0 + 100.0,
                    random_seed=w * n_walkers + j,
                )
                for j in range(n_walkers)
            ],
            random_seed=w,
        )
        for w in range(n_windows)
    ]
    return SerialWangLandauPool(groups, energy_spacing=0.1)


def test_per_window_data_containers_single_replica_slots():
    """per_window_data_containers wraps each WangLandauReplica in a single-item list."""
    pool = _make_serial_wl_pool(n_replicas=2)
    result = pool.per_window_data_containers()
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1


def test_per_window_data_containers_window_group_slots():
    """per_window_data_containers returns W containers per WindowGroup slot."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=3)
    result = pool.per_window_data_containers()
    assert len(result) == 2
    assert len(result[0]) == 3
    assert len(result[1]) == 3


def test_per_window_stats_with_window_group_slots():
    """per_window_stats works when slots are WangLandauWindowGroup instances."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=2)
    stats = pool.per_window_stats()
    assert len(stats) == 2
    for s in stats:
        assert "fill_factor" in s
        assert "halvings" in s
        assert "histogram" in s
        assert "converged" in s


def test_serial_wl_pool_attach_observer_class_dispatches_to_all_walkers():
    """attach_observer_class gives each walker in a WindowGroup its own observer."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_wl_pool_with_groups(n_windows=1, n_walkers=2)
    pool.attach_observer_class(StatefulCounter, interval=10)
    pool.advance_all(30)

    for r in pool._replicas[0]._replicas:
        assert "counter" in r.ensemble.observers
    observer_ids = {
        id(r.ensemble.observers["counter"])
        for r in pool._replicas[0]._replicas
    }
    assert len(observer_ids) == len(pool._replicas[0]._replicas)


def test_serial_wl_pool_swap_configurations_with_window_groups():
    """swap_configurations exchanges walker 0's occupations across groups."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=2)

    occ0_before = pool.current_occupations(0).copy()
    occ1_before = pool.current_occupations(1).copy()

    pool.swap_configurations(0, 1)

    assert np.array_equal(pool.current_occupations(0), occ1_before)
    assert np.array_equal(pool.current_occupations(1), occ0_before)


def test_serial_wl_pool_trace_tracks_walker_zero():
    """The per-window trace deterministically reads walker 0."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=2)
    pool.advance_all(30)
    # Drive walker 1 to a configuration distinct from walker 0 so a trace
    # that read any other walker would diverge from the walker-0 trace.
    for slot in pool._replicas:
        occ = slot.walker_occupations(0).copy()
        occ[[0, -1]] = occ[[-1, 0]]
        slot.set_walker_occupations(1, occ)

    energies = pool.current_energies()
    for i in range(len(pool)):
        assert energies[i] == pool.walker_energy(i, 0)
        assert (
            pool.current_occupations(i) == pool._replicas[i].walker_occupations(0)
        ).all()


def test_serial_pool_resolves_bins_filled_by_mode():
    """per_window_stats collapses the candidate counts into ``bins_filled``.

    A multi-walker slot's ``window_stats`` returns ``bins_filled_pooled``
    and ``bins_filled_per_walker`` but no singular ``bins_filled``. The
    serial pool resolves these against its ``flatness_mode``, leaving a
    single ``bins_filled`` and stripping the candidates.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.parallel.serial import SerialWangLandauPool
    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    group = WangLandauWindowGroup(
        [
            WangLandauReplica(
                cluster_expansion=ce,
                atoms=atoms,
                energy_spacing=0.1,
                energy_limit_left=e0 - 100.0,
                energy_limit_right=e0 + 100.0,
                random_seed=j,
            )
            for j in range(2)
        ],
        random_seed=0,
    )
    pool = SerialWangLandauPool(
        [group], energy_spacing=0.1, flatness_mode="per_walker"
    )

    pool.advance_all(10)
    stats = pool.per_window_stats()
    s = stats[0]
    assert "bins_filled" in s
    assert "bins_filled_pooled" not in s
    assert "bins_filled_per_walker" not in s
    assert isinstance(s["bins_filled"], int)
    assert s["flatness_mode"] == "per_walker"


def test_serial_pool_resolves_recency_flatness_by_mode():
    """per_window_stats collapses the recency-flatness candidates.

    A multi-walker slot's ``window_stats`` returns
    ``recency_flatness_pooled`` and ``recency_flatness_per_walker`` but
    no singular ``recency_flatness``. The serial pool resolves these
    against its ``flatness_mode``, leaving a single ``recency_flatness``
    and stripping the candidates.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.parallel.serial import SerialWangLandauPool
    from mchammer_pt.wl_replica import WangLandauReplica
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    group = WangLandauWindowGroup(
        [
            WangLandauReplica(
                cluster_expansion=ce,
                atoms=atoms,
                energy_spacing=0.1,
                energy_limit_left=e0 - 100.0,
                energy_limit_right=e0 + 100.0,
                random_seed=j,
            )
            for j in range(2)
        ],
        random_seed=0,
    )
    pool = SerialWangLandauPool(
        [group], energy_spacing=0.1, flatness_mode="per_walker"
    )

    pool.advance_all(10)
    s = pool.per_window_stats()[0]
    assert "recency_flatness" in s
    assert "recency_flatness_pooled" not in s
    assert "recency_flatness_per_walker" not in s


def test_process_wl_pool_multi_walker_slots_structure(tmp_path):
    """n_walkers_per_window=2 creates 2 workers per slot."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=2,
    ) as pool:
        assert len(pool) == 2                 # 2 windows
        assert len(pool._slots[0].workers) == 2   # 2 walkers in window 0
        assert len(pool._slots[1].workers) == 2   # 2 walkers in window 1


def test_process_wl_pool_mixed_walkers_per_window(tmp_path):
    """n_walkers_per_window=[1, 2] gives 1 and 2 workers per slot."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=[1, 2],
    ) as pool:
        assert len(pool._slots[0].workers) == 1
        assert len(pool._slots[1].workers) == 2


def test_process_wl_pool_multi_walker_converged_requires_all_walkers(tmp_path):
    """converged_flags is False for a slot unless all walkers are converged."""
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    # fill_factor_limit=0.5 means a single halving (1.0 -> 0.5) converges.
    # We drive halving directly via FORCE_HALVE on individual workers to
    # decouple this test from coordinator dynamics.
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 0.05, e0 + 0.05)],
        seeds=[0],
        n_walkers_per_window=2,
        ensemble_kwargs={"fill_factor_limit": 0.5},
    ) as pool:
        _, conn0 = pool._slots[0].workers[0]
        _, conn1 = pool._slots[0].workers[1]

        assert not pool.converged_flags()[0]

        # Halve only walker 0; slot must still be reported unconverged.
        request(conn0, ("FORCE_HALVE",), 0)
        assert bool(request(conn0, ("CONVERGED",), 0))
        assert not bool(request(conn1, ("CONVERGED",), 0))
        assert not pool.converged_flags()[0]

        # Halve walker 1 as well; slot now converges.
        request(conn1, ("FORCE_HALVE",), 0)
        assert pool.converged_flags()[0]


def test_merge_per_window_stats_multi_walker_sums_histograms():
    """Multi-walker slot: histogram is summed; bins_visited is the size
    of the union of per-walker ``visited_bins`` lists; bins_known is
    the size of the combined-histogram key set; fill_factor/halvings
    come from walker 0; converged is the AND across walkers;
    flatness_mode and per_walker_flat_min are attached.
    """
    s0 = {
        "fill_factor": 0.5,
        "halvings": 2,
        "histogram": {0: 10, 1: 20, 2: 30},
        "bins_visited": 3,
        "bins_known": 3,
        "converged": False,
        "phase": "halving",
        "schedule": "halving",
        "visited_bins": [0, 1, 2],
        "recency_weights": {0: 1.0, 1: 1.0, 2: 1.0},
    }
    s1 = {
        "fill_factor": 0.25,
        "halvings": 3,
        "histogram": {1: 5, 2: 15, 3: 25},
        "bins_visited": 3,
        "bins_known": 3,
        "converged": True,
        "phase": "halving",
        "schedule": "halving",
        "visited_bins": [1, 2, 3],
        "recency_weights": {1: 1.0, 2: 1.0, 3: 1.0},
    }
    out = _merge_per_window_stats([s0, s1], flatness_mode="pooled")
    assert out["fill_factor"] == 0.5
    assert out["halvings"] == 2
    assert out["histogram"] == {0: 10, 1: 25, 2: 45, 3: 25}
    # Union of visited_bins: {0, 1, 2, 3} → 4.
    assert out["bins_visited"] == 4
    # Union of histogram keys: {0, 1, 2, 3} → 4.
    assert out["bins_known"] == 4
    assert out["converged"] is False
    assert out["flatness_mode"] == "pooled"
    assert "per_walker_flat_min" in out
    assert out["phase"] == "halving"
    # The internal field is stripped from the user-facing dict.
    assert "visited_bins" not in out


def test_merge_per_window_stats_propagates_1_over_t_phase():
    """A walker whose phase has flipped to 1_over_t propagates through
    the multi-walker merge unchanged. ProgressPrinter uses this field
    to distinguish a stalled halving phase from a 1/t plateau.
    """
    s0 = {
        "fill_factor": 1e-7,
        "halvings": 24,
        "histogram": {0: 100, 1: 100},
        "bins_visited": 2,
        "bins_known": 2,
        "converged": False,
        "phase": "1_over_t",
        "schedule": "1_over_t",
        "visited_bins": [0, 1],
        "recency_weights": {0: 1.0, 1: 1.0},
    }
    s1 = {**s0, "histogram": {0: 110, 1: 90}, "visited_bins": [0, 1]}
    out = _merge_per_window_stats([s0, s1], flatness_mode="pooled")
    assert out["phase"] == "1_over_t"


def test_merge_per_window_stats_adds_filled_and_breakdown():
    from mchammer_pt.parallel.processes import _merge_per_window_stats
    slot_stats = [
        {
            "fill_factor": 1.0, "halvings": 0,
            "histogram": {0: 1, 1: 4, 2: 0},
            "bins_visited": 2, "bins_filled": 2, "bins_known": 3,
            "converged": False, "phase": "halving", "schedule": "halving",
            "visited_bins": [0, 1],
            "recency_weights": {0: 1.0, 1: 1.0, 2: 0.0},
        },
        {
            "fill_factor": 1.0, "halvings": 0,
            "histogram": {0: 3, 1: 0, 2: 7},
            "bins_visited": 2, "bins_filled": 2, "bins_known": 3,
            "converged": False, "phase": "halving", "schedule": "halving",
            "visited_bins": [0, 2],
            "recency_weights": {0: 1.0, 1: 0.0, 2: 1.0},
        },
    ]
    merged = _merge_per_window_stats(slot_stats, "per_walker")
    assert merged["bins_filled"] == 1            # intersection {0}
    assert merged["per_walker_breakdown"] == [
        {"filled": 2, "known": 3, "flat_min": 0.0},
        {"filled": 2, "known": 3, "flat_min": 0.0},
    ]
    pooled = _merge_per_window_stats(slot_stats, "pooled")
    assert pooled["bins_filled"] == 3            # union {0,1,2}


def test_merge_per_window_stats_single_walker_carries_bins_filled():
    from mchammer_pt.parallel.processes import _merge_per_window_stats
    slot_stats = [{
        "fill_factor": 1.0, "halvings": 0,
        "histogram": {0: 1, 1: 0},
        "bins_visited": 1, "bins_filled": 1, "bins_known": 2,
        "converged": False, "phase": "halving",
        "visited_bins": [0],
    }]
    merged = _merge_per_window_stats(slot_stats, "pooled")
    assert merged["bins_filled"] == 1
    assert "per_walker_breakdown" not in merged
    assert "visited_bins" not in merged


def test_merge_per_window_stats_adds_recency_flatness():
    from mchammer_pt.parallel.processes import _merge_per_window_stats
    base = {
        "fill_factor": 1.0, "halvings": 0, "histogram": {0: 1, 1: 1},
        "bins_visited": 2, "bins_filled": 2, "bins_known": 2,
        "converged": False, "phase": "halving", "schedule": "1_over_t",
        "visited_bins": [0, 1],
    }
    s0 = {**base, "recency_weights": {0: 1.0, 1: 0.0}}
    s1 = {**base, "recency_weights": {0: 0.0, 1: 1.0}}
    merged = _merge_per_window_stats([s0, s1], "pooled")
    # summed {0:1,1:1} -> min/mean = 1.0
    assert merged["recency_flatness"] == 1.0
    assert "recency_weights" not in merged
    assert merged["schedule"] == "1_over_t"
    pw = _merge_per_window_stats([s0, s1], "per_walker")
    assert pw["recency_flatness"] == 0.0   # each walker has a zero bin


def test_merge_single_walker_carries_recency_flatness():
    from mchammer_pt.parallel.processes import _merge_per_window_stats
    s = {
        "fill_factor": 1.0, "halvings": 0, "histogram": {0: 1},
        "bins_visited": 1, "bins_filled": 1, "bins_known": 1,
        "converged": False, "phase": "halving", "schedule": "halving",
        "recency_flatness": 1.0, "recency_weights": {0: 1.0},
        "visited_bins": [0],
    }
    merged = _merge_per_window_stats([s], "pooled")
    assert merged["recency_flatness"] == 1.0
    assert "recency_weights" not in merged


def test_serial_and_process_recency_flatness_agree():
    """Serial and process backends resolve the same recency_flatness.

    The serial path (window group → ``_compute_recency_flatness``) and the
    process path (``_merge_per_window_stats`` → ``_compute_recency_flatness``)
    are distinct code routes, but both must collapse identical per-walker
    EWMA weight dicts to the same value under each flatness mode.
    """
    from mchammer_pt.parallel.processes import _merge_per_window_stats
    from mchammer_pt.wl_coordinator import _compute_recency_flatness

    weights = [{0: 3.0, 1: 1.0, 2: 0.0}, {0: 0.0, 1: 2.0, 2: 2.0}]
    for mode in ("pooled", "per_walker"):
        serial_value = _compute_recency_flatness(weights, mode)
        slot_stats = [
            {
                "fill_factor": 1.0, "halvings": 0,
                "histogram": {0: 1, 1: 1, 2: 1},
                "bins_visited": 3, "bins_filled": 3, "bins_known": 3,
                "converged": False, "phase": "halving", "schedule": "halving",
                "visited_bins": [0, 1, 2], "recency_weights": w,
            }
            for w in weights
        ]
        process_value = _merge_per_window_stats(slot_stats, mode)[
            "recency_flatness"
        ]
        assert serial_value == process_value


def test_process_wl_pool_multi_walker_per_window_stats_merges_histograms(tmp_path):
    """per_window_stats sums histograms across walkers; fill_factor from walker 0."""
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0)],
        seeds=[0],
        n_walkers_per_window=2,
    ) as pool:
        # Advance both workers directly so histograms accumulate without
        # entropy-sync zeroing
        _, conn0 = pool._slots[0].workers[0]
        _, conn1 = pool._slots[0].workers[1]
        conn0.send(("ADVANCE", 20))
        conn1.send(("ADVANCE", 20))
        recv_reply(conn0, "ADVANCE", "window 0 walker 0")
        recv_reply(conn1, "ADVANCE", "window 0 walker 1")

        # Collect individual WL_STATS from both workers
        s0 = request(conn0, ("WL_STATS",), 0)
        s1 = request(conn1, ("WL_STATS",), 1)

        stats = pool.per_window_stats()
        assert len(stats) == 1

        # fill_factor from walker 0
        assert stats[0]["fill_factor"] == pytest.approx(s0["fill_factor"])

        # histogram is sum of both workers' histograms
        for bin_key in set(s0["histogram"]) | set(s1["histogram"]):
            expected = s0["histogram"].get(bin_key, 0) + s1["histogram"].get(bin_key, 0)
            assert stats[0]["histogram"].get(bin_key, 0) == expected


def test_process_wl_pool_multi_walker_stats_report_union_bin_counts(tmp_path):
    """For a multi-walker slot, per_window_stats reports bins_visited as
    the size of the union of MC-visited bins across walkers, and
    bins_known as the size of the union of histogram keys.
    """
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0)],
        seeds=[0],
        n_walkers_per_window=2,
    ) as pool:
        _, conn0 = pool._slots[0].workers[0]
        _, conn1 = pool._slots[0].workers[1]
        conn0.send(("ADVANCE", 20))
        conn1.send(("ADVANCE", 20))
        recv_reply(conn0, "ADVANCE", "window 0 walker 0")
        recv_reply(conn1, "ADVANCE", "window 0 walker 1")
        s0 = request(conn0, ("WL_STATS",), 0)
        s1 = request(conn1, ("WL_STATS",), 1)

        stats = pool.per_window_stats()

    assert len(stats) == 1
    expected_known = len(set(s0["histogram"]) | set(s1["histogram"]))
    expected_visited = len(set(s0["visited_bins"]) | set(s1["visited_bins"]))
    assert stats[0]["bins_visited"] == expected_visited
    assert stats[0]["bins_known"] == expected_known
    # The internal field is stripped from the user-facing dict.
    assert "visited_bins" not in stats[0]

    # bins_filled resolves through the real worker -> merge path
    # (pooled default = union of per-walker positive bins), and the
    # per-walker breakdown carries one entry per walker.
    expected_filled = len(
        {b for b, c in s0["histogram"].items() if c > 0}
        | {b for b, c in s1["histogram"].items() if c > 0}
    )
    assert stats[0]["bins_filled"] == expected_filled
    breakdown = stats[0]["per_walker_breakdown"]
    assert len(breakdown) == 2
    assert all(set(e) == {"filled", "known", "flat_min"} for e in breakdown)
    expected_pairs = sorted(
        (sum(1 for c in s["histogram"].values() if c > 0), len(s["histogram"]))
        for s in (s0, s1)
    )
    got_pairs = sorted((e["filled"], e["known"]) for e in breakdown)
    assert got_pairs == expected_pairs


def test_process_wl_pool_multi_walker_per_window_data_containers(tmp_path):
    """per_window_data_containers returns W containers per slot."""
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        seeds=[0, 1],
        n_walkers_per_window=[1, 2],
    ) as pool:
        result = pool.per_window_data_containers()
        assert len(result) == 2
        assert len(result[0]) == 1   # window 0: 1 walker
        assert len(result[1]) == 2   # window 1: 2 walkers


def test_process_wl_pool_at_halve_cadence_skips_non_halve_merge(tmp_path):
    """merge_cadence='at_halve' + no halve event ⇒ no inter-walker merge."""
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    with make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0)],
        seeds=[0],
        n_walkers_per_window=2,
        flatness_mode="per_walker",
        merge_cadence="at_halve",
    ) as pool:
        # Push divergent entropies; if any merge happens during
        # advance(0) (no MC, so no halve possible), the two walker
        # entropies would converge. Under merge_cadence='at_halve' with
        # no halve event, they must stay distinct.
        slot = pool._slots[0]
        _, c0 = slot.workers[0]
        _, c1 = slot.workers[1]
        request(c0, ("SET_ENTROPY", {0: 2.0, 1: 4.0}), 0)
        request(c1, ("SET_ENTROPY", {0: 6.0, 1: 8.0}), 1)
        # Mark walkers not-flat by zeroing histograms (so collective
        # halve cannot fire).
        pool.advance_all(0)
        e0_dict = request(c0, ("GET_ENTROPY",), 0)
        e1_dict = request(c1, ("GET_ENTROPY",), 1)
        assert e0_dict != e1_dict


def test_wl_worker_advance_ack_carries_state(tmp_path):
    """ADVANCE ack returns a WalkerPostBlockState with typed fields."""
    from mchammer_pt.wl_coordinator import WalkerPostBlockState

    process, conn = _spawn_wl_worker(tmp_path)
    try:
        payload = request(conn, ("ADVANCE", 50), 0)
        assert isinstance(payload, WalkerPostBlockState)
        assert isinstance(payload.halving_criterion_met, bool)
        assert isinstance(payload.fill_factor, float)
        assert isinstance(payload.entropy, dict)
        assert isinstance(payload.step, int)
        assert payload.window_entry_step is None or isinstance(
            payload.window_entry_step, int
        )
        assert isinstance(payload.histogram, dict)
        assert isinstance(payload.reached_energy_window, bool)
    finally:
        request(conn, ("SHUTDOWN",), 0)
        process.join(timeout=5.0)


def test_wl_worker_get_entropy_round_trip():
    """GET_ENTROPY returns the worker's current entropy dict."""
    conn = _make_wl_in_process_conn()
    # Run some MC so entropy is non-empty.
    request(conn, ("ADVANCE", 50), 0)
    payload = request(conn, ("GET_ENTROPY",), 0)
    assert isinstance(payload, dict)
    assert all(isinstance(k, int) for k in payload)
    assert all(isinstance(v, float) for v in payload.values())


def test_wl_worker_set_entropy_overwrites_in_place():
    """SET_ENTROPY replaces ensemble._entropy."""
    conn = _make_wl_in_process_conn()
    merged = {0: 1.5, 1: 2.5}
    request(conn, ("SET_ENTROPY", merged), 0)
    got = request(conn, ("GET_ENTROPY",), 0)
    assert got == merged


def test_wl_worker_force_halve_round_trip():
    """FORCE_HALVE halves fill_factor and grows history."""
    conn = _make_wl_in_process_conn()
    before = request(conn, ("ADVANCE", 50), 0)
    request(conn, ("FORCE_HALVE",), 0)
    after = request(conn, ("ADVANCE", 0), 0)
    assert after.fill_factor == pytest.approx(before.fill_factor / 2.0)


def test_wl_worker_finalise_merge_writes_entropy_and_refreshes_last_state():
    """FINALISE_MERGE writes the supplied dict to _entropy and refreshes _last_state."""
    conn = _make_wl_in_process_conn()
    # Run some MC so the data container exists and _last_state is initialised.
    request(conn, ("ADVANCE", 50), 0)
    merged = {0: 0.0, 1: 5.0}
    request(conn, ("FINALISE_MERGE", merged), 0)
    # Entropy on the ensemble matches the supplied dict.
    got = request(conn, ("GET_ENTROPY",), 0)
    assert got == merged
    # The data container's _last_state was refreshed with the merged values.
    dc = request(conn, ("GET_DC",), 0)
    assert dict(dc._last_state["entropy"]) == merged


def test_wl_worker_set_phase_round_trip():
    """SET_PHASE switches ensemble._phase under the 1/t schedule."""
    # SET_PHASE -> "1_over_t" is only meaningful for schedule="1_over_t"
    # because that schedule records ``_window_entry_step``, which the
    # autonomous 1/t-phase ``_fill_factor = 1/t`` update requires. In
    # production the coordinator only fires SET_PHASE under the same
    # schedule gate.
    conn = _make_wl_in_process_conn(ensemble_kwargs={"schedule": "1_over_t"})
    # Advance to record _window_entry_step on the first in-window step.
    request(conn, ("ADVANCE", 50), 0)
    request(conn, ("SET_PHASE", "1_over_t"), 0)
    # Subsequent ADVANCE in 1_over_t phase tracks 1/t.
    after = request(conn, ("ADVANCE", 5), 0)
    assert isinstance(after.fill_factor, float)


def test_wl_worker_set_phase_f_continuous_records_origin():
    """SET_PHASE delegates to ``switch_to_phase``: under f_continuous
    the schedule-clock origin is recorded and f is unchanged."""
    conn = _make_wl_in_process_conn(
        ensemble_kwargs={"schedule": "1_over_t"},
        one_over_t_entry="f_continuous",
    )
    request(conn, ("ADVANCE", 50), 0)
    before = request(conn, ("ADVANCE", 0), 0)
    request(conn, ("SET_PHASE", "1_over_t"), 0)
    e = conn._worker._replica.ensemble
    assert e._one_over_t_origin_step is not None
    assert e._fill_factor == before.fill_factor


def test_serial_pool_finalise_for_reporting_merges_walker_entropies():
    """SerialWangLandauPool.finalise_for_reporting merges per-window."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=2)

    # Assign distinct per-walker entropies. The two walkers in each
    # window share key set {0, 1} so intersection-mean rebasing is
    # well-defined; the second walker is offset by 10 so the merge
    # is observably non-trivial.
    for group in pool._replicas:
        group._replicas[0].ensemble._entropy = {0: 0.0, 1: 5.0}
        group._replicas[1].ensemble._entropy = {0: 10.0, 1: 15.0}

    pool.finalise_for_reporting()

    # After merging, both walkers in each window share the same dict.
    # Intersection-mean rebasing on {0: 0.0, 1: 5.0} subtracts the
    # mean (2.5) giving {0: -2.5, 1: 2.5}; the same shape comes out of
    # the offset walker. Bin-wise average yields {0: -2.5, 1: 2.5},
    # then post-shift to min == 0 gives {0: 0.0, 1: 5.0}.
    expected = {0: 0.0, 1: 5.0}
    for group in pool._replicas:
        for r in group._replicas:
            assert r.ensemble._entropy == expected


def test_serial_pool_finalise_for_reporting_skips_single_walker_slots():
    """Bare WangLandauReplica slots have no finaliser; pool tolerates that."""
    pool = _make_serial_wl_pool(n_replicas=2)
    # Set an entropy on each replica that would not survive any
    # implicit merge; absent a finaliser the value must be untouched.
    pool._replicas[0].ensemble._entropy = {0: 1.0, 1: 2.0}
    pool._replicas[1].ensemble._entropy = {0: 3.0, 1: 4.0}

    pool.finalise_for_reporting()

    assert pool._replicas[0].ensemble._entropy == {0: 1.0, 1: 2.0}
    assert pool._replicas[1].ensemble._entropy == {0: 3.0, 1: 4.0}


def test_process_pool_finalise_for_reporting_merges_walker_entropies(tmp_path):
    """ProcessWangLandauPool.finalise_for_reporting merges via IPC."""
    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=2,
    ) as pool:
        # Push distinct per-walker entropies via raw SET_ENTROPY, then
        # refresh the coordinator-side snapshot via ADVANCE(0) so
        # slot.walker_states carries the new values for the merge to read.
        for slot in pool._slots:
            _, c0 = slot.workers[0]
            _, c1 = slot.workers[1]
            request(c0, ("SET_ENTROPY", {0: 0.0, 1: 5.0}), 0)
            request(c1, ("SET_ENTROPY", {0: 10.0, 1: 15.0}), 1)
        pool.advance_all(0)

        pool.finalise_for_reporting()

        expected = {0: 0.0, 1: 5.0}
        for i, slot in enumerate(pool._slots):
            for w, (_, conn) in enumerate(slot.workers):
                got = request(conn, ("GET_ENTROPY",), f"window {i} walker {w}")
                assert got == expected


def test_serial_wl_pool_snapshot_returns_per_walker_and_group_state():
    """For a pool with one W=1 slot and one W=2 slot, the snapshot
    returns a dict with M=3 per_walker entries and N=2 group_state
    entries (None for the W=1 slot)."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed  # [1, 2]

    pool = make_serial_wl_pool_mixed()
    snap = pool.snapshot_for_checkpoint()
    assert set(snap.keys()) == {"per_walker", "group_state"}
    assert len(snap["per_walker"]) == 3
    assert len(snap["group_state"]) == 2
    assert snap["group_state"][0] is None  # W=1 slot
    assert isinstance(snap["group_state"][1], dict)
    assert set(snap["group_state"][1].keys()) == {"rng_state", "phase"}


def test_serial_wl_pool_restore_round_trips_via_snapshot():
    """Restore takes the same dict snapshot_for_checkpoint produces."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool_a = make_serial_wl_pool_mixed()
    snap = pool_a.snapshot_for_checkpoint()
    containers = pool_a.data_containers()

    pool_b = make_serial_wl_pool_mixed()  # fresh; same construction args
    # Drift pool_b so its W=2 slot's group RNG advances.
    pool_b._replicas[1]._rng.integers(0, 100, size=5)

    pool_b.restore_replica_state(
        containers=containers,
        per_walker_extras=snap["per_walker"],
        group_state=snap["group_state"],
    )
    # The W>1 slot's group RNG must now produce identical draws.
    assert (
        pool_a._replicas[1]._rng.integers(0, 100)
        == pool_b._replicas[1]._rng.integers(0, 100)
    )


def test_serial_wl_pool_restore_rejects_wrong_lengths():
    """Wrong-length containers / per_walker_extras / group_state all raise."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool = make_serial_wl_pool_mixed()
    snap = pool.snapshot_for_checkpoint()
    containers = pool.data_containers()

    with pytest.raises(ValueError, match="expects 3 containers"):
        pool.restore_replica_state(
            containers=[], per_walker_extras=snap["per_walker"],
            group_state=snap["group_state"],
        )
    with pytest.raises(ValueError, match="expects 3 per_walker_extras"):
        pool.restore_replica_state(
            containers=containers, per_walker_extras=[],
            group_state=snap["group_state"],
        )
    with pytest.raises(ValueError, match="expects 2 group_state entries"):
        pool.restore_replica_state(
            containers=containers, per_walker_extras=snap["per_walker"],
            group_state=[],
        )


def test_serial_wl_pool_restore_rejects_mismatched_group_state_kind():
    """A W=1 slot must get None group_state; a W>1 slot must get a dict."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool = make_serial_wl_pool_mixed()
    snap = pool.snapshot_for_checkpoint()
    containers = pool.data_containers()

    # Swap the None and dict entries so each slot gets the wrong kind.
    flipped = [snap["group_state"][1], None]
    with pytest.raises(ValueError, match="multi-walker slot|bare-replica slot"):
        pool.restore_replica_state(
            containers=containers,
            per_walker_extras=snap["per_walker"],
            group_state=flipped,
        )


def test_process_wl_pool_snapshot_returns_structured_dict(tmp_path):
    """ProcessWangLandauPool snapshot under W=2 returns per_walker
    (length M) and group_state (length N, dicts for W>1 slots)."""
    from tests._wl_fixtures import make_process_wl_pool_w2  # 2 windows x W=2

    pool = make_process_wl_pool_w2(tmp_path)
    try:
        snap = pool.snapshot_for_checkpoint()
        assert set(snap.keys()) == {"per_walker", "group_state"}
        assert len(snap["per_walker"]) == 4
        assert len(snap["group_state"]) == 2
        for gs in snap["group_state"]:
            assert set(gs.keys()) == {"rng_state", "phase"}
    finally:
        pool.shutdown()


def test_process_wl_pool_restore_round_trips_w2(tmp_path):
    """Snapshot a W=2 process pool, restore into a fresh one; group-level
    state (rng_state, phase) matches the snapshot."""
    import json

    from tests._wl_fixtures import make_process_wl_pool_w2

    pool_a = make_process_wl_pool_w2(tmp_path / "a")
    try:
        snap = pool_a.snapshot_for_checkpoint()
        containers = pool_a.data_containers()
    finally:
        pool_a.shutdown()

    pool_b = make_process_wl_pool_w2(tmp_path / "b")
    try:
        pool_b.restore_replica_state(
            containers=containers,
            per_walker_extras=snap["per_walker"],
            group_state=snap["group_state"],
        )
        for slot, gs in zip(pool_b._slots, snap["group_state"], strict=True):
            if gs is not None:
                assert slot.rng.bit_generator.state == json.loads(gs["rng_state"])
                assert slot.phase == gs["phase"]
    finally:
        pool_b.shutdown()


def test_process_wl_pool_restore_rejects_mismatched_inputs(tmp_path):
    """Wrong-length containers / extras / group_state all raise."""
    from tests._wl_fixtures import make_process_wl_pool_w2

    pool = make_process_wl_pool_w2(tmp_path)
    try:
        snap = pool.snapshot_for_checkpoint()
        containers = pool.data_containers()
        with pytest.raises(ValueError, match="expects 4 containers"):
            pool.restore_replica_state(
                containers=[],
                per_walker_extras=snap["per_walker"],
                group_state=snap["group_state"],
            )
        with pytest.raises(ValueError, match="expects 4 extras"):
            pool.restore_replica_state(
                containers=containers,
                per_walker_extras=[],
                group_state=snap["group_state"],
            )
        with pytest.raises(ValueError, match="expects 2 group_state"):
            pool.restore_replica_state(
                containers=containers,
                per_walker_extras=snap["per_walker"],
                group_state=[],
            )
    finally:
        pool.shutdown()


def test_process_pool_finalise_for_reporting_skips_single_walker_slots(tmp_path):
    """Single-walker slots are no-ops; their entropy is untouched."""
    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=1,
    ) as pool:
        for i, slot in enumerate(pool._slots):
            _, conn = slot.workers[0]
            request(conn, ("SET_ENTROPY", {0: float(i), 1: float(i) + 1.0}), i)

        pool.finalise_for_reporting()

        for i, slot in enumerate(pool._slots):
            _, conn = slot.workers[0]
            got = request(conn, ("GET_ENTROPY",), i)
            assert got == {0: float(i), 1: float(i) + 1.0}


def _distinct_in_window_pair_for_pool(ce_path):
    """(a, b, ea, eb) with energies computed under the CE at ce_path."""
    ce = ClusterExpansion.read(str(ce_path))
    return distinct_in_window_pair(ce)


def test_process_wl_pool_per_walker_initial_atoms_reach_workers(tmp_path):
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[[a, b], a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=[2, 1],
    ) as pool:
        _, c0 = pool._slots[0].workers[0]
        _, c1 = pool._slots[0].workers[1]
        e0 = float(request(c0, ("ENERGY",), "w0w0"))
        e1 = float(request(c1, ("ENERGY",), "w0w1"))
    assert e0 == pytest.approx(ea)
    assert e1 == pytest.approx(eb)


def test_process_wl_pool_per_walker_length_mismatch_raises(tmp_path):
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, _b, ea, _eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = ea - 1.0, ea + 1.0
    with pytest.raises(ValueError, match=r"window 0 has 2 walkers"):
        ProcessWangLandauPool(
            ce_path=ce_path,
            initial_atoms=[[a], a],
            windows=[(lo, hi), (lo, hi)],
            energy_spacing=0.1,
            seeds=[0, 1],
            n_walkers_per_window=[2, 1],
        )


def test_process_wl_pool_per_walker_out_of_window_raises(tmp_path):
    """A per-walker structure outside its window is rejected.

    The rejection happens inside the second walker's worker subprocess
    (``WangLandauReplica`` validates the initial energy) and propagates
    to the constructor through the STARTUP handshake as a ``RuntimeError``
    carrying the worker traceback -- a different path from the serial
    per-walker validation.
    """
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    # Narrow window brackets a only; b (walker 1) is outside.
    lo, hi = ea - 0.5, ea + 0.5
    assert not (lo < eb < hi)
    with pytest.raises(RuntimeError, match="outside window"):
        ProcessWangLandauPool(
            ce_path=ce_path,
            initial_atoms=[[a, b], a],
            windows=[(lo, hi), (lo, hi)],
            energy_spacing=0.1,
            seeds=[0, 1],
            n_walkers_per_window=[2, 1],
        )


def test_process_wl_pool_rejects_bare_atoms(tmp_path):
    """A single bare ``Atoms`` is rejected, mirroring the serial path.

    Without the guard the bare ``Atoms`` would be iterated into per-site
    ``Atom`` objects by ``list(initial_atoms)`` and surface a confusing
    ``initial_atoms has N entries`` error.
    """
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with pytest.raises(TypeError, match="sequence of Atoms"):
        ProcessWangLandauPool(
            ce_path=ce_path,
            initial_atoms=atoms,
            windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
            energy_spacing=0.1,
            seeds=[0, 1],
        )


def test_process_wl_pool_broadcast_yields_independent_workers(tmp_path):
    """A single Atoms broadcast to a W=2 window gives independent workers.

    Each walker spawns its own subprocess from its own AtomsSpec, so
    mutating one worker's configuration leaves the other untouched.
    Guards against a future broadcast shortcut that points both walkers
    at one shared worker.
    """
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[a, a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=[2, 1],
    ) as pool:
        _, c0 = pool._slots[0].workers[0]
        _, c1 = pool._slots[0].workers[1]
        occ0_before = np.asarray(request(c0, ("GET_OCC",), "w0w0"))
        occ1_before = np.asarray(request(c1, ("GET_OCC",), "w0w1"))
        np.testing.assert_array_equal(occ0_before, occ1_before)

        # Mutate walker 0 to b's (in-window) configuration; walker 1
        # must be unaffected.
        request(c0, ("SET_OCC", np.asarray(b.numbers, dtype=np.int64)), "w0w0")
        occ0_after = np.asarray(request(c0, ("GET_OCC",), "w0w0"))
        occ1_after = np.asarray(request(c1, ("GET_OCC",), "w0w1"))
    assert not np.array_equal(occ0_after, occ1_after)
    np.testing.assert_array_equal(occ1_after, occ1_before)


# ---------------------------------------------------------------------------
# Matching-exchange surface on SerialWangLandauPool
# ---------------------------------------------------------------------------


def test_serial_wl_pool_walker_counts_and_positions():
    """n_walkers, window_of_position, and n_carriers are correct for a mixed pool."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool = make_serial_wl_pool_mixed()  # windows with [1, 2] walkers
    assert [pool.n_walkers(i) for i in range(len(pool))] == [1, 2]
    assert list(pool.window_of_position()) == [0, 1, 1]
    assert pool.n_carriers() == 3


def test_serial_wl_pool_candidate_pairs_match_primitive():
    """candidate_pairs delegates to matching_for_boundary with the same RNG seed."""
    from mchammer_pt.exchange import matching_for_boundary
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool = make_serial_wl_pool_mixed()
    got = pool.candidate_pairs(0, 1, np.random.default_rng(5))
    expected = matching_for_boundary(
        pool.n_walkers(0), pool.n_walkers(1), np.random.default_rng(5)
    )
    assert got == expected


def test_serial_wl_pool_swap_by_walker_moves_configs():
    """swap_walker_configurations exchanges the configurations of the named walkers."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool = make_serial_wl_pool_mixed()
    e_before_00 = pool.walker_energy(0, 0)
    e_before_10 = pool.walker_energy(1, 0)
    pool.swap_walker_configurations(0, 0, 1, 0)
    assert pool.walker_energy(0, 0) == e_before_10
    assert pool.walker_energy(1, 0) == e_before_00


def test_serial_wl_pool_walker_log_g_delegates():
    """walker_log_g routes to the named walker's own log g."""
    from tests._wl_fixtures import make_serial_wl_pool_mixed

    pool = make_serial_wl_pool_mixed()
    e = pool.walker_energy(1, 0)
    assert pool.walker_log_g(1, 0, e) == pool.log_g(1, e)


# ---------------------------------------------------------------------------
# Matching-exchange surface on ProcessWangLandauPool
# ---------------------------------------------------------------------------


def test_process_wl_pool_caches_walker_energy_after_advance(tmp_path):
    """walker_energy returns a finite per-walker energy from the cache."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms],
        windows=[(e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0],
        n_walkers_per_window=2,
    ) as pool:
        pool.advance_all(20)
        assert pool.n_walkers(0) == 2
        for i in range(len(pool)):
            for w in range(pool.n_walkers(i)):
                assert np.isfinite(pool.walker_energy(i, w))


def test_process_wl_pool_walker_log_g_matches_direct_query(tmp_path):
    """The cached parent-side walker_log_g equals the worker's own log g.

    walker_log_g reads the entropy snapshot cached in the parent (no
    IPC); the comparison value comes from a direct ``LOG_G_AT`` request
    to that walker's worker connection. Equality proves the parent's
    bin-bounds derivation agrees with the worker's live ensemble.
    """
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms],
        windows=[(e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0],
        n_walkers_per_window=2,
    ) as pool:
        pool.advance_all(20)
        for w in range(pool.n_walkers(0)):
            e = pool.walker_energy(0, w)
            cached = pool.walker_log_g(0, w, e)
            _, conn = pool._slots[0].workers[w]
            direct, _ = request(conn, ("LOG_G_AT", float(e), float(e)), w)
            assert cached == float(direct)


def test_process_wl_pool_apply_swaps_moves_configs(tmp_path):
    """apply_swaps physically exchanges configurations between processes.

    Energies are re-read directly from the workers (independent of the
    parent-side cache) to prove the occupations moved across the process
    boundary, not merely that the cache was rearranged.
    """
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[a, b],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        seeds=[0, 1],
    ) as pool:
        _, c_i = pool._slots[0].workers[0]
        _, c_j = pool._slots[1].workers[0]
        assert float(request(c_i, ("ENERGY",), "i0")) == pytest.approx(ea)
        assert float(request(c_j, ("ENERGY",), "j0")) == pytest.approx(eb)

        pool.apply_swaps([(0, 0, 1, 0)])

        # Re-read from the workers, not the (unrefreshed) parent cache.
        assert float(request(c_i, ("ENERGY",), "i0")) == pytest.approx(eb)
        assert float(request(c_j, ("ENERGY",), "j0")) == pytest.approx(ea)


def test_process_wl_pool_swap_walker_configurations_delegates(tmp_path):
    """swap_walker_configurations moves a single walker pair via apply_swaps."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[a, b],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        seeds=[0, 1],
    ) as pool:
        _, c_i = pool._slots[0].workers[0]
        _, c_j = pool._slots[1].workers[0]
        pool.swap_walker_configurations(0, 0, 1, 0)
        assert float(request(c_i, ("ENERGY",), "i0")) == pytest.approx(eb)
        assert float(request(c_j, ("ENERGY",), "j0")) == pytest.approx(ea)


def test_process_wl_pool_walker_counts_and_positions(tmp_path):
    """n_walkers, window_of_position, n_carriers for a mixed [2, 1] pool."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[a, a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=[2, 1],
    ) as pool:
        assert [pool.n_walkers(i) for i in range(len(pool))] == [2, 1]
        assert list(pool.window_of_position()) == [0, 0, 1]
        assert pool.n_carriers() == 3


def test_process_wl_pool_candidate_pairs_match_primitive(tmp_path):
    """candidate_pairs delegates to matching_for_boundary with the same seed."""
    from mchammer_pt.exchange import matching_for_boundary
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, _atoms, _e0 = _wl_pool_factory_kwargs(tmp_path)
    a, b, ea, eb = _distinct_in_window_pair_for_pool(ce_path)
    lo, hi = min(ea, eb) - 1.0, max(ea, eb) + 1.0
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[a, a],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        seeds=[0, 1],
        n_walkers_per_window=[2, 1],
    ) as pool:
        got = pool.candidate_pairs(0, 1, np.random.default_rng(5))
        expected = matching_for_boundary(
            pool.n_walkers(0), pool.n_walkers(1), np.random.default_rng(5)
        )
    assert got == expected


def test_process_pool_exchange_issues_no_energy_or_log_g_ipc(
    tmp_path, monkeypatch
):
    """Exchange acceptance on the process pool is evaluated parent-side.

    The REWL efficiency claim: in the process pool, every term of the
    exchange acceptance ratio is read from the parent-side per-walker
    cache (``walker_energy`` / ``walker_log_g``), so the exchange phase
    issues no per-exchange round-trip to the workers. Concretely, across
    a full advance+exchange run the pool must send **no** ``"ENERGY"``
    opcode (the on-demand ``current_energy(i)`` path) and **no**
    ``"LOG_G_AT"`` opcode (the on-demand ``log_g(i, ...)`` path).
    Accepted swaps still move configurations, but only via batched
    ``"GET_OCC"`` / ``"SET_OCC"``.

    All worker traffic funnels through ``request`` / ``broadcast_gather``
    / ``fanout_gather`` in ``mchammer_pt.parallel.processes``; spying the
    opcode (``msg[0]``) at those three choke points captures every
    command sent. A regression that re-introduced a per-exchange energy
    or ln-g query would show up here as an ``"ENERGY"`` or ``"LOG_G_AT"``
    opcode.
    """
    import mchammer_pt.parallel.processes as proc
    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import (
        make_process_wl_pool_w2,
        make_wl_atoms,
        make_wl_ce,
    )

    seen_opcodes: list[str] = []
    real_request = proc.request
    real_broadcast = proc.broadcast_gather
    real_fanout = proc.fanout_gather

    def spy_request(conn, msg, label):
        seen_opcodes.append(msg[0])
        return real_request(conn, msg, label)

    def spy_broadcast(targets, msg):
        seen_opcodes.append(msg[0])
        return real_broadcast(targets, msg)

    def spy_fanout(targets):
        for _conn, _label, msg in targets:
            seen_opcodes.append(msg[0])
        return real_fanout(targets)

    # 2 windows x W=2 (M=4 walkers), in-process workers. Both windows
    # share the same wide energy range, so proposed swaps land in-window
    # and are accepted -- exercising the GET_OCC/SET_OCC swap path.
    pool = make_process_wl_pool_w2(tmp_path / "pool")
    try:
        pt = WangLandauParallelTempering(
            cluster_expansion=make_wl_ce(),
            atoms=[make_wl_atoms(), make_wl_atoms()],
            windows=pool.windows,
            energy_spacing=pool.energy_spacing,
            block_size=20,
            random_seed=0,
            pool=pool,
        )
        monkeypatch.setattr(proc, "request", spy_request)
        monkeypatch.setattr(proc, "broadcast_gather", spy_broadcast)
        monkeypatch.setattr(proc, "fanout_gather", spy_fanout)
        pt.run(n_cycles=6)
    finally:
        if pool.is_open:
            pool.shutdown()

    opcodes = set(seen_opcodes)
    # The exchange phase must not query worker energy or ln g.
    assert "ENERGY" not in opcodes, (
        f"exchange acceptance sent an ENERGY opcode -- it should read the "
        f"parent-side cache, not round-trip to workers. opcodes: {opcodes}"
    )
    assert "LOG_G_AT" not in opcodes, (
        f"exchange acceptance sent a LOG_G_AT opcode -- it should read the "
        f"parent-side cache, not round-trip to workers. opcodes: {opcodes}"
    )
    # Every opcode sent must be one of the advance/halve/merge/phase/swap
    # opcodes; nothing else (in particular no per-exchange energy/ln-g).
    allowed = {
        "ADVANCE",
        "FORCE_HALVE",
        "SET_ENTROPY",
        "SET_PHASE",
        "GET_OCC",
        "SET_OCC",
        "CONVERGED",
        "FINALISE_MERGE",
    }
    assert opcodes <= allowed, (
        f"unexpected opcode(s) {opcodes - allowed} sent during "
        f"advance+exchange; expected a subset of {allowed}"
    )
    # Swaps were accepted, so the batched move path ran. This guards
    # against the assertion passing vacuously (no swaps -> no IPC at all).
    assert "GET_OCC" in opcodes and "SET_OCC" in opcodes, (
        f"no accepted swaps observed (opcodes: {opcodes}); the test setup "
        f"no longer exercises the exchange move path -- adjust the window "
        f"overlap or cycle count"
    )


def test_walker_post_block_state_schema_unchanged() -> None:
    """The worker ADVANCE reply payload schema is IPC-neutral.

    The 1/t-gate feature tracks decoupled-switch stall state parent-side;
    nothing new crosses the worker wire. The reply each worker sends after
    an ADVANCE is the ``WalkerPostBlockState`` dataclass, so pinning its
    exact field list guards against a regression that silently adds a
    reply field (and therefore a new term to the IPC payload).
    """
    from dataclasses import fields

    from mchammer_pt.wl_coordinator import WalkerPostBlockState

    assert [f.name for f in fields(WalkerPostBlockState)] == [
        "halving_criterion_met",
        "fill_factor",
        "entropy",
        "step",
        "window_entry_step",
        "histogram",
        "reached_energy_window",
        "current_energy",
    ]


# ---------------------------------------------------------------------------
# 1/t-gate policy threading and stall tracking on SerialWangLandauPool
# ---------------------------------------------------------------------------


def _two_wl_replicas_1overt() -> list:
    """Two bare WangLandauReplica over a shared window, schedule=1_over_t."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_replica import WangLandauReplica
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    lo, hi = e0 - 50.0, e0 + 50.0
    return [
        WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=0.1,
            energy_limit_left=lo,
            energy_limit_right=hi,
            random_seed=s,
            ensemble_kwargs={"schedule": "1_over_t"},
        )
        for s in (0, 1)
    ]


class TestSerialPoolPolicyThreading:
    def test_pool_stores_and_views_new_policy(self) -> None:
        from mchammer_pt.parallel.serial import SerialWangLandauPool

        pool = SerialWangLandauPool(
            _two_wl_replicas_1overt(),
            energy_spacing=0.1,
            one_over_t_gate="flatness",
            bp_stall_multiple=3.0,
        )
        view = pool._view_of(0, pool._replicas[0])
        assert view.one_over_t_gate == "flatness"
        assert view.bp_stall_multiple == 3.0
        assert view.last_halve_step is None
        assert view.first_halve_duration is None

    def test_pool_rejects_bad_policy(self) -> None:
        from mchammer_pt.parallel.serial import SerialWangLandauPool

        with pytest.raises(ValueError, match="one_over_t_gate"):
            SerialWangLandauPool(
                _two_wl_replicas_1overt(),
                energy_spacing=0.1,
                one_over_t_gate="bogus",
            )

    def test_stall_state_updates_on_halve(self) -> None:
        from mchammer_pt.parallel.serial import SerialWangLandauPool

        pool = SerialWangLandauPool(
            _two_wl_replicas_1overt(), energy_spacing=0.1
        )
        # The caller guards on plan.halve; _update_stall_state records
        # unconditionally (symmetric with the process backend).
        pool._update_stall_state(0, step=900, window_entry_steps=[100])
        assert pool._last_halve_step[0] == 900
        assert pool._first_halve_duration[0] == 800
        # A second halve keeps T1 and advances last_halve_step.
        pool._update_stall_state(0, step=1500, window_entry_steps=[100])
        assert pool._last_halve_step[0] == 1500
        assert pool._first_halve_duration[0] == 800


class TestProcessWindowStallFields:
    def test_window_has_policy_and_stall_fields(self) -> None:
        import numpy as np

        from mchammer_pt.parallel.processes import ProcessWangLandauWindow

        win = ProcessWangLandauWindow(
            workers=[],
            rng=np.random.default_rng(0),
            schedule="1_over_t",
            one_over_t_gate="flatness",
            bp_stall_multiple=3.0,
        )
        assert win._one_over_t_gate == "flatness"
        assert win._bp_stall_multiple == 3.0
        assert win.last_halve_step is None
        assert win.first_halve_duration is None

    def test_view_of_injects_fields(self) -> None:
        import numpy as np

        from mchammer_pt.parallel.processes import (
            ProcessWangLandauWindow,
            _view_of,
        )

        win = ProcessWangLandauWindow(
            workers=[],
            rng=np.random.default_rng(0),
            schedule="1_over_t",
            one_over_t_gate="flatness",
            bp_stall_multiple=2.0,
        )
        win.last_halve_step = 1000
        win.first_halve_duration = 100
        view = _view_of(win)
        assert view.one_over_t_gate == "flatness"
        assert view.bp_stall_multiple == 2.0
        assert view.last_halve_step == 1000
        assert view.first_halve_duration == 100

    def test_flatness_without_one_over_t_schedule_raises(self) -> None:
        import numpy as np

        from mchammer_pt.parallel.processes import ProcessWangLandauWindow

        # The window validates the gate string; it must also reject the
        # silently-inert flatness + halving-schedule pairing.
        with pytest.raises(ValueError, match="1/t schedule"):
            ProcessWangLandauWindow(
                workers=[],
                rng=np.random.default_rng(0),
                schedule="halving",
                one_over_t_gate="flatness",
            )


def _make_serial_entry_replicas(entries):
    """One wide-window 1/t-schedule replica per requested entry policy."""
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
            ensemble_kwargs={"schedule": "1_over_t"},
            one_over_t_entry=entry,
        )
        for i, entry in enumerate(entries)
    ]


def test_serial_pool_one_over_t_entry_derived_from_replicas():
    """SerialWangLandauPool exposes the entry policy its replicas hold."""
    from mchammer_pt.parallel.serial import SerialWangLandauPool

    pool = SerialWangLandauPool(
        _make_serial_entry_replicas(["f_continuous", "f_continuous"]),
        energy_spacing=0.1,
    )
    assert pool.one_over_t_entry == "f_continuous"


def test_serial_pool_rejects_mixed_one_over_t_entry():
    """Checkpoint metadata records a single entry policy for the run, so
    a pool whose slots disagree could not round-trip faithfully."""
    from mchammer_pt.parallel.serial import SerialWangLandauPool

    with pytest.raises(ValueError, match="one_over_t_entry"):
        SerialWangLandauPool(
            _make_serial_entry_replicas(["window_clock", "f_continuous"]),
            energy_spacing=0.1,
        )


def test_process_wl_pool_stores_one_over_t_entry(tmp_path):
    """ProcessWangLandauPool retains the one_over_t_entry it threads."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
        ensemble_kwargs={"schedule": "1_over_t"},
        one_over_t_entry="f_continuous",
    ) as pool:
        assert pool.one_over_t_entry == "f_continuous"


def test_serial_wl_pool_frozen_measurement_skips_coordinator():
    """frozen_measurement=True advances walkers but leaves g(E) untouched.

    The coordinator (halving, entropy-merge, phase-switch) must not fire
    when the pool is frozen; walkers must still advance (MC step counter
    increases) and no merge events must be recorded.
    """
    from mchammer_pt.parallel.serial import SerialWangLandauPool

    pool = _make_serial_wl_pool(n_replicas=2)
    # Prime each walker's entropy so there is something to check.
    pool.advance_all(20)

    entropy_before = [
        dict(r.ensemble._entropy) for r in pool._replicas
    ]
    steps_before = [r.walker_states[0].step for r in pool._replicas]

    # Build a second frozen pool sharing the same replica objects
    # (the fixture builds fresh replicas; reuse the primed ones).
    frozen_pool = SerialWangLandauPool(
        pool._replicas,
        energy_spacing=0.1,
        frozen_measurement=True,
    )

    frozen_pool.advance_all(20)

    # Coordinator never fired: entropy dicts must be identical.
    for i, r in enumerate(frozen_pool._replicas):
        assert r.ensemble._entropy == entropy_before[i], (
            f"replica {i} entropy changed under frozen_measurement"
        )

    # No merge events were recorded.
    assert frozen_pool._merge_events == []

    # Walkers DID advance: MC step counter must have increased.
    for i, r in enumerate(frozen_pool._replicas):
        assert r.walker_states[0].step > steps_before[i], (
            f"replica {i} did not advance under frozen_measurement"
        )


def test_wl_worker_force_halve_then_set_phase_uses_post_halve_origin():
    """FORCE_HALVE then SET_PHASE (the coordinator's EXECUTE order for
    a coupled halve+switch plan) records the origin from the
    post-halve fill factor under f_continuous."""
    import math

    conn = _make_wl_in_process_conn(
        ensemble_kwargs={"schedule": "1_over_t"},
        one_over_t_entry="f_continuous",
    )
    request(conn, ("ADVANCE", 50), 0)
    e = conn._worker._replica.ensemble
    if e._window_entry_step is None:
        e._window_entry_step = 0
    e._fill_factor = 2.0 ** -4
    step = int(e.step)
    request(conn, ("FORCE_HALVE",), 0)
    request(conn, ("SET_PHASE", "1_over_t"), 0)
    assert e._fill_factor == 2.0 ** -5
    assert e._one_over_t_origin_step == step - math.ceil(2.0 ** 5) + 1
