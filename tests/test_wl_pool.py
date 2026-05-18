"""Tests for Wang-Landau pool implementations."""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.parallel._comms import Reply, recv_reply, request
from mchammer_pt.parallel.processes import (
    ProcessWangLandauPool,
    ProcessWangLandauWindow,
    _compute_plan,
    _merge_per_window_stats,
)
from tests._in_process_worker import InProcessWorkerConn
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _make_wl_in_process_conn(ensemble_kwargs: dict | None = None):
    """Build an :class:`InProcessWorkerConn` around a real WL replica.

    Mirrors the seeds, energy limits and ensemble class used by
    :func:`_spawn_wl_worker` so opcode tests can swap one for the other
    without changing the rest of the test body. Used by the tier-1
    in-process opcode round-trip tests; the surviving subprocess pin
    (``test_wl_worker_advance_ack_carries_state``) still uses
    :func:`_spawn_wl_worker` to exercise the real pickle boundary.
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
    )
    return InProcessWorkerConn(replica)


def _spawn_wl_worker(tmp_path, ensemble_kwargs: dict | None = None):
    """Spawn a single _wl_worker and return (process, parent_conn).

    Caller is responsible for sending SHUTDOWN and joining.
    """
    import multiprocessing as mp

    from mchammer.calculators import ClusterExpansionCalculator

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
    atoms_dict = {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": np.asarray(atoms.positions, dtype=np.float64),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
    }
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_wl_worker,
        args=(
            child_conn,
            str(ce_path),
            atoms_dict,
            0.1,           # energy_spacing
            e0 - 100.0,    # energy_limit_left
            e0 + 100.0,    # energy_limit_right
            42,            # seed
            CoordinatedWangLandauEnsemble,
            dict(ensemble_kwargs or {}),
        ),
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


def test_serial_wl_pool_log_g_pair_returns_four_tuple():
    pool = _make_serial_wl_pool()
    e_i = pool.current_energy(0)
    e_j = pool.current_energy(1)
    result = pool.log_g_pair(0, 1, e_i, e_j)
    assert len(result) == 4
    assert all(isinstance(x, float) for x in result)


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


def test_process_wl_pool_log_g_pair_round_trips(tmp_path):
    from mchammer_pt.parallel.processes import ProcessWangLandauPool
    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
    ) as pool:
        e_i = pool.current_energy(0)
        e_j = pool.current_energy(1)
        result = pool.log_g_pair(0, 1, e_i, e_j)
        assert len(result) == 4
        # Unvisited bins in window default to 0.0.
        assert all(x == 0.0 for x in result)
        flags = pool.converged_flags()
        assert flags.dtype == bool
        assert not flags.any()


def test_process_wl_pool_per_window_stats_returns_metrics(tmp_path):
    """per_window_stats() round-trips through WL_STATS opcode and returns
    fill_factor, halvings, histogram, and converged for each window."""
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
        assert set(s.keys()) == {"fill_factor", "halvings", "histogram", "converged"}
        assert isinstance(s["fill_factor"], float)
        assert isinstance(s["halvings"], int)
        assert isinstance(s["histogram"], dict)
        assert isinstance(s["converged"], bool)
        assert s["fill_factor"] > 0.0


def test_merge_per_window_stats_single_walker_returns_payload_unchanged():
    """Single-walker slot: the per-walker stats dict is returned as-is."""
    s = {
        "fill_factor": 0.5,
        "halvings": 1,
        "histogram": {0: 10, 1: 20},
        "converged": False,
    }
    out = _merge_per_window_stats([s], flatness_mode="pooled")
    assert out is s
    assert set(out.keys()) == {"fill_factor", "halvings", "histogram", "converged"}


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
    """swap_configurations exchanges the exchange-walker's occupations across groups."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=2)

    occ0_before = pool.current_occupations(0).copy()
    occ1_before = pool.current_occupations(1).copy()

    pool.swap_configurations(0, 1)

    assert np.array_equal(pool.current_occupations(0), occ1_before)
    assert np.array_equal(pool.current_occupations(1), occ0_before)


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
    """Multi-walker slot: histogram is summed; fill_factor/halvings come
    from walker 0; converged is the AND across walkers; flatness_mode and
    per_walker_flat_min are attached."""
    s0 = {
        "fill_factor": 0.5,
        "halvings": 2,
        "histogram": {0: 10, 1: 20, 2: 30},
        "converged": False,
    }
    s1 = {
        "fill_factor": 0.25,
        "halvings": 3,
        "histogram": {1: 5, 2: 15, 3: 25},
        "converged": True,
    }
    out = _merge_per_window_stats([s0, s1], flatness_mode="pooled")
    assert out["fill_factor"] == 0.5
    assert out["halvings"] == 2
    assert out["histogram"] == {0: 10, 1: 25, 2: 45, 3: 25}
    assert out["converged"] is False
    assert out["flatness_mode"] == "pooled"
    assert "per_walker_flat_min" in out


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


def _make_slot(
    *,
    n_walkers: int = 2,
    is_flat: bool,
    schedule: str,
    flatness_mode: str,
    merge_cadence: str,
    window_entry_step: int | None,
    fill_factor: float,
    entropy: dict[int, float],
    histogram: dict[int, int] | None = None,
    phase: str = "halving",
    reached_energy_window: bool | None = None,
) -> ProcessWangLandauWindow:
    """Build a ``ProcessWangLandauWindow`` with mock workers for unit-testing
    ``_compute_plan``.

    The mock worker tuples carry ``MagicMock`` placeholders for the
    process and connection; ``_compute_plan`` only reads
    ``slot.workers`` via ``len(...)`` and the ``(_, conn)`` unpacking
    pattern -- it never sends or receives on the connection. When
    ``reached_energy_window`` is ``None`` it defaults to
    ``window_entry_step is not None`` (the relationship icet maintains
    under the 1/t schedule); pass an explicit bool to decouple the two
    (as the halving schedule does -- walkers can have entered without
    ``window_entry_step`` being set).
    """
    from unittest.mock import MagicMock

    from mchammer_pt.wl_window_group import WalkerPostBlockState

    workers = [(MagicMock(), MagicMock()) for _ in range(n_walkers)]
    slot = ProcessWangLandauWindow(
        workers=workers,
        rng=np.random.default_rng(0),
        flatness_mode=flatness_mode,  # type: ignore[arg-type]
        merge_cadence=merge_cadence,  # type: ignore[arg-type]
        schedule=schedule,
    )
    slot.phase = phase
    rew = (
        reached_energy_window
        if reached_energy_window is not None
        else window_entry_step is not None
    )
    slot._last = [
        WalkerPostBlockState(
            is_flat=is_flat,
            fill_factor=fill_factor,
            entropy=dict(entropy),
            step=100,
            window_entry_step=window_entry_step,
            histogram=dict(histogram) if histogram is not None else {},
            reached_energy_window=rew,
        )
        for _ in range(n_walkers)
    ]
    return slot


def test_process_wl_window_bp_switch_refuses_unentered_walker():
    """Coordinator plan omits BP switch when any walker has not entered."""
    from unittest.mock import MagicMock

    from mchammer_pt.wl_window_group import WalkerPostBlockState

    # Walker 0 has entered, walker 1 has not. f is tiny so 1/t > f for
    # any plausible t after a collective halve -- the BP switch would
    # fire if t were defined for both walkers.
    workers = [(MagicMock(), MagicMock()) for _ in range(2)]
    slot = ProcessWangLandauWindow(
        workers=workers,
        rng=np.random.default_rng(0),
        flatness_mode="per_walker",
        merge_cadence="at_halve",
        schedule="1_over_t",
    )
    slot._last = [
        WalkerPostBlockState(
            is_flat=True,
            fill_factor=1e-6,
            entropy={},
            step=100,
            window_entry_step=0,
            histogram={},
            reached_energy_window=True,
        ),
        WalkerPostBlockState(
            is_flat=True,
            fill_factor=1e-6,
            entropy={},
            step=100,
            window_entry_step=None,
            histogram={},
            reached_energy_window=False,
        ),
    ]
    plan = _compute_plan(slot)
    # All flat ⇒ plan to halve, but BP switch refused because one
    # walker has no defined ``t``.
    assert plan.halve is True
    assert plan.switch_to_phase is None


def test_compute_plan_halving_all_flat_w2_at_halve_cadence():
    """W=2 + all flat + halving phase ⇒ halve + merged entropy plan."""
    slot = _make_slot(
        is_flat=True,
        schedule="halving",
        flatness_mode="per_walker",
        merge_cadence="at_halve",
        window_entry_step=0,
        fill_factor=1.0,
        entropy={0: 1.0, 1: 2.0},
    )
    plan = _compute_plan(slot)
    assert plan.halve is True
    # Both walkers have identical entropy {0: 1.0, 1: 2.0}; after
    # intersection-mean rebasing each becomes {0: -0.5, 1: 0.5},
    # the bin-wise average is the same, and the post-shift to
    # min=0 yields {0: 0.0, 1: 1.0}.
    assert plan.merged_entropy == {0: 0.0, 1: 1.0}
    assert plan.switch_to_phase is None


def test_compute_plan_pooled_flatness_default_schedule_halves():
    """Default config (halving schedule, pooled flatness, W=2) halves.

    Regression: under the halving schedule icet leaves
    ``window_entry_step`` unset, so the pooled gate must rely on
    ``reached_energy_window`` (which is set under both schedules).
    """
    slot = _make_slot(
        is_flat=True,
        schedule="halving",
        flatness_mode="pooled",
        merge_cadence="at_halve",
        window_entry_step=None,
        reached_energy_window=True,
        fill_factor=1.0,
        entropy={0: 1.0, 1: 2.0},
        histogram={0: 1000, 1: 1000},
    )
    plan = _compute_plan(slot)
    assert plan.halve is True


def test_compute_plan_halving_not_flat_at_halve_cadence_skips_merge():
    """halving phase + not flat + merge_cadence='at_halve' => no halve and no merge."""
    slot = _make_slot(
        is_flat=False,
        schedule="halving",
        flatness_mode="per_walker",
        merge_cadence="at_halve",
        window_entry_step=0,
        fill_factor=1.0,
        entropy={0: 1.0, 1: 2.0},
    )
    plan = _compute_plan(slot)
    assert plan.halve is False
    assert plan.merged_entropy is None
    assert plan.switch_to_phase is None


def test_compute_plan_one_over_t_w2_no_midrun_merge():
    """In 1/t phase, _compute_plan does not produce merged_entropy."""
    slot = _make_slot(
        is_flat=False,
        schedule="1_over_t",
        flatness_mode="per_walker",
        merge_cadence="at_halve",
        window_entry_step=0,
        fill_factor=0.001,
        entropy={0: 1.0, 1: 2.0},
        phase="1_over_t",
    )
    plan = _compute_plan(slot)
    assert plan.halve is False
    assert plan.merged_entropy is None
    assert plan.switch_to_phase is None


def test_compute_plan_halving_collective_halve_triggers_bp_switch():
    """All flat + 1/t schedule + 1/t > post-halve f ⇒ halve + switch_to_phase."""
    # f tiny ⇒ 1/t > f/2 trivially for any reasonable t.
    slot = _make_slot(
        is_flat=True,
        schedule="1_over_t",
        flatness_mode="per_walker",
        merge_cadence="at_halve",
        window_entry_step=0,
        fill_factor=1e-6,
        entropy={0: 1.0, 1: 2.0},
    )
    plan = _compute_plan(slot)
    assert plan.halve is True
    assert plan.switch_to_phase == "1_over_t"


def test_process_wl_pool_multi_walker_snapshot_raises(tmp_path):
    """snapshot_for_checkpoint raises NotImplementedError for multi-walker slots."""
    from tests._in_process_pool import make_in_process_wl_pool

    _, _, e0 = _wl_pool_factory_kwargs(tmp_path)
    pool = make_in_process_wl_pool(
        tmp_path,
        windows=[(e0 - 50.0, e0 + 50.0)],
        seeds=[0],
        n_walkers_per_window=2,
    )
    try:
        with pytest.raises(NotImplementedError, match="checkpointing"):
            pool.snapshot_for_checkpoint()
    finally:
        pool.shutdown()


def test_wl_worker_advance_ack_carries_state(tmp_path):
    """ADVANCE ack returns a WalkerPostBlockState with typed fields."""
    from mchammer_pt.wl_window_group import WalkerPostBlockState

    process, conn = _spawn_wl_worker(tmp_path)
    try:
        payload = request(conn, ("ADVANCE", 50), 0)
        assert isinstance(payload, WalkerPostBlockState)
        assert isinstance(payload.is_flat, bool)
        assert isinstance(payload.fill_factor, float)
        assert isinstance(payload.entropy, dict)
        assert isinstance(payload.step, int)
        assert payload.window_entry_step is None or isinstance(
            payload.window_entry_step, int
        )
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
        # slot._last carries the new values for the merge to read.
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
