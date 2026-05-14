"""Tests for Wang-Landau pool implementations."""

from __future__ import annotations

import numpy as np
import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


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
    from mchammer_pt.parallel.processes import ProcessWangLandauPool
    from tests._observer_fixtures import StatefulCounter

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
    ) as pool:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0, 1])
        snapshot = pool.get_observers(0)
        assert "counter" in snapshot
        assert isinstance(snapshot["counter"], StatefulCounter)


def test_process_wl_pool_swap_configurations_refreshes_worker_state(tmp_path):
    """After a swap, each worker's _potential reflects the new configuration."""
    from mchammer_pt.parallel.processes import ProcessWangLandauPool

    ce_path, atoms, e0 = _wl_pool_factory_kwargs(tmp_path)
    with ProcessWangLandauPool(
        ce_path=ce_path,
        initial_atoms=[atoms, atoms],
        windows=[(e0 - 50.0, e0 + 50.0), (e0 - 50.0, e0 + 50.0)],
        energy_spacing=0.1,
        seeds=[0, 1],
    ) as pool:
        # Drive the two replicas to different occupation vectors.
        occ0 = pool.current_occupations(0).copy()
        occ1 = occ0.copy()
        occ1[[0, -1]] = occ1[[-1, 0]]  # swap first/last to differ

        # Direct SET_OCC on worker 1 so its configuration is distinct.
        _, conn1 = pool._workers[1]
        conn1.send(("SET_OCC", np.asarray(occ1, dtype=np.int64)))
        pool._recv_or_raise(conn1, "SET_OCC", 1)

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


def test_serial_wl_pool_swap_configurations_with_window_groups():
    """swap_configurations exchanges the exchange-walker's occupations across groups."""
    pool = _make_wl_pool_with_groups(n_windows=2, n_walkers=2)

    occ0_before = pool.current_occupations(0).copy()
    occ1_before = pool.current_occupations(1).copy()

    pool.swap_configurations(0, 1)

    assert np.array_equal(pool.current_occupations(0), occ1_before)
    assert np.array_equal(pool.current_occupations(1), occ0_before)
