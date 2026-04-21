"""Tests for replica pools (serial + processes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mchammer_pt import ObservablePool, ProcessPool, ReplicaPool, SerialPool
from mchammer_pt.replica import Replica


def _make_serial(toy_ce, toy_atoms) -> SerialPool:
    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]
    return SerialPool(replicas)


def _make_process(toy_ce, toy_atoms, tmp_path: Path) -> ProcessPool:
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    return ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        seeds=[0, 1, 2],
    )


def test_serial_pool_satisfies_observable_pool(toy_ce, toy_atoms):
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        assert isinstance(pool, ObservablePool)
    finally:
        pool.shutdown()


def test_process_pool_is_replica_pool_but_not_observable_pool(
    toy_ce, toy_atoms, tmp_path: Path
):
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        assert isinstance(pool, ReplicaPool)
        assert not isinstance(pool, ObservablePool)
    finally:
        pool.shutdown()


def test_serial_pool_basic_methods(toy_ce, toy_atoms):
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        assert len(pool) == 3
        assert list(pool.temperatures) == [300.0, 400.0, 500.0]
        es = pool.current_energies()
        assert es.shape == (3,)
        assert np.allclose(es, pool.current_energy(0), atol=0)
    finally:
        pool.shutdown()


def test_process_pool_basic_methods(toy_ce, toy_atoms, tmp_path: Path):
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        assert len(pool) == 3
        assert list(pool.temperatures) == [300.0, 400.0, 500.0]
        es = pool.current_energies()
        assert es.shape == (3,)
        assert np.isclose(es[0], pool.current_energy(0))
    finally:
        pool.shutdown()


def test_serial_pool_swap_actually_swaps(toy_ce, toy_atoms):
    # Two replicas at different T, advance the high-T one to build
    # some drift, then swap and check the energies are also swapped.
    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1),
        Replica(toy_ce, toy_atoms, temperature=10000.0, random_seed=2),
    ]
    pool = SerialPool(replicas)
    try:
        pool.advance_all(200)
        e_before = pool.current_energies().copy()
        pool.swap_configurations(0, 1)
        e_after = pool.current_energies()
        # swapped -> energies are exchanged
        assert np.isclose(e_after[0], e_before[1])
        assert np.isclose(e_after[1], e_before[0])
    finally:
        pool.shutdown()


def test_process_pool_swap_actually_swaps(toy_ce, toy_atoms, tmp_path: Path):
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    pool = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 10000.0],
        seeds=[1, 2],
    )
    try:
        pool.advance_all(200)
        e_before = pool.current_energies().copy()
        pool.swap_configurations(0, 1)
        e_after = pool.current_energies()
        assert np.isclose(e_after[0], e_before[1])
        assert np.isclose(e_after[1], e_before[0])
    finally:
        pool.shutdown()


@pytest.mark.parametrize("pool_name", ["serial", "process"])
def test_both_pools_advance_actually_progresses_state(
    pool_name, toy_ce, toy_atoms, tmp_path: Path
):
    """Conformance: both pools make current_energies() change after advance.

    Pins that the pool's post-advance state is visible to the caller.
    Without this, the earlier split-ownership bug (orchestrator reads
    stale state while worker state advances) could silently return.
    """
    if pool_name == "serial":
        pool: ReplicaPool = _make_serial(toy_ce, toy_atoms)
    else:
        pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        e_before = pool.current_energies().copy()
        pool.advance_all(200)
        e_after = pool.current_energies()
        assert not np.allclose(e_before, e_after), (
            f"{pool_name} pool: advance_all did not change any replica energies"
        )
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_fires(toy_ce, toy_atoms):
    from mchammer.observers.base_observer import BaseObserver

    class Counter(BaseObserver):
        def __init__(self, interval: int) -> None:
            super().__init__(interval=interval, return_type=int, tag="counter")
            self.n_calls = 0

        def get_observable(self, structure) -> int:
            self.n_calls += 1
            return self.n_calls

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        obs = Counter(interval=10)
        pool.attach_observer(obs, indices=[0, 1])
        pool.advance_all(50)
        assert obs.n_calls > 0
    finally:
        pool.shutdown()


def test_process_pool_data_containers_nonempty_after_advance(
    toy_ce, toy_atoms, tmp_path: Path
):
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.advance_all(20)
        dcs = pool.data_containers()
        assert len(dcs) == 3
        for dc in dcs:
            assert len(dc.data) > 0
    finally:
        pool.shutdown()


def test_shutdown_is_idempotent_on_both_pools(toy_ce, toy_atoms, tmp_path: Path):
    s = _make_serial(toy_ce, toy_atoms)
    s.shutdown()
    s.shutdown()

    p = _make_process(toy_ce, toy_atoms, tmp_path)
    p.shutdown()
    p.shutdown()


def test_process_pool_context_manager_shuts_down_on_exit(
    toy_ce, toy_atoms, tmp_path: Path
):
    """with ProcessPool(...) as pool: ... joins all worker processes on exit."""
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    with ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0],
        seeds=[0, 1],
    ) as pool:
        pool.advance_all(10)
        # Copy of the worker list while still inside the with block.
        workers_during = list(pool._workers)
    assert workers_during, "workers were not live inside the context"
    assert not pool._workers, "workers not cleared on context exit"


def test_process_pool_worker_startup_failure_surfaces_at_init(
    toy_ce, toy_atoms, tmp_path: Path
):
    """A broken CE path makes workers fail; that surfaces at __init__.

    Without the ready-handshake the parent would only notice at the
    first ADVANCE via a BrokenPipeError with the original traceback
    lost. The handshake forwards the worker's traceback so the caller
    sees the actual cause synchronously.
    """
    nonexistent_ce = tmp_path / "does_not_exist.ce"
    with pytest.raises(RuntimeError, match="worker startup failed"):
        ProcessPool(
            ce_path=nonexistent_ce,
            initial_atoms=toy_atoms,
            temperatures=[300.0, 400.0],
            seeds=[0, 1],
        )


def test_process_pool_context_manager_shuts_down_on_exception(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Shutdown still happens if the `with` block raises."""
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    pool = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0],
        seeds=[0, 1],
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        with pool:
            raise RuntimeError("deliberate")
    assert not pool._workers
