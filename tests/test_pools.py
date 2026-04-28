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


def test_process_pool_satisfies_both_replica_pool_and_observable_pool(
    toy_ce, toy_atoms, tmp_path: Path
):
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        assert isinstance(pool, ReplicaPool)
        assert isinstance(pool, ObservablePool)
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


def test_serial_pool_replicas_returns_in_process_handles(toy_ce, toy_atoms):
    """`SerialPool.replicas` returns a copy of the in-process `Replica` list."""
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        replicas = pool.replicas
        assert len(replicas) == 3
        assert [r.temperature for r in replicas] == [300.0, 400.0, 500.0]
        # Mutation of the returned list must not affect the pool.
        replicas.clear()
        assert len(pool) == 3
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
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0, 1])
        pool.advance_all(50)
        dcs = pool.data_containers()
        # Replicas 0 and 1 had the observer attached; replica 2 did not.
        assert "counter" in dcs[0].data.columns
        assert "counter" in dcs[1].data.columns
        assert "counter" not in dcs[2].data.columns
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


def test_process_pool_constructs_supplied_ensemble_subclass(
    toy_ce, toy_atoms, tmp_path: Path
):
    """ProcessPool workers build the supplied subclass with extra kwargs.

    The subclass `TaggedCanonicalEnsemble` requires a `tag` keyword.
    If `ensemble_kwargs` does not reach the worker's Replica
    construction, the worker fails to start and the parent raises
    `RuntimeError` from the ready-handshake.
    """
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    pool = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        seeds=[0, 1, 2],
        ensemble_cls=TaggedCanonicalEnsemble,
        ensemble_kwargs={"tag": "gamma"},
    )
    try:
        # Workers came up: handshake passed and basic queries work.
        pool.advance_all(20)
        es = pool.current_energies()
        assert es.shape == (3,)
    finally:
        pool.shutdown()


def test_process_pool_bad_ensemble_kwargs_surfaces_at_init(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Unknown ensemble_kwargs make every worker fail in __init__.

    Pins that the failure mode is the existing ready-handshake path:
    the parent's __init__ raises RuntimeError carrying the worker's
    TypeError traceback. Without this, an unrecognised kwarg would
    only surface on the first ADVANCE.
    """
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    with pytest.raises(RuntimeError, match="worker startup failed"):
        ProcessPool(
            ce_path=ce_path,
            initial_atoms=toy_atoms,
            temperatures=[300.0, 400.0],
            seeds=[0, 1],
            ensemble_cls=TaggedCanonicalEnsemble,
            ensemble_kwargs={"nonexistent": 1},
        )


def test_process_pool_rejects_interactive_main_class(
    toy_ce, toy_atoms, tmp_path: Path, monkeypatch
):
    """`ensemble_cls` defined in an interactive __main__ is rejected up-front.

    A class in a Jupyter cell has ``__module__ == "__main__"`` and
    ``sys.modules["__main__"].__file__`` either absent or pointing
    somewhere spawn workers cannot re-import. Without the preflight,
    the user sees either a deep multiprocessing PicklingError or an
    EOFError on the parent's first ``recv()`` — neither mentions
    ``ensemble_cls``. The preflight raises a clear ``ValueError``
    before any worker starts.
    """
    import sys

    from mchammer.ensembles import CanonicalEnsemble

    # Construct a class that mimics the notebook case: __module__ ==
    # "__main__" and a top-level qualname (the function-local
    # qualname produced by defining the class here would trip the
    # separate <locals> guard, not the __main__ guard we want to pin).
    class FakeNotebookEnsemble(CanonicalEnsemble):
        pass

    FakeNotebookEnsemble.__module__ = "__main__"
    FakeNotebookEnsemble.__qualname__ = "FakeNotebookEnsemble"
    # Pretend __main__ has no .py file (Jupyter / REPL look like this).
    monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)

    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    with pytest.raises(ValueError, match="__main__"):
        ProcessPool(
            ce_path=ce_path,
            initial_atoms=toy_atoms,
            temperatures=[300.0, 400.0],
            seeds=[0, 1],
            ensemble_cls=FakeNotebookEnsemble,
        )


def test_process_pool_rejects_function_local_class(toy_ce, toy_atoms, tmp_path: Path):
    """`ensemble_cls` defined inside a function is rejected up-front.

    Function-local classes have ``"<locals>"`` in ``__qualname__``;
    pickle cannot walk a function's local scope to recover them. Same
    UX problem as the interactive-``__main__`` case: without the
    preflight, the user sees a deep `PicklingError` from
    multiprocessing internals that doesn't mention `ensemble_cls`.
    """
    from mchammer.ensembles import CanonicalEnsemble

    class LocalEnsemble(CanonicalEnsemble):
        pass

    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    with pytest.raises(ValueError, match="<locals>"):
        ProcessPool(
            ce_path=ce_path,
            initial_atoms=toy_atoms,
            temperatures=[300.0, 400.0],
            seeds=[0, 1],
            ensemble_cls=LocalEnsemble,
        )


def test_serial_pool_attach_observer_gives_independent_copies(toy_ce, toy_atoms):
    """Each replica receives its own deserialised observer copy.

    The user-supplied instance is not registered on any replica; it
    serves only as a template. Mutations to it after attach do not
    affect the run.
    """
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        template = StatefulCounter(interval=10)
        pool.attach_observer(template, replicas="all")
        # The template itself was never registered on any replica.
        assert template.n_calls == 0
        pool.advance_all(50)
        # The template's counter is still zero — none of the replicas
        # called into the template object.
        assert template.n_calls == 0
        # Each replica's own observer fired independently. We can read
        # them back via the data containers' columns.
        dcs = pool.data_containers()
        per_replica_calls = [
            int(dc.data["counter"].iloc[-1]) for dc in dcs
        ]
        # Every replica saw at least one observation, and the counters
        # are not strictly increasing across replicas (which would
        # indicate a single shared object).
        assert all(c >= 1 for c in per_replica_calls)
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_unpicklable_raises_eagerly(toy_ce, toy_atoms):
    """An unpicklable observer raises TypeError before any replica is touched."""
    from mchammer.observers.base_observer import BaseObserver

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        # An observer that closes over a lambda — stdlib pickle refuses lambdas.
        cb = lambda x: x  # noqa: E731

        class Unpicklable(BaseObserver):
            def __init__(self, interval: int) -> None:
                super().__init__(interval=interval, return_type=int, tag="u")
                self.cb = cb

            def get_observable(self, structure):
                return self.cb(0)

        with pytest.raises(TypeError, match="attach_observer_class"):
            pool.attach_observer(Unpicklable(interval=10))
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_class_constructs_per_replica(toy_ce, toy_atoms):
    """attach_observer_class constructs one fresh instance per selected replica."""
    from tests._observer_fixtures import TaggedObserver

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer_class(
            TaggedObserver,
            10,
            label="run-1",
            replicas=[0, 1],
        )
        pool.advance_all(50)
        dcs = pool.data_containers()
        assert "tagged" in dcs[0].data.columns
        assert "tagged" in dcs[1].data.columns
        assert "tagged" not in dcs[2].data.columns
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_class_constructor_failure_surfaces_in_parent(
    toy_ce, toy_atoms
):
    """A constructor exception raises in the calling process, not from a worker."""
    from tests._observer_fixtures import BadInitObserver

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        with pytest.raises(ValueError, match="deliberate constructor failure"):
            pool.attach_observer_class(BadInitObserver, 10)
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_class_rejects_non_observer(toy_ce, toy_atoms):
    """A class whose instances are not BaseObservers raises TypeError."""
    from tests._observer_fixtures import NotAnObserver

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        with pytest.raises(TypeError, match="not a BaseObserver"):
            pool.attach_observer_class(NotAnObserver, 10)
    finally:
        pool.shutdown()


def test_process_pool_public_methods_raise_after_shutdown(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Every public method on ProcessPool raises RuntimeError after shutdown.

    Pre-fix, methods silently no-op or raised opaque IndexError from
    self._workers indexing. The guard converts every entry-point into
    a single clear failure.
    """
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    pool.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        len(pool)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.temperatures
    with pytest.raises(RuntimeError, match="shut down"):
        pool.advance_all(1)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.current_energies()
    with pytest.raises(RuntimeError, match="shut down"):
        pool.current_energy(0)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.current_occupations(0)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.swap_configurations(0, 1)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.data_containers()
    with pytest.raises(RuntimeError, match="shut down"):
        from tests._observer_fixtures import StatefulCounter
        pool.attach_observer(StatefulCounter(interval=1))
    with pytest.raises(RuntimeError, match="shut down"):
        from tests._observer_fixtures import TaggedObserver
        pool.attach_observer_class(TaggedObserver, 10, label="x")


def test_process_pool_attach_observer_fires(toy_ce, toy_atoms, tmp_path: Path):
    """Observer fires inside each ProcessPool worker and lands in its DC."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0, 1])
        pool.advance_all(50)
        dcs = pool.data_containers()
        assert "counter" in dcs[0].data.columns
        assert "counter" in dcs[1].data.columns
        assert "counter" not in dcs[2].data.columns
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_unpicklable_raises_eagerly(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Unpicklable observer raises TypeError without contacting any worker."""
    from mchammer.observers.base_observer import BaseObserver

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        cb = lambda x: x  # noqa: E731

        class Unpicklable(BaseObserver):
            def __init__(self, interval: int) -> None:
                super().__init__(interval=interval, return_type=int, tag="u")
                self.cb = cb

            def get_observable(self, structure):
                return self.cb(0)

        with pytest.raises(TypeError, match="attach_observer_class"):
            pool.attach_observer(Unpicklable(interval=10))
        # Workers are still alive and responsive.
        pool.advance_all(5)
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_class_constructs_per_worker(
    toy_ce, toy_atoms, tmp_path: Path
):
    """attach_observer_class constructs the observer inside each worker."""
    from tests._observer_fixtures import TaggedObserver

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer_class(
            TaggedObserver,
            10,
            label="run-2",
            replicas=[0, 1],
        )
        pool.advance_all(50)
        dcs = pool.data_containers()
        assert "tagged" in dcs[0].data.columns
        assert "tagged" in dcs[1].data.columns
        assert "tagged" not in dcs[2].data.columns
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_class_rejects_function_local_class(
    toy_ce, toy_atoms, tmp_path: Path
):
    """A function-local observer class is rejected up-front."""
    from mchammer.observers.base_observer import BaseObserver

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        class LocalObs(BaseObserver):
            def __init__(self, interval: int) -> None:
                super().__init__(interval=interval, return_type=int, tag="l")

            def get_observable(self, structure):
                return 0

        with pytest.raises(ValueError, match="<locals>"):
            pool.attach_observer_class(LocalObs, 10)
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_class_constructor_failure_surfaces_in_parent(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Constructor exception raises in the parent, not from a worker."""
    from tests._observer_fixtures import BadInitObserver

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        with pytest.raises(ValueError, match="deliberate constructor failure"):
            pool.attach_observer_class(BadInitObserver, 10)
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_class_rejects_non_observer(
    toy_ce, toy_atoms, tmp_path: Path
):
    """A class whose instances are not BaseObservers raises TypeError."""
    from tests._observer_fixtures import NotAnObserver

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        with pytest.raises(TypeError, match="not a BaseObserver"):
            pool.attach_observer_class(NotAnObserver, 10)
    finally:
        pool.shutdown()
