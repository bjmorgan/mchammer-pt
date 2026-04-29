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


def test_serial_pool_attach_observer_rejects_out_of_range_replicas(toy_ce, toy_atoms):
    """Out-of-range replica index raises IndexError eagerly."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        with pytest.raises(IndexError, match="out of range"):
            pool.attach_observer(StatefulCounter(interval=10), replicas=[5])
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
        # Each replica recorded at least one observation in its own data container.
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
        _ = pool.temperatures
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
    with pytest.raises(RuntimeError, match="shut down"):
        from tests._observer_fixtures import factory_returning_non_observer
        pool.attach_observer_factory(factory_returning_non_observer)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.get_observers(replica_index=0)


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


def test_observer_data_matches_across_pools(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Same setup + same observer should produce identical data containers
    on SerialPool and ProcessPool.

    Catches subtle divergences in observation timing, RNG threading, or
    tag handling between the two pool implementations.
    """
    import pandas as pd

    from tests._observer_fixtures import StatefulCounter

    temperatures = [300.0, 400.0, 500.0]
    seeds = [11, 22, 33]
    n_steps = 200
    interval = 20

    # Serial side.
    serial_replicas = [
        Replica(toy_ce, toy_atoms, temperature=T, random_seed=s)
        for T, s in zip(temperatures, seeds, strict=True)
    ]
    serial = SerialPool(serial_replicas)
    try:
        serial.attach_observer(
            StatefulCounter(interval=interval),
        )
        serial.advance_all(n_steps)
        serial_dcs = serial.data_containers()
    finally:
        serial.shutdown()

    # Process side.
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    process = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=temperatures,
        seeds=seeds,
    )
    try:
        process.attach_observer(
            StatefulCounter(interval=interval),
        )
        process.advance_all(n_steps)
        process_dcs = process.data_containers()
    finally:
        process.shutdown()

    # Identical column-for-column.
    assert len(serial_dcs) == len(process_dcs)
    for s_dc, p_dc in zip(serial_dcs, process_dcs, strict=True):
        pd.testing.assert_frame_equal(
            s_dc.data, p_dc.data, check_exact=False, rtol=1e-12
        )
def test_serial_pool_attach_observer_factory_constructs_per_replica(toy_ce, toy_atoms):
    """attach_observer_factory calls the factory once per replica with that replica."""
    from tests._observer_fixtures import StatefulCounter

    def make_counter(replica):
        # Use the replica's temperature in the tag so different replicas
        # get different tags — proves the factory really did see each
        # replica's own state.
        return StatefulCounter(interval=10, tag=f"counter_T{int(replica.temperature)}")

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer_factory(make_counter, replicas=[0, 1])
        pool.advance_all(50)
        dcs = pool.data_containers()
        assert "counter_T300" in dcs[0].data.columns
        assert "counter_T400" in dcs[1].data.columns
        # Replica 2 had no factory called for it.
        assert "counter_T500" not in dcs[2].data.columns
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_factory_rejects_non_observer(toy_ce, toy_atoms):
    """Factory returning a non-BaseObserver raises TypeError."""

    def bad_factory(replica):
        return "not an observer"

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        with pytest.raises(TypeError, match="not a BaseObserver"):
            pool.attach_observer_factory(bad_factory)
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_factory_with_unpicklable_construction_inputs(
    toy_ce, toy_atoms
):
    """Factory can use icet objects (ClusterSpace) that wouldn't survive pickle.

    This pins the central use case: ClusterCountObserver requires a
    ClusterSpace, which is not picklable. The factory pattern lets the
    worker reach the cluster space via the replica's ensemble.
    """
    from mchammer.observers import ClusterCountObserver  # type: ignore[import-untyped]

    def make_cco(replica):
        cs = replica.ensemble.calculator.cluster_expansion.get_cluster_space_copy()
        return ClusterCountObserver(
            cluster_space=cs,
            structure=replica.ensemble.structure,
            interval=20,
        )

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer_factory(make_cco)
        pool.advance_all(50)
        dcs = pool.data_containers()
        for dc in dcs:
            # ClusterCountObserver writes columns named
            # "<cluster_label>_<species>" — at least one such column
            # must exist if the observer fired.
            assert any(col not in {"mctrial", "potential"} for col in dc.data.columns)
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_factory_constructs_per_worker(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Factory runs inside each worker and produces a per-replica observer."""
    from tests._observer_fixtures import stateful_counter_factory

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer_factory(stateful_counter_factory, replicas=[0, 1])
        pool.advance_all(50)
        dcs = pool.data_containers()
        assert "counter_T300" in dcs[0].data.columns
        assert "counter_T400" in dcs[1].data.columns
        assert "counter_T500" not in dcs[2].data.columns
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_factory_with_icet_objects(
    toy_ce, toy_atoms, tmp_path: Path
):
    """The escape-hatch use case: attach ClusterCountObserver across processes.

    ClusterSpace doesn't pickle, so neither attach_observer nor
    attach_observer_class can ship a ClusterCountObserver. The factory
    path constructs it inside the worker from worker-local state.
    """
    from tests._observer_fixtures import cluster_count_factory

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer_factory(cluster_count_factory)
        pool.advance_all(50)
        dcs = pool.data_containers()
        for dc in dcs:
            # ClusterCountObserver writes columns named per cluster orbit;
            # any non-baseline column is evidence it fired.
            assert any(col not in {"mctrial", "potential"} for col in dc.data.columns)
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_factory_rejects_function_local_callable(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Function-local factory is rejected up-front (re-import in worker would fail)."""
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        def local_factory(replica):
            from tests._observer_fixtures import StatefulCounter
            return StatefulCounter(interval=10)
        with pytest.raises(ValueError, match="<locals>"):
            pool.attach_observer_factory(local_factory)
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_factory_rejects_non_observer(
    toy_ce, toy_atoms, tmp_path: Path
):
    """A factory that returns a non-BaseObserver shuts the pool down.

    The worker-side error surfaces as RuntimeError; the pool is then shut
    down so subsequent operations raise via _check_open rather than
    silently operating in a partially-attached state.
    """
    from tests._observer_fixtures import factory_returning_non_observer

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="not a BaseObserver"):
            pool.attach_observer_factory(factory_returning_non_observer)
        # Pool is now shut down. Any subsequent op raises.
        with pytest.raises(RuntimeError, match="shut down"):
            pool.advance_all(5)
    finally:
        # shutdown() is idempotent and safe after the failure path.
        pool.shutdown()


def test_process_pool_mid_run_attach(toy_ce, toy_atoms, tmp_path: Path):
    """Observer attached mid-run only fires on subsequent advances.

    The data container's ``mctrial`` column records the step number of
    each observation. After advancing 100 steps and then attaching with
    ``interval=10``, every recorded observation must have
    ``mctrial >= 100``.
    """
    from tests._observer_fixtures import StatefulCounter

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.advance_all(100)  # No observer attached during this run.
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0])
        pool.advance_all(100)  # Observer fires here.
        dcs = pool.data_containers()
        observed = dcs[0].data
        assert "counter" in observed.columns
        counter_rows = observed[observed["counter"].notna()]
        assert int(counter_rows["mctrial"].min()) >= 100
    finally:
        pool.shutdown()


def test_observer_factory_data_matches_across_pools(
    toy_ce, toy_atoms, tmp_path: Path
):
    """attach_observer_factory produces identical data containers on both pools.

    Headline use case: ClusterCountObserver requires a ClusterSpace
    that does not pickle, so this can only ride the factory path. The
    test pins that the factory path is cross-pool consistent for an
    observer users actually want to use.
    """
    import pandas as pd

    from tests._observer_fixtures import cluster_count_factory

    temperatures = [300.0, 400.0, 500.0]
    seeds = [11, 22, 33]
    n_steps = 200

    # Serial side.
    serial_replicas = [
        Replica(toy_ce, toy_atoms, temperature=T, random_seed=s)
        for T, s in zip(temperatures, seeds, strict=True)
    ]
    serial = SerialPool(serial_replicas)
    try:
        serial.attach_observer_factory(cluster_count_factory)
        serial.advance_all(n_steps)
        serial_dcs = serial.data_containers()
    finally:
        serial.shutdown()

    # Process side.
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    process = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=temperatures,
        seeds=seeds,
    )
    try:
        process.attach_observer_factory(cluster_count_factory)
        process.advance_all(n_steps)
        process_dcs = process.data_containers()
    finally:
        process.shutdown()

    # Identical column-for-column.
    assert len(serial_dcs) == len(process_dcs)
    for s_dc, p_dc in zip(serial_dcs, process_dcs, strict=True):
        pd.testing.assert_frame_equal(
            s_dc.data, p_dc.data, check_exact=False, rtol=1e-12
        )


def test_process_pool_attach_observer_failure_poisons_pool(
    toy_ce, toy_atoms, tmp_path: Path
):
    """A worker-side attach failure shuts the pool down so no further ops.

    Pre-fix the docstring promised 'the run should abort' but nothing
    enforced it — a user calling ``advance_all`` after a partial-attach
    failure would silently get data containers with mismatched observer
    columns. Now the pool is shut down by the failure path itself, and
    subsequent calls raise via ``_check_open``.
    """
    from tests._observer_fixtures import factory_returning_non_observer

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        # factory_returning_non_observer raises TypeError inside each
        # worker's ATTACH_OBS_FACTORY handler — surfaces from the parent
        # as RuntimeError. Attach is the failure-path test bed because
        # there is no parent-side eager check that catches a bad return
        # type for the factory variant.
        with pytest.raises(RuntimeError, match="not a BaseObserver"):
            pool.attach_observer_factory(factory_returning_non_observer)
        # Pool should now be shut down. Any subsequent op raises.
        with pytest.raises(RuntimeError, match="shut down"):
            pool.advance_all(5)
    finally:
        # shutdown() is idempotent and safe even after the failure path.
        pool.shutdown()


def test_process_pool_attach_factory_drains_pending_replies_on_partial_failure(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Mixed-success attach: drain consumes succeeded workers' replies before SHUTDOWN.

    With three workers at [300, 400, 500], the factory fails on
    replica 1 (400 K). Workers 0 and 2 produce OK replies; worker 1
    produces ERR. The parent receives worker 0's OK, hits worker 1's
    ERR, and must drain worker 2's queued OK before sending SHUTDOWN
    -- otherwise the SHUTDOWN handshake would race against the unread
    attach reply.
    """
    from tests._observer_fixtures import factory_fails_on_400k

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="not a BaseObserver"):
            pool.attach_observer_factory(factory_fails_on_400k)
        # Pool poisoned; all subsequent ops refuse.
        with pytest.raises(RuntimeError, match="shut down"):
            pool.advance_all(5)
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_class_discards_probe(toy_ce, toy_atoms):
    """Dry-run probe is constructed for validation, then discarded.

    The probe gets construction_id=1; the per-replica fresh instance
    gets id=2. The observed values record construction_id, so a probe
    that was wrongly registered would write 1s to the data container
    instead of 2s.
    """
    from tests._observer_fixtures import ConstructionCounter

    ConstructionCounter.n_constructions = 0
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer_class(ConstructionCounter, 5, replicas=[0])
        assert ConstructionCounter.n_constructions == 2
        pool.advance_all(20)
        dcs = pool.data_containers()
        # Every recorded value is the registered observer's
        # construction_id. The probe (id=1) was discarded; the
        # registered observer is the fresh per-replica instance (id=2).
        observed = dcs[0].data["construction_counter"].dropna()
        assert len(observed) > 0
        assert (observed == 2).all()
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_dedupes_replicas(toy_ce, toy_atoms):
    """Duplicate indices in `replicas` are coalesced; observer fires once per replica.

    Also pins that only replica 0 gets the column, and that the
    registered instance is the fresh per-replica copy (id=2), not the
    probe (id=1).
    """
    from tests._observer_fixtures import ConstructionCounter

    ConstructionCounter.n_constructions = 0
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        # Pass replica 0 three times; should attach once, not three times.
        pool.attach_observer_class(ConstructionCounter, 5, replicas=[0, 0, 0])
        # 1 dry-run + 1 per-unique-replica = 2 constructions.
        assert ConstructionCounter.n_constructions == 2
        pool.advance_all(20)
        dcs = pool.data_containers()
        # Single observer registered on replica 0 — its construction_id
        # is 2 (the per-replica fresh; the dry-run probe was id=1 and
        # was discarded). Replicas 1 and 2 saw nothing.
        observed = dcs[0].data["construction_counter"].dropna()
        assert len(observed) > 0
        assert (observed == 2).all()
        assert "construction_counter" not in dcs[1].data.columns
        assert "construction_counter" not in dcs[2].data.columns
    finally:
        pool.shutdown()


def test_serial_pool_attach_observer_class_empty_replicas_no_construction(
    toy_ce, toy_atoms
):
    """Empty replicas=[] is a no-op: no constructor calls, no validation."""
    from tests._observer_fixtures import ConstructionCounter

    ConstructionCounter.n_constructions = 0
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer_class(ConstructionCounter, 5, replicas=[])
        assert ConstructionCounter.n_constructions == 0
    finally:
        pool.shutdown()


def test_process_pool_attach_observer_class_empty_replicas_no_worker_contact(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Empty replicas=[] short-circuits before any worker is contacted.

    The class-level counter fires only in the parent's dry-run. If the
    empty short-circuit guard works, the parent never reaches dry-run,
    so n_constructions stays 0. Workers are untouched; basic ops still work.
    """
    from tests._observer_fixtures import ConstructionCounter

    ConstructionCounter.n_constructions = 0
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer_class(ConstructionCounter, 5, replicas=[])
        assert ConstructionCounter.n_constructions == 0
        # Workers untouched; basic ops still work.
        pool.advance_all(5)
    finally:
        pool.shutdown()


def test_serial_pool_get_observers_round_trip(toy_ce, toy_atoms):
    """attach -> advance -> get_observers returns the live observer's state."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0])
        pool.advance_all(50)
        observers = pool.get_observers(replica_index=0)
        assert "counter" in observers
        assert observers["counter"].n_calls > 0
    finally:
        pool.shutdown()


def test_serial_pool_get_observers_returns_independent_snapshot(toy_ce, toy_atoms):
    """Mutating the returned observer does not affect the pool's running state."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0])
        pool.advance_all(50)
        snapshot_a = pool.get_observers(replica_index=0)["counter"]
        n_a = snapshot_a.n_calls
        # Mutate on the parent side.
        snapshot_a.n_calls = 999
        # Advance further; the pool's live observer should be unaffected.
        pool.advance_all(50)
        snapshot_b = pool.get_observers(replica_index=0)["counter"]
        assert snapshot_b.n_calls > n_a
        assert snapshot_b.n_calls < 999
    finally:
        pool.shutdown()


def test_serial_pool_get_observers_empty_returns_empty_dict(toy_ce, toy_atoms):
    """A replica with no attached observers returns {}."""
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        observers = pool.get_observers(replica_index=0)
        assert observers == {}
        # Caller's KeyError, not a special pool error.
        with pytest.raises(KeyError):
            _ = observers["nope"]
    finally:
        pool.shutdown()


def test_serial_pool_get_observers_unpicklable_raises_clearly(toy_ce, toy_atoms):
    """An attached observer with non-picklable state surfaces TypeError on retrieval."""
    from tests._observer_fixtures import LambdaAccumulatingObs

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        # Use the factory path so the class itself doesn't need to pickle
        # at attach time; the observer only gains non-picklable state
        # after its first get_observable call.
        pool.attach_observer_factory(
            lambda replica: LambdaAccumulatingObs(interval=5),
            replicas=[0],
        )
        pool.advance_all(20)  # observer fires, attribute mutates.
        with pytest.raises(TypeError, match="not picklable"):
            pool.get_observers(replica_index=0)
    finally:
        pool.shutdown()


def test_serial_pool_get_observers_out_of_range_raises(toy_ce, toy_atoms):
    """Out-of-range replica index raises IndexError eagerly via _resolve_replicas."""
    pool = _make_serial(toy_ce, toy_atoms)
    try:
        with pytest.raises(IndexError, match="out of range"):
            pool.get_observers(replica_index=5)
    finally:
        pool.shutdown()


def test_process_pool_get_observers_unpicklable_in_worker_raises(
    toy_ce, toy_atoms, tmp_path: Path
):
    """If a worker's observer becomes non-picklable mid-run, get_observers
    surfaces a framed RuntimeError carrying the worker's traceback."""
    from tests._observer_fixtures import LambdaAccumulatingObs

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer(LambdaAccumulatingObs(interval=5), replicas=[0])
        pool.advance_all(20)  # Observer fires; stashes a lambda.
        with pytest.raises(RuntimeError, match="GET_OBSERVERS"):
            pool.get_observers(replica_index=0)
    finally:
        pool.shutdown()


def test_process_pool_get_observers_out_of_range_raises(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Out-of-range replica index raises IndexError eagerly."""
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        with pytest.raises(IndexError, match="out of range"):
            pool.get_observers(replica_index=99)
    finally:
        pool.shutdown()


def test_process_pool_get_observers_after_shutdown_raises(
    toy_ce, toy_atoms, tmp_path: Path
):
    """get_observers after shutdown raises RuntimeError via _check_open."""
    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    pool.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        pool.get_observers(replica_index=0)


def test_process_pool_get_observers_round_trip(
    toy_ce, toy_atoms, tmp_path: Path
):
    """attach -> advance -> get_observers returns the worker's observer state."""
    from tests._observer_fixtures import StatefulCounter

    pool = _make_process(toy_ce, toy_atoms, tmp_path)
    try:
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0])
        pool.advance_all(50)
        observers = pool.get_observers(replica_index=0)
        assert "counter" in observers
        assert observers["counter"].n_calls > 0
    finally:
        pool.shutdown()


def test_serial_pool_mid_run_attach(toy_ce, toy_atoms):
    """Observer attached mid-run only fires on subsequent advances.

    Parity with ``test_process_pool_mid_run_attach``. The data
    container's ``mctrial`` column records the step number of each
    observation. After advancing 100 steps and then attaching with
    interval=10, every recorded observation must have mctrial >= 100.
    """
    from tests._observer_fixtures import StatefulCounter

    pool = _make_serial(toy_ce, toy_atoms)
    try:
        pool.advance_all(100)  # No observer attached during this run.
        pool.attach_observer(StatefulCounter(interval=10), replicas=[0])
        pool.advance_all(100)  # Observer fires here.
        dcs = pool.data_containers()
        observed = dcs[0].data
        assert "counter" in observed.columns
        # Filter to rows where the counter actually fired (mchammer
        # writes a row at mctrial=0 from its native energy observer
        # with NaN in the counter column).
        counter_rows = observed.dropna(subset=["counter"])
        assert int(counter_rows["mctrial"].min()) >= 100
    finally:
        pool.shutdown()


def test_get_observers_matches_across_pools(
    toy_ce, toy_atoms, tmp_path: Path
):
    """Same setup + same observer should produce equivalent get_observers
    output on SerialPool and ProcessPool.

    Pins that worker-side instance state (n_calls on StatefulCounter)
    is captured identically by both implementations.
    """
    from tests._observer_fixtures import StatefulCounter

    temperatures = [300.0, 400.0, 500.0]
    seeds = [11, 22, 33]
    n_steps = 200

    # Serial side.
    serial_replicas = [
        Replica(toy_ce, toy_atoms, temperature=T, random_seed=s)
        for T, s in zip(temperatures, seeds, strict=True)
    ]
    serial = SerialPool(serial_replicas)
    try:
        serial.attach_observer(StatefulCounter(interval=20))
        serial.advance_all(n_steps)
        serial_obs = [serial.get_observers(r) for r in range(3)]
    finally:
        serial.shutdown()

    # Process side.
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    process = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=temperatures,
        seeds=seeds,
    )
    try:
        process.attach_observer(StatefulCounter(interval=20))
        process.advance_all(n_steps)
        process_obs = [process.get_observers(r) for r in range(3)]
    finally:
        process.shutdown()

    # Same set of tags, same n_calls per replica.
    for s, p in zip(serial_obs, process_obs, strict=True):
        assert set(s.keys()) == set(p.keys()) == {"counter"}
        assert s["counter"].n_calls == p["counter"].n_calls
