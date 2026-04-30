"""Tests for the abstract BaseParallelTempering orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.base import BaseParallelTempering
from mchammer_pt.parallel.serial import SerialPool
from mchammer_pt.replica import Replica


class _AlwaysAcceptPT(BaseParallelTempering):
    """Concrete subclass whose exchange always accepts."""

    def _log_prob_ratio(self, i: int, j: int) -> float:
        return 0.0


class _AlwaysRejectPT(BaseParallelTempering):
    """Concrete subclass whose exchange always rejects."""

    def _log_prob_ratio(self, i: int, j: int) -> float:
        return -1e9


def _pool(toy_ce, toy_atoms) -> SerialPool:
    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]
    return SerialPool(replicas)


def test_base_is_abstract(toy_ce, toy_atoms):
    with pytest.raises(TypeError):
        BaseParallelTempering(
            pool=_pool(toy_ce, toy_atoms), block_size=10, random_seed=0,
            template_atoms=toy_atoms,
        )


def test_run_records_energies(toy_ce, toy_atoms):
    pt = _AlwaysAcceptPT(
        pool=_pool(toy_ce, toy_atoms),
        block_size=50,
        random_seed=0,
        template_atoms=toy_atoms,
    )
    pt.run(n_cycles=4)
    assert pt.history is not None
    assert pt.history.energies_per_cycle.shape == (5, 3)
    assert np.all(np.isfinite(pt.history.energies_per_cycle))


def test_always_reject_preserves_replica_label_order(toy_ce, toy_atoms):
    pt = _AlwaysRejectPT(
        pool=_pool(toy_ce, toy_atoms),
        block_size=50,
        random_seed=0,
        template_atoms=toy_atoms,
    )
    pt.run(n_cycles=4)
    # Labels never permute when every exchange rejects.
    np.testing.assert_array_equal(
        pt.history.replica_labels_per_cycle,
        np.tile(np.arange(3), (5, 1)),
    )
    assert pt.history.swap_accepted.sum() == 0


def test_always_accept_permutes_replica_labels(toy_ce, toy_atoms):
    pt = _AlwaysAcceptPT(
        pool=_pool(toy_ce, toy_atoms),
        block_size=50,
        random_seed=0,
        template_atoms=toy_atoms,
    )
    pt.run(n_cycles=6)
    labels = pt.history.replica_labels_per_cycle
    # With always-accept and alternating pair sets, the label ladder
    # must depart from the identity permutation at some point during
    # the run. (For 3 replicas the permutation happens to close back
    # on itself every 6 cycles, so checking only the final row is
    # not a reliable witness.)
    identity = np.arange(3)
    assert any(not np.array_equal(row, identity) for row in labels[1:])


def test_accepted_exchange_actually_swaps_configurations(toy_ce, toy_atoms):
    """On accept, the pool's occupation vectors are actually exchanged.

    Callbacks fire immediately before `swap_configurations`, so a
    callback captures the post-advance / pre-swap state; comparing that
    to the post-swap state via `pool.current_occupations(i)` reveals
    whether the swap actually happened.
    """
    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1),
        Replica(toy_ce, toy_atoms, temperature=10000.0, random_seed=2),
    ]
    pool = SerialPool(replicas)
    pool.advance_all(200)  # drive the configurations apart

    snapshot: dict[str, np.ndarray] = {}

    class _PreSwapSnapshot:
        def on_exchange(
            self,
            cycle: int,
            pair_index: int,
            accepted: bool,
            log_prob_ratio: float,
        ) -> None:
            snapshot["pre_0"] = pool.current_occupations(0).copy()
            snapshot["pre_1"] = pool.current_occupations(1).copy()

    pt = _AlwaysAcceptPT(
        pool=pool, block_size=1, random_seed=0, template_atoms=toy_atoms,
    )
    pt.attach_callback(_PreSwapSnapshot())
    pt.run(n_cycles=1)
    # Callback snapshotted post-advance, pre-swap; pool now holds
    # post-advance, post-swap. Those must be the exchanged vectors.
    np.testing.assert_array_equal(pool.current_occupations(0), snapshot["pre_1"])
    np.testing.assert_array_equal(pool.current_occupations(1), snapshot["pre_0"])


def test_attach_observer_routes_to_specified_replicas(toy_ce, toy_atoms):
    """attach_observer(replicas=[1, 2]) attaches to exactly those replicas.

    Each replica receives its own deserialised copy of the observer;
    routing is verified via the data containers rather than parent-side
    instance state, since the parent-side instance is never registered
    on any replica.
    """
    from tests._observer_fixtures import StatefulCounter

    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]
    pool = SerialPool(replicas)
    pt = _AlwaysRejectPT(
        pool=pool, block_size=20, random_seed=0, template_atoms=toy_atoms,
    )

    # Attach a counter to replicas 1 and 2 only; replica 0 gets nothing.
    pt.attach_observer(StatefulCounter(interval=5, tag="counter"), replicas=[1, 2])
    pt.run(n_cycles=3)

    dcs = pool.data_containers()
    assert "counter" not in dcs[0].data.columns, "replica 0 should have no observer"
    assert "counter" in dcs[1].data.columns, "replica 1 should have observer output"
    assert "counter" in dcs[2].data.columns, "replica 2 should have observer output"


def test_callbacks_fire_per_exchange(toy_ce, toy_atoms):
    events: list[tuple[int, int, bool]] = []

    class _Recorder:
        def on_exchange(
            self, cycle: int, pair_index: int, accepted: bool, log_prob_ratio: float
        ) -> None:
            events.append((cycle, pair_index, accepted))

    pt = _AlwaysRejectPT(
        pool=_pool(toy_ce, toy_atoms),
        block_size=10,
        random_seed=0,
        template_atoms=toy_atoms,
    )
    pt.attach_callback(_Recorder())
    pt.run(n_cycles=4)
    # 3 replicas, 2 pairs. Alternation: even cycles -> pair 0; odd -> pair 1.
    # Over 4 cycles: 2 + 2 = 4 exchange attempts.
    assert len(events) == 4
    assert all(not accepted for _, _, accepted in events)


def test_non_finite_log_ratio_raises_with_diagnostic_context(toy_ce, toy_atoms):
    """A NaN/inf log ratio raises RuntimeError naming cycle/pair/energies."""

    class _NaNPT(BaseParallelTempering):
        def _log_prob_ratio(self, i: int, j: int) -> float:
            return float("nan")

    pt = _NaNPT(
        pool=_pool(toy_ce, toy_atoms), block_size=10, random_seed=0,
        template_atoms=toy_atoms,
    )
    with pytest.raises(RuntimeError, match="Non-finite log-probability ratio"):
        pt.run(n_cycles=2)


def test_run_assigns_history_when_initial_snapshot_raises(toy_ce, toy_atoms):
    """Exception during the pre-loop snapshot leaves self.history assigned.

    The try/finally wraps the initial current_energies() + label
    snapshot as well as the cycle loop, so a failure in the pool's
    first energy query still produces a (zeros-filled) history rather
    than leaving self._history None.
    """

    class _BoomPool(SerialPool):
        def current_energies(self) -> np.ndarray:
            raise RuntimeError("pre-loop pool failure")

    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]
    pool = _BoomPool(replicas)
    pt = _AlwaysAcceptPT(
        pool=pool, block_size=10, random_seed=0, template_atoms=toy_atoms,
    )
    with pytest.raises(RuntimeError, match="pre-loop pool failure"):
        pt.run(n_cycles=3)
    assert pt.history is not None
    assert pt.history.energies_per_cycle.shape == (4, 3)


def test_run_assigns_history_on_mid_run_exception(toy_ce, toy_atoms):
    """A callback exception mid-run still leaves pt.history assigned."""

    class _BlowUp:
        def on_exchange(
            self, cycle: int, pair_index: int, accepted: bool, log_prob_ratio: float
        ) -> None:
            if cycle >= 1:
                raise ValueError("intentional mid-run failure")

    pt = _AlwaysAcceptPT(
        pool=_pool(toy_ce, toy_atoms), block_size=10, random_seed=0,
        template_atoms=toy_atoms,
    )
    pt.attach_callback(_BlowUp())
    with pytest.raises(ValueError, match="intentional mid-run failure"):
        pt.run(n_cycles=5)
    # History is assigned; energies_per_cycle row 0 is pre-run, row 1 is
    # post-cycle-0 (before the exception on cycle 1).
    assert pt.history is not None
    assert pt.history.energies_per_cycle.shape == (6, 3)
    assert np.all(np.isfinite(pt.history.energies_per_cycle[:2]))



def test_orchestrator_context_manager_shuts_down_pool(toy_ce, toy_atoms, tmp_path):
    """`with pt: ...` calls pool.shutdown() on exit, including on exception."""
    from mchammer_pt.parallel.processes import ProcessPool

    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    pool = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0],
        seeds=[0, 1],
    )
    pt = _AlwaysAcceptPT(
        pool=pool, block_size=10, random_seed=0, template_atoms=toy_atoms,
    )
    with pt:
        pt.run(n_cycles=1)
    assert not pool._workers, "pool workers not cleared on context exit"

    # And again with an exception inside the with block.
    pool2 = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0],
        seeds=[0, 1],
    )
    pt2 = _AlwaysAcceptPT(
        pool=pool2, block_size=10, random_seed=0, template_atoms=toy_atoms,
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        with pt2:
            raise RuntimeError("deliberate")
    assert not pool2._workers


def test_attach_observer_raises_on_non_observable_pool(toy_ce, toy_atoms):
    """Orchestrator rejects attach_observer when its pool isn't ObservablePool.

    Both built-in pools (`SerialPool`, `ProcessPool`) now satisfy
    `ObservablePool`, but the runtime check still matters for any
    future pool that implements only `ReplicaPool`. A small dummy
    pool that lacks the attach methods stands in for that case.
    """
    from collections.abc import Sequence

    from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
        BaseDataContainer,
    )
    from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
        BaseObserver,
    )

    class _NotObservablePool:
        """Minimum surface to satisfy `ReplicaPool` but not `ObservablePool`.

        Methods raise NotImplementedError because the test never calls
        them — it only triggers the orchestrator's runtime
        isinstance check, which fails before any pool method is
        reached.
        """

        def __len__(self) -> int:
            return 2

        @property
        def temperatures(self) -> Sequence[float]:
            return [300.0, 400.0]

        def advance_all(self, n_steps: int) -> None:
            raise NotImplementedError

        def current_energies(self) -> np.ndarray:
            raise NotImplementedError

        def current_energy(self, i: int) -> float:
            raise NotImplementedError

        def current_occupations(self, i: int) -> np.ndarray:
            raise NotImplementedError

        def swap_configurations(self, i: int, j: int) -> None:
            raise NotImplementedError

        def data_containers(self) -> list[BaseDataContainer]:
            raise NotImplementedError

        def shutdown(self) -> None:
            return None

    class _DummyObs(BaseObserver):
        def __init__(self) -> None:
            super().__init__(interval=10, return_type=int, tag="dummy")

        def get_observable(self, structure) -> int:
            return 0

    pool = _NotObservablePool()
    pt = _AlwaysRejectPT(
        pool=pool, block_size=20, random_seed=0, template_atoms=toy_atoms,
    )
    with pytest.raises(TypeError, match="ObservablePool"):
        pt.attach_observer(_DummyObs())


def test_final_configurations_returns_current_occupations(toy_ce, toy_atoms):
    """Each returned Atoms has the current occupations from its replica."""
    pool = _pool(toy_ce, toy_atoms)
    pt = _AlwaysAcceptPT(
        pool=pool, block_size=50, random_seed=0, template_atoms=toy_atoms,
    )
    pt.run(n_cycles=4)
    configs = pt.final_configurations()
    assert len(configs) == 3
    for i, atoms in enumerate(configs):
        np.testing.assert_array_equal(
            atoms.numbers, pool.current_occupations(i)
        )


def test_final_configurations_returns_independent_copies(toy_ce, toy_atoms):
    """Mutating a returned Atoms does not affect the pool."""
    pool = _pool(toy_ce, toy_atoms)
    pt = _AlwaysAcceptPT(
        pool=pool, block_size=50, random_seed=0, template_atoms=toy_atoms,
    )
    pt.run(n_cycles=2)
    configs = pt.final_configurations()
    original_occ = pool.current_occupations(0).copy()
    configs[0].numbers[:] = 0  # mutate
    np.testing.assert_array_equal(pool.current_occupations(0), original_occ)


def test_final_configurations_preserves_cell_and_positions(toy_ce, toy_atoms):
    """Returned Atoms have the template's cell, positions, and pbc."""
    pool = _pool(toy_ce, toy_atoms)
    pt = _AlwaysAcceptPT(
        pool=pool, block_size=10, random_seed=0, template_atoms=toy_atoms,
    )
    pt.run(n_cycles=1)
    configs = pt.final_configurations()
    for atoms in configs:
        np.testing.assert_array_equal(atoms.cell.array, toy_atoms.cell.array)
        np.testing.assert_array_equal(atoms.positions, toy_atoms.positions)
        np.testing.assert_array_equal(atoms.pbc, toy_atoms.pbc)
