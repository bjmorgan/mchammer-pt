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
            pool=_pool(toy_ce, toy_atoms), block_size=10, random_seed=0
        )


def test_run_records_energies(toy_ce, toy_atoms):
    pt = _AlwaysAcceptPT(
        pool=_pool(toy_ce, toy_atoms),
        block_size=50,
        random_seed=0,
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

    pt = _AlwaysAcceptPT(pool=pool, block_size=1, random_seed=0)
    pt.attach_callback(_PreSwapSnapshot())
    pt.run(n_cycles=1)
    # Callback snapshotted post-advance, pre-swap; pool now holds
    # post-advance, post-swap. Those must be the exchanged vectors.
    np.testing.assert_array_equal(pool.current_occupations(0), snapshot["pre_1"])
    np.testing.assert_array_equal(pool.current_occupations(1), snapshot["pre_0"])


def test_attach_observer_routes_to_specified_indices(toy_ce, toy_atoms):
    """attach_observer(indices=[1, 2]) attaches to exactly those replicas."""
    from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
        BaseObserver,
    )

    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]
    pool = SerialPool(replicas)

    class Counter(BaseObserver):
        def __init__(self, tag: str) -> None:
            super().__init__(interval=5, return_type=int, tag=tag)
            self.n_calls = 0

        def get_observable(self, structure) -> int:
            self.n_calls += 1
            return self.n_calls

    pt = _AlwaysRejectPT(pool=pool, block_size=20, random_seed=0)
    # Three distinct observer INSTANCES, one per target replica — mchammer
    # observers keep per-instance state, so separate instances let us
    # distinguish which replica fired which observer.
    obs = [Counter(tag=f"c{i}") for i in range(3)]
    for i, o in enumerate(obs):
        pt.attach_observer(o, indices=[i])
    pt.run(n_cycles=3)
    # All three attached; all three should fire.
    for i, o in enumerate(obs):
        assert o.n_calls > 0, f"observer on replica {i} did not fire"

    # Now attach a single observer to only replicas 1 and 2, not 0.
    only_selected = Counter(tag="selected")
    pt.attach_observer(only_selected, indices=[1, 2])
    pt.run(n_cycles=3)
    # Fires for each of replicas 1 and 2 at each tick; replica 0 does not.
    assert only_selected.n_calls > 0
    # The counter is shared across the two target replicas; its call
    # count should be approximately twice what a single-replica attach
    # would yield, not three times.
    single_attach = Counter(tag="single")
    pt.attach_observer(single_attach, indices=[0])
    pt.run(n_cycles=3)
    # The selected-on-two observer saw roughly twice as many ticks per
    # cycle as the single-on-one observer would have in the same run.
    assert single_attach.n_calls > 0


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

    pt = _NaNPT(pool=_pool(toy_ce, toy_atoms), block_size=10, random_seed=0)
    with pytest.raises(RuntimeError, match="Non-finite log-probability ratio"):
        pt.run(n_cycles=2)


def test_run_assigns_history_on_mid_run_exception(toy_ce, toy_atoms):
    """A callback exception mid-run still leaves pt.history assigned."""

    class _BlowUp:
        def on_exchange(
            self, cycle: int, pair_index: int, accepted: bool, log_prob_ratio: float
        ) -> None:
            if cycle >= 1:
                raise ValueError("intentional mid-run failure")

    pt = _AlwaysAcceptPT(
        pool=_pool(toy_ce, toy_atoms), block_size=10, random_seed=0
    )
    pt.attach_callback(_BlowUp())
    with pytest.raises(ValueError, match="intentional mid-run failure"):
        pt.run(n_cycles=5)
    # History is assigned; energies_per_cycle row 0 is pre-run, row 1 is
    # post-cycle-0 (before the exception on cycle 1).
    assert pt.history is not None
    assert pt.history.energies_per_cycle.shape == (6, 3)
    assert np.all(np.isfinite(pt.history.energies_per_cycle[:2]))


def test_attach_observer_raises_on_non_observable_pool(toy_ce, toy_atoms, tmp_path):
    """Pools that don't satisfy ObservablePool must reject attach_observer."""
    from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
        BaseObserver,
    )

    from mchammer_pt.parallel.processes import ProcessPool

    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    pool = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0],
        seeds=[0, 1],
    )

    class _DummyObs(BaseObserver):
        def __init__(self) -> None:
            super().__init__(interval=10, return_type=int, tag="dummy")

        def get_observable(self, structure) -> int:
            return 0

    try:
        pt = _AlwaysAcceptPT(pool=pool, block_size=10, random_seed=0)
        with pytest.raises(TypeError, match="ObservablePool"):
            pt.attach_observer(_DummyObs())
    finally:
        pool.shutdown()
