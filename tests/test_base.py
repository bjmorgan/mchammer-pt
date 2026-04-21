"""Tests for the abstract BaseParallelTempering orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.base import BaseParallelTempering
from mchammer_pt.replica import Replica


class _AlwaysAcceptPT(BaseParallelTempering):
    """Concrete subclass whose exchange always accepts."""

    def _log_prob_ratio(self, i: int, j: int) -> float:
        return 0.0


class _AlwaysRejectPT(BaseParallelTempering):
    """Concrete subclass whose exchange always rejects."""

    def _log_prob_ratio(self, i: int, j: int) -> float:
        return -1e9


def _replicas(toy_ce, toy_atoms) -> list[Replica]:
    return [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]


def test_base_is_abstract(toy_ce, toy_atoms):
    with pytest.raises(TypeError):
        BaseParallelTempering(
            replicas=_replicas(toy_ce, toy_atoms), block_size=10, random_seed=0
        )


def test_run_records_energies(toy_ce, toy_atoms):
    pt = _AlwaysAcceptPT(
        replicas=_replicas(toy_ce, toy_atoms),
        block_size=50,
        random_seed=0,
    )
    pt.run(n_cycles=4)
    assert pt.history is not None
    assert pt.history.energies_per_cycle.shape == (5, 3)
    assert np.all(np.isfinite(pt.history.energies_per_cycle))


def test_always_reject_preserves_replica_label_order(toy_ce, toy_atoms):
    pt = _AlwaysRejectPT(
        replicas=_replicas(toy_ce, toy_atoms),
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
        replicas=_replicas(toy_ce, toy_atoms),
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


def test_callbacks_fire_per_exchange(toy_ce, toy_atoms):
    events: list[tuple[int, int, bool]] = []

    class _Recorder:
        def on_exchange(
            self, cycle: int, pair_index: int, accepted: bool, log_prob_ratio: float
        ) -> None:
            events.append((cycle, pair_index, accepted))

    pt = _AlwaysRejectPT(
        replicas=_replicas(toy_ce, toy_atoms),
        block_size=10,
        random_seed=0,
    )
    pt.attach_callback(_Recorder())
    pt.run(n_cycles=4)
    # 3 replicas, 2 pairs. Alternation: even cycles -> pair 0; odd -> pair 1.
    # Over 4 cycles: 2 + 2 = 4 exchange attempts.
    assert len(events) == 4
    assert all(not accepted for _, _, accepted in events)
