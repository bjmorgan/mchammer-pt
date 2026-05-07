"""Tests for the CycleCallback protocol and the ProgressPrinter built-in."""

from __future__ import annotations

import io
import re
import time

import pytest

from mchammer_pt.base import BaseParallelTempering
from mchammer_pt.callbacks import CycleCallback, ProgressPrinter
from mchammer_pt.parallel.serial import SerialPool
from mchammer_pt.replica import Replica


class _AlwaysAcceptPT(BaseParallelTempering):
    """Concrete subclass whose exchange always accepts."""

    def _log_prob_ratio(self, i: int, j: int) -> float:
        return 0.0


def _pt(toy_ce, toy_atoms, n_replicas: int = 3) -> _AlwaysAcceptPT:
    replicas = [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(n_replicas)
    ]
    return _AlwaysAcceptPT(
        pool=SerialPool(replicas),
        block_size=10,
        random_seed=0,
        template_atoms=toy_atoms,
    )


def test_cycle_callback_fires_once_per_cycle_in_order(toy_ce, toy_atoms):
    """The orchestrator invokes cycle callbacks once per cycle, in order,
    after history rows for that cycle have been written."""

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int]] = []

        def on_cycle_end(self, cycle, n_cycles, history) -> None:
            self.calls.append(
                (cycle, n_cycles, int(history.swap_attempted.sum()))
            )

    rec = _Recorder()
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(rec)
    pt.run(n_cycles=5)

    cycles = [c[0] for c in rec.calls]
    n_cycles_seen = [c[1] for c in rec.calls]
    swap_sums = [c[2] for c in rec.calls]

    assert cycles == [0, 1, 2, 3, 4]
    assert n_cycles_seen == [5, 5, 5, 5, 5]
    assert swap_sums == sorted(swap_sums)  # monotonically non-decreasing
    # The recorder satisfies the protocol structurally:
    cb: CycleCallback = rec
    del cb
