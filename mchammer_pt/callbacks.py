"""PT-level exchange and per-cycle callbacks.

An exchange proposal is a two-replica, cycle-granularity event
carrying a log-probability ratio and an acceptance flag. This module
defines the `ExchangeCallback` protocol for handlers of that event,
plus two built-ins: `SwapRateTracker` (per-pair attempt and accept
counts) and `ExchangePrinter` (stdout trace on a configurable
cadence).

A PT cycle is one round of "advance all replicas + propose the
cycle's exchanges + record history rows". `CycleCallback` is the
protocol for handlers fired once per cycle, after that cycle's
history rows are written. `ProgressPrinter` is the standard built-in
implementation, emitting periodic progress lines on stderr.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .history import ExchangeHistory


class ExchangeCallback(Protocol):
    """Protocol for callables invoked on each exchange proposal.

    Implementations receive one call per proposed exchange. Return
    values are ignored.
    """

    def on_exchange(
        self,
        cycle: int,
        pair_index: int,
        accepted: bool,
        log_prob_ratio: float,
    ) -> None: ...


class SwapRateTracker:
    """Accumulates per-pair attempt and acceptance counts.

    Args:
        n_pairs: number of adjacent replica pairs (``n_replicas - 1``).

    Attributes:
        attempted: per-pair attempt counts, shape ``(n_pairs,)``.
        accepted: per-pair accepted counts, shape ``(n_pairs,)``.
    """

    def __init__(self, n_pairs: int) -> None:
        self.attempted = np.zeros(int(n_pairs), dtype=np.int64)
        self.accepted = np.zeros(int(n_pairs), dtype=np.int64)

    def on_exchange(
        self,
        cycle: int,
        pair_index: int,
        accepted: bool,
        log_prob_ratio: float,
    ) -> None:
        """Record one exchange event."""
        self.attempted[pair_index] += 1
        if accepted:
            self.accepted[pair_index] += 1

    @property
    def acceptance_rates(self) -> np.ndarray:
        """Per-pair acceptance fractions (NaN where no attempts made)."""
        return np.where(
            self.attempted > 0,
            self.accepted / np.maximum(self.attempted, 1),
            np.nan,
        )


class ExchangePrinter:
    """Prints exchange events to stdout on a configurable cadence.

    A minimal built-in handler for quick interactive inspection. For
    log routing, formatting, or level control, implement your own
    `ExchangeCallback` using the standard `logging` module.

    Args:
        interval: print every `interval`-th cycle. ``1`` prints every
            cycle; ``0`` disables printing.
    """

    def __init__(self, interval: int = 1) -> None:
        self._interval = int(interval)

    def on_exchange(
        self,
        cycle: int,
        pair_index: int,
        accepted: bool,
        log_prob_ratio: float,
    ) -> None:
        if self._interval <= 0:
            return
        if cycle % self._interval != 0:
            return
        verdict = "ACCEPT" if accepted else "REJECT"
        print(
            f"[cycle {cycle:6d}] pair {pair_index:3d}  "
            f"log_r = {log_prob_ratio:+.3f}  {verdict}"
        )


class CycleCallback(Protocol):
    """Protocol for callables invoked at the end of each PT cycle.

    Fires after `advance_all`, exchange proposals, and the cycle's
    history rows have all been written. The callback can therefore
    read fresh per-cycle state from `history` (energies, replica
    labels, swap counters) for the cycle that just finished.

    Implementations receive one call per cycle with `cycle` 0-indexed
    in ``[0, n_cycles)``. Return values are ignored.
    """

    def on_cycle_end(
        self,
        cycle: int,
        n_cycles: int,
        history: "ExchangeHistory",
    ) -> None: ...


class ProgressPrinter:
    """Placeholder — replaced in Task 2."""

    def on_cycle_end(self, cycle: int, n_cycles: int, history) -> None:
        return None
