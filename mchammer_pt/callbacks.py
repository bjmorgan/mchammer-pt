"""PT-level exchange callbacks.

The `ExchangeCallback` protocol is a narrow alternative to
`mchammer.BaseObserver` for events that do not fit the
single-ensemble observer model: an exchange proposal is a two-replica
event, fires at cycle granularity (not MC-step granularity), and
carries metadata (log-probability ratio, acceptance) that mchammer
observers do not expect.

Two built-ins are provided: `ExchangeLogger` (prints swap events to
stdout on a cadence) and `SwapRateTracker` (accumulates per-pair
attempt and acceptance counts).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


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
        with np.errstate(invalid="ignore"):
            rates = np.where(
                self.attempted > 0,
                self.accepted / np.maximum(self.attempted, 1),
                np.nan,
            )
        return rates


class ExchangeLogger:
    """Prints exchange events to stdout on a configurable cadence.

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
