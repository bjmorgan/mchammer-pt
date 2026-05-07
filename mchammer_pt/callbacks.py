"""PT-level exchange and cycle callbacks.

Two granularities of event:

- **Exchange proposal:** a two-replica, mid-cycle event carrying a
  log-probability ratio and an acceptance flag. Handled via the
  `ExchangeCallback` protocol; built-ins `SwapRateTracker` (per-pair
  attempt and accept counts) and `ExchangePrinter` (stdout trace).
- **Cycle end:** a per-cycle event signalled after the cycle's
  energies and replica labels have been recorded into history.
  Handled via the `CycleCallback` protocol; built-in
  `ProgressPrinter` writes a percentage / ETA line to stderr at a
  configurable (or auto-tuned) cadence.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Protocol, TextIO

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

    Implementations receive one call per cycle, after that cycle's
    energies and replica labels have been recorded into history.
    Return values are ignored.
    """

    def on_cycle_end(self, cycle: int, n_cycles: int) -> None: ...


def _format_seconds(seconds: float) -> str:
    """Render a non-negative duration in seconds as ``Xh Ym Zs``.

    Returns ``"?"`` for non-finite or negative inputs (used for ETA on
    the very first cycle, when no rate is yet known).
    """
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


class ProgressPrinter:
    """Prints cycle-level progress on a configurable cadence.

    A built-in `CycleCallback` for long PT runs that would otherwise
    produce no output until completion.

    Args:
        interval: print every ``interval``-th cycle.
            ``None`` (default) auto-selects an interval that yields
            roughly 100 lines of output total — i.e.
            ``max(1, n_cycles // 100)``, computed lazily on the first
            call. ``0`` disables printing.
        stream: stream to write to. Defaults to ``sys.stderr`` so
            progress output does not contaminate stdout (which a
            calling script may itself be parsing).

    Lines are flushed on every print so progress is visible in
    log-following tools (``tail -f``, ``less +F``, container log
    drivers). The final cycle is always printed regardless of
    interval, so completion is unambiguous.
    """

    def __init__(
        self,
        interval: int | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self._interval = interval
        self._stream = stream
        self._t0: float | None = None

    def on_cycle_end(self, cycle: int, n_cycles: int) -> None:
        if self._interval is None:
            self._interval = max(1, n_cycles // 100)
        if self._interval <= 0:
            return
        if self._t0 is None:
            self._t0 = time.perf_counter()
        last_cycle = n_cycles - 1
        if cycle != last_cycle and cycle % self._interval != 0:
            return
        completed = cycle + 1
        elapsed = time.perf_counter() - self._t0
        pct = 100.0 * completed / n_cycles
        if completed > 0 and elapsed > 0:
            eta = elapsed * (n_cycles - completed) / completed
        else:
            eta = float("inf")
        width = len(str(n_cycles))
        msg = (
            f"[cycle {completed:>{width}}/{n_cycles}] "
            f"{pct:5.1f}%  elapsed {_format_seconds(elapsed)}  "
            f"ETA {_format_seconds(eta)}"
        )
        print(msg, file=self._stream or sys.stderr, flush=True)
