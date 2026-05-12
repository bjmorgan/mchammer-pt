"""PT-level exchange and per-cycle callbacks.

`ExchangeCallback` fires per exchange proposal — a two-replica,
cycle-granularity event with a log-probability ratio and an
acceptance flag. Built-ins: `SwapRateTracker` (per-pair
attempt/accept counters) and `ExchangePrinter` (stdout trace on a
configurable cadence).

`CycleCallback` fires once per PT cycle, after that cycle's history
rows are written. Built-in: `ProgressPrinter` (periodic progress
lines on stderr for long runs).
"""

from __future__ import annotations

import datetime
import sys
import time
from typing import TYPE_CHECKING, Protocol, TextIO

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
        history: ExchangeHistory,
    ) -> None: ...


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration in seconds as ``H:MM:SS``.

    Hours are unbounded: ``30:00:00`` rather than ``1 day, 6:00:00``.
    """
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class ProgressPrinter:
    """Periodic per-cycle progress lines on stderr.

    A built-in `CycleCallback` for monitoring long PT runs in
    non-interactive environments where stderr is captured to a log
    file or pipe. Emits one append-only line every ``interval``
    cycles plus one at the final cycle, with a wall-clock timestamp,
    completion fraction, elapsed and ETA, and (optionally) cumulative
    per-pair swap-acceptance rates.

    Args:
        interval: emit a line every ``interval`` completed cycles.
            Must be ``>= 1``. The final cycle of every ``run()``
            always emits regardless of the modulus.
        show_swap_rates: include cumulative per-pair acceptance rates
            in each line. Defaults to ``True``.
        file: stream to write to. Defaults to ``sys.stderr``.

    Reusing one `ProgressPrinter` across multiple ``pt.run(...)``
    calls is safe: the elapsed/ETA clock resets at the start of each
    run.
    """

    def __init__(
        self,
        interval: int = 100,
        *,
        show_swap_rates: bool = True,
        file: TextIO | None = None,
    ) -> None:
        if int(interval) < 1:
            raise ValueError(f"interval must be >= 1, got {interval!r}")
        self._interval = int(interval)
        self._show_swap_rates = bool(show_swap_rates)
        self._file: TextIO = sys.stderr if file is None else file
        self._start: float | None = None

    def on_cycle_end(
        self,
        cycle: int,
        n_cycles: int,
        history: ExchangeHistory,
    ) -> None:
        if cycle == 0:
            self._start = time.monotonic()
        is_interval_emission = (cycle + 1) % self._interval == 0
        is_final_emission = cycle == n_cycles - 1
        if not (is_interval_emission or is_final_emission):
            return
        assert self._start is not None  # cycle == 0 always sets it first
        elapsed = time.monotonic() - self._start
        cycles_done = cycle + 1
        fraction = cycles_done / n_cycles
        eta = (
            elapsed * (n_cycles / cycles_done - 1)
            if cycles_done < n_cycles
            else 0.0
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"[PT {timestamp}] "
            f"cycle {cycles_done}/{n_cycles}  "
            f"{100.0 * fraction:.1f}%  "
            f"elapsed {_format_duration(elapsed)}  "
            f"ETA {_format_duration(eta)}"
        )
        if self._show_swap_rates:
            rates = np.where(
                history.swap_attempted > 0,
                history.swap_accepted / np.maximum(history.swap_attempted, 1),
                np.nan,
            )
            rate_str = np.array2string(
                rates,
                precision=2,
                suppress_small=True,
                separator=" ",
                max_line_width=10**9,
            )
            line += f"  acc {rate_str}"
        print(line, file=self._file, flush=True)
