"""Tests for the CycleCallback protocol and the ProgressPrinter built-in."""

from __future__ import annotations

import io
import re
import sys
import time

import numpy as np
import pytest

from mchammer_pt.base import BaseParallelTempering
from mchammer_pt.callbacks import CycleCallback, ProgressPrinter, _format_duration
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
            self.calls: list[tuple[int, int, int, bool]] = []

        def on_cycle_end(self, cycle, n_cycles, history) -> None:
            # Pin that the cycle's history rows have been written
            # before the callback fires: row ``cycle + 1`` should be
            # populated (the empty history initialises to zero, so a
            # non-zero row is evidence of a write).
            row_written = bool(
                np.any(history.energies_per_cycle[cycle + 1] != 0.0)
            )
            self.calls.append(
                (
                    cycle,
                    n_cycles,
                    int(history.swap_attempted.sum()),
                    row_written,
                )
            )

    rec = _Recorder()
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(rec)
    pt.run(n_cycles=5)

    cycles = [c[0] for c in rec.calls]
    n_cycles_seen = [c[1] for c in rec.calls]
    swap_sums = [c[2] for c in rec.calls]
    row_written_flags = [c[3] for c in rec.calls]

    assert cycles == [0, 1, 2, 3, 4]
    assert n_cycles_seen == [5, 5, 5, 5, 5]
    assert swap_sums == sorted(swap_sums)  # monotonically non-decreasing
    assert all(row_written_flags), (
        "callback fired before history.energies_per_cycle was written "
        "for that cycle"
    )
    # The recorder satisfies the protocol structurally:
    cb: CycleCallback = rec
    del cb


def _line_cycles(buf: io.StringIO) -> list[int]:
    """Extract the 1-indexed cycle counter from each emitted line."""
    return [
        int(re.search(r"cycle (\d+)/", line).group(1))
        for line in buf.getvalue().splitlines()
    ]


def test_progress_printer_emits_at_interval_and_final_cycle(toy_ce, toy_atoms):
    """ProgressPrinter emits at every `interval`-th cycle plus the final cycle."""
    buf = io.StringIO()
    printer = ProgressPrinter(interval=3, show_swap_rates=False, file=buf)
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=10)

    assert _line_cycles(buf) == [3, 6, 9, 10]


_LINE_RE = re.compile(
    r"^\[PT \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] "
    r"cycle (\d+)/(\d+)  "
    r"(\d+\.\d)%  "
    r"elapsed (\d+:\d{2}:\d{2})  "
    r"ETA (\d+:\d{2}:\d{2})"
    r"(?:  acc \[[^\]]+\])?$"
)


def test_progress_printer_line_format_with_swap_rates(toy_ce, toy_atoms):
    """Each line carries timestamp, cycle, percent, elapsed, ETA, and acc block."""
    buf = io.StringIO()
    printer = ProgressPrinter(interval=2, show_swap_rates=True, file=buf)
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=4)

    lines = buf.getvalue().splitlines()
    assert lines, buf.getvalue()
    for line in lines:
        m = _LINE_RE.match(line)
        assert m is not None, f"line did not match expected format: {line!r}"
        assert "acc [" in line


def test_progress_printer_line_format_without_swap_rates(toy_ce, toy_atoms):
    """When show_swap_rates=False, the acc block is dropped."""
    buf = io.StringIO()
    printer = ProgressPrinter(interval=2, show_swap_rates=False, file=buf)
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=4)

    for line in buf.getvalue().splitlines():
        assert _LINE_RE.match(line) is not None, line
        assert "acc [" not in line


def test_progress_printer_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        ProgressPrinter(interval=0)
    with pytest.raises(ValueError):
        ProgressPrinter(interval=-1)


def test_progress_printer_defaults_to_stderr(toy_ce, toy_atoms, monkeypatch):
    """`ProgressPrinter()` with no `file=` writes to `sys.stderr`."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    printer = ProgressPrinter(interval=1, show_swap_rates=False)
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=2)

    assert "cycle " in buf.getvalue()


def test_format_duration_does_not_roll_over_to_days():
    """`_format_duration` keeps hours unbounded so the output shape is
    stable across multi-day runs."""
    assert _format_duration(0) == "0:00:00"
    assert _format_duration(59) == "0:00:59"
    assert _format_duration(60) == "0:01:00"
    assert _format_duration(3600) == "1:00:00"
    assert _format_duration(86400) == "24:00:00"
    assert _format_duration(90061) == "25:01:01"
    assert _format_duration(7 * 86400) == "168:00:00"


def test_progress_printer_emits_single_line_for_wide_ladders(toy_ce, toy_atoms):
    """The acc block must not wrap onto a second line, even for wide ladders."""
    buf = io.StringIO()
    printer = ProgressPrinter(interval=1, show_swap_rates=True, file=buf)
    pt = _pt(toy_ce, toy_atoms, n_replicas=20)
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=2)

    lines = buf.getvalue().splitlines()
    # Two cycles with interval=1 means two emissions; if the acc block
    # wraps, splitlines() will return more than two entries.
    assert len(lines) == 2, buf.getvalue()
    for line in lines:
        assert _LINE_RE.match(line) is not None, line


def test_progress_printer_resets_clock_across_runs(
    toy_ce, toy_atoms, monkeypatch
):
    """ProgressPrinter's elapsed/ETA clock starts fresh on each pt.run() call.

    Reusing one printer across multiple short runs must not accumulate
    elapsed time across runs.
    """
    # Two runs of n_cycles=2 with interval=1 means each on_cycle_end call
    # both sets/reads the clock as appropriate. Per cycle:
    #   cycle == 0: monotonic() called twice (set start, then for elapsed)
    #   cycle != 0: monotonic() called once (for elapsed)
    # So each run uses 3 monotonic() calls.
    fake_times = iter([
        # Run 1: simulated wall-clock 0s -> 100s.
        0.0, 50.0, 100.0,
        # Run 2: simulated wall-clock 1000s start; elapsed within the run
        # should be tiny (0s and 1s), not 1000s.
        1000.0, 1000.0, 1001.0,
    ])
    monkeypatch.setattr(time, "monotonic", lambda: next(fake_times))

    buf = io.StringIO()
    printer = ProgressPrinter(interval=1, show_swap_rates=False, file=buf)
    pt = _pt(toy_ce, toy_atoms)
    pt.attach_cycle_callback(printer)

    pt.run(n_cycles=2)
    pt.run(n_cycles=2)

    lines = buf.getvalue().splitlines()
    assert len(lines) == 4

    def _elapsed_seconds(line: str) -> int:
        h, m, s = re.search(
            r"elapsed (\d+):(\d{2}):(\d{2})", line
        ).groups()
        return int(h) * 3600 + int(m) * 60 + int(s)

    # Run 1's two emissions: elapsed 50s, then 100s.
    assert _elapsed_seconds(lines[0]) == 50
    assert _elapsed_seconds(lines[1]) == 100
    # Run 2's two emissions: elapsed 0s, then 1s — clock was reset.
    assert _elapsed_seconds(lines[2]) == 0
    assert _elapsed_seconds(lines[3]) == 1
