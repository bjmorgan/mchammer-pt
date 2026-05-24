"""Tests for the CycleCallback protocol and the ProgressPrinter built-in."""

from __future__ import annotations

import io
import re
import sys
import time

import numpy as np
import pytest

from mchammer_pt.base import BaseParallelTempering
from mchammer_pt.callbacks import (
    CycleCallback,
    ProgressPrinter,
    WangLandauProgressPrinter,
    _format_duration,
)
from mchammer_pt.parallel.serial import SerialPool
from mchammer_pt.replica import Replica
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


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


# ---------------------------------------------------------------------------
# WangLandauProgressPrinter tests
# ---------------------------------------------------------------------------


def _initial_energy() -> float:
    from mchammer.calculators import ClusterExpansionCalculator
    atoms = make_wl_atoms()
    ce = make_wl_ce()
    return float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers
    ))


def _wl_pt(n_cycles_hint: int = 5):
    """Build a minimal 2-window REWL orchestrator on the toy WL CE."""
    from mchammer_pt.wl import WangLandauParallelTempering
    ce = make_wl_ce()
    atoms = make_wl_atoms()
    e0 = _initial_energy()
    return WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
    )


def _count_blocks(output: str) -> int:
    """Count distinct emission blocks by counting '[REWL ' occurrences."""
    return output.count("[REWL ")


def _block_lines(output: str) -> list[list[str]]:
    """Split output into per-block line lists, splitting on '[REWL ' headers."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in output.splitlines():
        if line.startswith("[REWL "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def test_wl_progress_printer_emits_at_interval_and_final():
    """WangLandauProgressPrinter emits at every interval-th cycle plus the final."""
    buf = io.StringIO()
    pt = _wl_pt()
    printer = WangLandauProgressPrinter(pt.pool, interval=3, file=buf)
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=10)

    cycle_nums = [
        int(re.search(r"cycle (\d+)/", line).group(1))
        for line in buf.getvalue().splitlines()
        if line.startswith("[REWL ")
    ]
    assert cycle_nums == [3, 6, 9, 10]


def test_wl_progress_printer_block_structure():
    """Each emitted block has a header, a column header, and one row per window."""
    buf = io.StringIO()
    pt = _wl_pt()
    n_windows = len(pt.pool)
    printer = WangLandauProgressPrinter(
        pt.pool, interval=5, show_swap_rates=False, file=buf
    )
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=5)

    blocks = _block_lines(buf.getvalue())
    assert len(blocks) == 1
    block = blocks[0]
    # Header line + column header + one row per window.
    assert len(block) == 2 + n_windows
    assert "win" in block[1] and "fill_factor" in block[1] and "converged" in block[1]
    for i in range(n_windows):
        assert re.search(rf"\b{i}\b", block[2 + i]) is not None


def test_wl_progress_printer_rows_contain_metrics_after_advance():
    """After at least one cycle, rows show numeric fill factor and halvings."""
    buf = io.StringIO()
    pt = _wl_pt()
    printer = WangLandauProgressPrinter(
        pt.pool, interval=1, show_swap_rates=False, file=buf
    )
    pt.attach_cycle_callback(printer)
    pt.run(n_cycles=1)

    blocks = _block_lines(buf.getvalue())
    assert blocks, "expected at least one emission"
    for row in blocks[-1][2:]:
        # fill_factor is always available via per_window_stats().
        assert re.search(r"\d\.\d{3}e[+-]\d{2}", row) is not None, row


def test_wl_progress_printer_empty_histogram_shows_zero_bins():
    """Before any bins are visited (fresh replica), bins_visited shows 0."""
    buf = io.StringIO()
    pt = _wl_pt()

    # Stub pool whose per_window_stats returns empty histograms.
    class _StubPool:
        def per_window_stats(self):
            return [
                {
                    "fill_factor": 1.0,
                    "halvings": 0,
                    "histogram": {},
                    "bins_visited": 0,
                    "bins_known": 0,
                    "converged": False,
                }
                for _ in range(len(pt.pool))
            ]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    blocks = _block_lines(buf.getvalue())
    assert blocks
    for row in blocks[-1][2:]:
        # bins_visited column should read "0", not "--".
        assert re.search(r"\b0\b", row) is not None, row
        # flat_min is undefined with no histogram data: should show "--".
        assert "--" in row, row


def test_wl_progress_printer_pooled_mode_reports_summed_flat_min():
    """flatness_mode='pooled': flat_min is min(H_summed) / mean(H_summed)."""
    buf = io.StringIO()
    pt = _wl_pt()

    class _StubPool:
        def __len__(self):
            return 1

        def per_window_stats(self):
            # Pooled: combined histogram is {0: 800, 1: 1000}; mean=900,
            # min/mean = 800/900 ~= 0.889.
            return [{
                "fill_factor": 1.0,
                "halvings": 0,
                "histogram": {0: 800, 1: 1000},
                "bins_visited": 2,
                "bins_known": 2,
                "converged": False,
                "flatness_mode": "pooled",
                "per_walker_flat_min": 0.5,  # pooled wins; this is ignored.
            }]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    blocks = _block_lines(buf.getvalue())
    assert blocks
    # 800/900 ~= 0.889 -> "0.889" in the row.
    last_row = blocks[-1][-1]
    assert "0.889" in last_row, last_row


def test_wl_progress_printer_per_walker_mode_reports_walker_min():
    """flatness_mode='per_walker': flat_min is min over walkers of walker_flat_min."""
    buf = io.StringIO()
    pt = _wl_pt()

    class _StubPool:
        def __len__(self):
            return 1

        def per_window_stats(self):
            # Summed histogram looks flat at min/mean = 800/900 ~= 0.889;
            # but per-walker minimum is 0.500 (one walker is far from flat).
            return [{
                "fill_factor": 1.0,
                "halvings": 0,
                "histogram": {0: 800, 1: 1000},
                "bins_visited": 2,
                "bins_known": 2,
                "converged": False,
                "flatness_mode": "per_walker",
                "per_walker_flat_min": 0.500,
            }]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    blocks = _block_lines(buf.getvalue())
    assert blocks
    last_row = blocks[-1][-1]
    assert "0.500" in last_row, last_row


def test_wl_progress_printer_back_compat_no_mode_field():
    """Stats without flatness_mode (legacy/single-replica) keep pooled behaviour."""
    buf = io.StringIO()
    pt = _wl_pt()

    class _StubPool:
        def __len__(self):
            return 1

        def per_window_stats(self):
            # No flatness_mode field — single-walker WangLandauReplica
            # case, or pre-migration data.
            return [{
                "fill_factor": 1.0,
                "halvings": 0,
                "histogram": {0: 800, 1: 1000},
                "bins_visited": 2,
                "bins_known": 2,
                "converged": False,
            }]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    # Should fall through to the existing pooled computation, 800/900 ~= 0.889.
    blocks = _block_lines(buf.getvalue())
    last_row = blocks[-1][-1]
    assert "0.889" in last_row, last_row


def test_wl_progress_printer_shows_phase_column():
    """The per-window table renders ``halv`` for halving-phase windows
    and ``1/t`` for windows whose BP switch has fired. The column is
    in the header line, and each row picks up the right marker.
    """
    buf = io.StringIO()
    pt = _wl_pt()

    class _StubPool:
        def __len__(self) -> int:
            return 2

        def per_window_stats(self):
            base = {
                "fill_factor": 1e-7,
                "halvings": 24,
                "histogram": {0: 100, 1: 100},
                "bins_visited": 2,
                "bins_known": 2,
                "converged": False,
                "flatness_mode": "pooled",
            }
            return [
                {**base, "phase": "halving"},
                {**base, "phase": "1_over_t"},
            ]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    blocks = _block_lines(buf.getvalue())
    assert blocks
    block = blocks[-1]
    header_row = block[1]
    assert "phase" in header_row
    # Two data rows, in window order.
    halving_row, one_over_t_row = block[2], block[3]
    assert " halv " in halving_row, halving_row
    assert " 1/t " in one_over_t_row, one_over_t_row


def test_wl_progress_printer_phase_defaults_to_halv_when_missing():
    """Stats dicts without ``phase`` (legacy callers / hand-built stubs)
    render as ``halv`` rather than blowing up. Keeps the column
    backward-compatible with any external code that passes its own
    stub pool to the printer."""
    buf = io.StringIO()
    pt = _wl_pt()

    class _StubPool:
        def __len__(self) -> int:
            return 1

        def per_window_stats(self):
            return [{
                "fill_factor": 1.0,
                "halvings": 0,
                "histogram": {0: 1},
                "bins_visited": 1,
                "bins_known": 1,
                "converged": False,
            }]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    blocks = _block_lines(buf.getvalue())
    assert blocks
    assert " halv " in blocks[-1][-1]


def test_wl_progress_printer_rejects_non_positive_interval():
    pt = _wl_pt()
    with pytest.raises(ValueError):
        WangLandauProgressPrinter(pt.pool, interval=0)
    with pytest.raises(ValueError):
        WangLandauProgressPrinter(pt.pool, interval=-1)


def test_wl_progress_printer_per_walker_zero_flat_min_displays_as_zero():
    """per_walker_flat_min == 0.0 is a valid value, not a fall-through trigger."""
    buf = io.StringIO()
    pt = _wl_pt()

    class _StubPool:
        def __len__(self):
            return 1

        def per_window_stats(self):
            # per_walker_flat_min is 0.0 (one walker has a fully empty bin).
            # Pooled would be 800/900 ~= 0.889; the reporter must show 0.000
            # under per_walker mode, not fall through.
            return [{
                "fill_factor": 1.0,
                "halvings": 0,
                "histogram": {0: 800, 1: 1000},
                "bins_visited": 2,
                "bins_known": 2,
                "converged": False,
                "flatness_mode": "per_walker",
                "per_walker_flat_min": 0.0,
            }]

    printer = WangLandauProgressPrinter(
        _StubPool(), interval=1, show_swap_rates=False, file=buf
    )
    history = pt.run(n_cycles=1)
    printer.on_cycle_end(0, 1, history)

    blocks = _block_lines(buf.getvalue())
    last_row = blocks[-1][-1]
    assert "0.000" in last_row, last_row
