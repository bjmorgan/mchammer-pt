"""Tests for ExchangeCallback protocol and built-in callbacks."""

from __future__ import annotations

import io
import re

import numpy as np

from mchammer_pt.callbacks import (
    CycleCallback,
    ExchangeCallback,
    ExchangePrinter,
    ProgressPrinter,
    SwapRateTracker,
    _format_seconds,
)


def test_exchange_callback_is_a_protocol_that_accepts_any_matching_class():
    class MyCallback:
        calls: list[tuple[int, int, bool, float]] = []

        def on_exchange(
            self,
            cycle: int,
            pair_index: int,
            accepted: bool,
            log_prob_ratio: float,
        ) -> None:
            self.calls.append((cycle, pair_index, accepted, log_prob_ratio))

    cb: ExchangeCallback = MyCallback()
    cb.on_exchange(cycle=0, pair_index=0, accepted=True, log_prob_ratio=0.1)
    assert MyCallback.calls == [(0, 0, True, 0.1)]


def test_swap_rate_tracker_counts_per_pair():
    tracker = SwapRateTracker(n_pairs=3)
    tracker.on_exchange(cycle=0, pair_index=0, accepted=True, log_prob_ratio=0.0)
    tracker.on_exchange(cycle=0, pair_index=0, accepted=False, log_prob_ratio=-1.0)
    tracker.on_exchange(cycle=0, pair_index=1, accepted=True, log_prob_ratio=0.0)
    np.testing.assert_array_equal(tracker.attempted, [2, 1, 0])
    np.testing.assert_array_equal(tracker.accepted, [1, 1, 0])
    rates = tracker.acceptance_rates
    assert rates[0] == 0.5
    assert rates[1] == 1.0
    assert np.isnan(rates[2])


def test_swap_rate_tracker_all_nan_before_any_exchange():
    tracker = SwapRateTracker(n_pairs=4)
    rates = tracker.acceptance_rates
    assert np.all(np.isnan(rates))
    assert rates.shape == (4,)


def test_exchange_printer_respects_interval(capsys):
    printer = ExchangePrinter(interval=3)
    for cycle in range(7):
        printer.on_exchange(
            cycle=cycle, pair_index=0, accepted=True, log_prob_ratio=0.0
        )
    out = capsys.readouterr().out.strip().splitlines()
    # With interval=3, we expect prints at cycles 0, 3, 6 (3 lines).
    assert len(out) == 3


def test_exchange_printer_interval_one_prints_every_cycle(capsys):
    printer = ExchangePrinter(interval=1)
    for cycle in range(5):
        printer.on_exchange(
            cycle=cycle, pair_index=0, accepted=True, log_prob_ratio=0.0
        )
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 5


def test_exchange_printer_interval_zero_disables_output(capsys):
    printer = ExchangePrinter(interval=0)
    for cycle in range(5):
        printer.on_exchange(
            cycle=cycle, pair_index=0, accepted=True, log_prob_ratio=0.0
        )
    out = capsys.readouterr().out
    assert out == ""


def test_cycle_callback_is_a_protocol_that_accepts_any_matching_class():
    class MyCycleCallback:
        calls: list[tuple[int, int]] = []

        def on_cycle_end(self, cycle: int, n_cycles: int) -> None:
            self.calls.append((cycle, n_cycles))

    cb: CycleCallback = MyCycleCallback()
    cb.on_cycle_end(cycle=0, n_cycles=10)
    assert MyCycleCallback.calls == [(0, 10)]


def test_format_seconds_renders_short_medium_and_long_durations():
    assert _format_seconds(0) == "0s"
    assert _format_seconds(45) == "45s"
    assert _format_seconds(125) == "2m 05s"
    assert _format_seconds(3600 + 70) == "1h 01m"


def test_format_seconds_returns_question_mark_for_non_finite():
    assert _format_seconds(float("inf")) == "?"
    assert _format_seconds(float("nan")) == "?"
    assert _format_seconds(-1.0) == "?"


def test_progress_printer_auto_interval_targets_about_a_hundred_lines():
    stream = io.StringIO()
    printer = ProgressPrinter(stream=stream)
    n_cycles = 1000
    for cycle in range(n_cycles):
        printer.on_cycle_end(cycle=cycle, n_cycles=n_cycles)
    lines = stream.getvalue().strip().splitlines()
    # Auto-interval = max(1, 1000 // 100) = 10. Cycles printed are
    # 0, 10, 20, ..., 990 (100 lines), and the final cycle 999 is
    # always printed even off-interval (101 lines total).
    assert len(lines) == 101


def test_progress_printer_explicit_interval():
    stream = io.StringIO()
    printer = ProgressPrinter(interval=4, stream=stream)
    n_cycles = 20
    for cycle in range(n_cycles):
        printer.on_cycle_end(cycle=cycle, n_cycles=n_cycles)
    lines = stream.getvalue().strip().splitlines()
    # interval=4 prints at 0, 4, 8, 12, 16 (5 lines), and final cycle
    # 19 always prints regardless of interval (6 lines total).
    assert len(lines) == 6


def test_progress_printer_interval_zero_disables_output():
    stream = io.StringIO()
    printer = ProgressPrinter(interval=0, stream=stream)
    for cycle in range(50):
        printer.on_cycle_end(cycle=cycle, n_cycles=50)
    assert stream.getvalue() == ""


def test_progress_printer_format_includes_cycle_count_and_eta():
    stream = io.StringIO()
    printer = ProgressPrinter(interval=1, stream=stream)
    printer.on_cycle_end(cycle=0, n_cycles=10)
    line = stream.getvalue().strip()
    # Format: "[cycle  1/10] 10.0%  elapsed Xs  ETA Y"
    assert re.match(
        r"\[cycle\s+1/10\]\s+10\.0%\s+elapsed\s+\S+\s+ETA\s+\S+", line,
    ), line


def test_progress_printer_always_prints_final_cycle():
    stream = io.StringIO()
    printer = ProgressPrinter(interval=1000, stream=stream)
    n_cycles = 50
    for cycle in range(n_cycles):
        printer.on_cycle_end(cycle=cycle, n_cycles=n_cycles)
    lines = stream.getvalue().strip().splitlines()
    # interval=1000 (>> n_cycles): only cycle 0 hits the modulo and the
    # final cycle 49 always prints.
    assert len(lines) == 2
    assert "1/50" in lines[0]
    assert "50/50" in lines[1]
