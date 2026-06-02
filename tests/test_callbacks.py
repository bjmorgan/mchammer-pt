"""Tests for ExchangeCallback protocol and built-in callbacks."""

from __future__ import annotations

import io

import numpy as np

from mchammer_pt.callbacks import (
    ExchangeCallback,
    ExchangePrinter,
    SwapRateTracker,
    WangLandauProgressPrinter,
)
from mchammer_pt.history import ExchangeHistory


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


def test_wl_progress_printer_shows_bins_filled_over_known():
    """The WL progress printer's bins column displays 'filled/known'."""

    class _FakePool:
        windows = [(None, None)]

        def per_window_stats(self):
            return [
                {
                    "fill_factor": 1.0,
                    "halvings": 0,
                    "histogram": {0: 0, 1: 5, 2: 0},
                    "bins_filled": 1,
                    "bins_known": 3,
                    "converged": False,
                    "phase": "halving",
                }
            ]

    out = io.StringIO()
    printer = WangLandauProgressPrinter(
        _FakePool(), interval=1, show_swap_rates=False, file=out,
    )
    history = ExchangeHistory(
        energies_per_cycle=np.zeros((1, 1)),
        replica_labels_per_cycle=np.zeros((1, 1), dtype=int),
        swap_attempted=np.zeros(0, dtype=int),
        swap_accepted=np.zeros(0, dtype=int),
    )
    printer.on_cycle_end(0, 1, history)
    output = out.getvalue()
    assert "bins (fill/known)" in output
    assert "1/3" in output


def _wl_history():
    return ExchangeHistory(
        energies_per_cycle=np.zeros((1, 1)),
        replica_labels_per_cycle=np.zeros((1, 1), dtype=int),
        swap_attempted=np.zeros(0, dtype=int),
        swap_accepted=np.zeros(0, dtype=int),
    )


class _DetailPool:
    windows = [(None, None), (None, None)]

    def per_window_stats(self):
        return [
            {
                "fill_factor": 1.0, "halvings": 0, "histogram": {},
                "bins_filled": 4, "bins_known": 4, "converged": False,
                "phase": "halving",
            },
            {
                "fill_factor": 1.0, "halvings": 0, "histogram": {},
                "bins_filled": 3, "bins_known": 4, "converged": False,
                "phase": "halving",
                "per_walker_breakdown": [
                    {"filled": 4, "known": 4, "flat_min": 0.91},
                    {"filled": 3, "known": 4, "flat_min": 0.0},
                ],
            },
        ]


def test_per_walker_detail_emits_sub_rows_for_selected_window():
    out = io.StringIO()
    printer = WangLandauProgressPrinter(
        _DetailPool(), interval=1, show_swap_rates=False,
        per_walker_detail=[1], file=out,
    )
    printer.on_cycle_end(0, 1, _wl_history())
    output = out.getvalue()
    assert "w0" in output and "w1" in output
    assert "4/4" in output and "3/4" in output
    assert "0.910" in output


def test_per_walker_detail_default_emits_no_sub_rows():
    out = io.StringIO()
    printer = WangLandauProgressPrinter(
        _DetailPool(), interval=1, show_swap_rates=False, file=out,
    )
    printer.on_cycle_end(0, 1, _wl_history())
    assert "w0" not in out.getvalue()


def test_per_walker_detail_out_of_range_raises_at_construction():
    import pytest
    with pytest.raises(IndexError):
        WangLandauProgressPrinter(
            _DetailPool(), interval=1, per_walker_detail=[5],
        )


def test_per_walker_detail_renders_dash_for_none_flat_min():
    """A walker with an empty histogram reports ``flat_min=None``; the
    sub-row must render ``--`` rather than crash on ``f"{None:.3f}"``."""

    class _NoneFlatPool:
        windows = [(None, None)]

        def per_window_stats(self):
            return [
                {
                    "fill_factor": 1.0, "halvings": 0, "histogram": {},
                    "bins_filled": 0, "bins_known": 0, "converged": False,
                    "phase": "halving",
                    "per_walker_breakdown": [
                        {"filled": 0, "known": 0, "flat_min": None},
                    ],
                }
            ]

    out = io.StringIO()
    printer = WangLandauProgressPrinter(
        _NoneFlatPool(), interval=1, show_swap_rates=False,
        per_walker_detail=[0], file=out,
    )
    printer.on_cycle_end(0, 1, _wl_history())
    output = out.getvalue()
    assert "w0" in output
    assert "--" in output
