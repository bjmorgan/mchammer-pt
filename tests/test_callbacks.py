"""Tests for ExchangeCallback protocol and built-in callbacks."""
from __future__ import annotations

import numpy as np

from mchammer_pt.callbacks import (
    ExchangeCallback,
    ExchangeLogger,
    SwapRateTracker,
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


def test_exchange_logger_respects_interval(capsys):
    logger = ExchangeLogger(interval=3)
    for cycle in range(7):
        logger.on_exchange(cycle=cycle, pair_index=0, accepted=True, log_prob_ratio=0.0)
    out = capsys.readouterr().out.strip().splitlines()
    # With interval=3, we expect prints at cycles 0, 3, 6 (3 lines).
    assert len(out) == 3
