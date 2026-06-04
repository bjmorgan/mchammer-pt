"""Tests for per-window harvest bookkeeping."""

from __future__ import annotations

import numpy as np

from mchammer_pt.seeding.bookkeeping import WindowHarvest


def _occ(*vals: int) -> np.ndarray:
    return np.array(vals, dtype=int)


def test_records_novel_config():
    h = WindowHarvest(counts=[2, 2])
    assert h.record(0, _occ(1, 2, 3)) is True
    cfgs = h.configs(0)
    assert len(cfgs) == 1
    assert np.array_equal(cfgs[0], _occ(1, 2, 3))


def test_rejects_duplicate_within_window():
    h = WindowHarvest(counts=[2])
    assert h.record(0, _occ(1, 2, 3)) is True
    assert h.record(0, _occ(1, 2, 3)) is False
    assert len(h.configs(0)) == 1


def test_respects_cap():
    h = WindowHarvest(counts=[1])
    assert h.record(0, _occ(1, 1)) is True
    assert h.record(0, _occ(2, 2)) is False  # window already full
    assert len(h.configs(0)) == 1


def test_same_config_distinct_windows_is_allowed():
    h = WindowHarvest(counts=[1, 1])
    assert h.record(0, _occ(5, 5)) is True
    assert h.record(1, _occ(5, 5)) is True


def test_is_full_and_all_full():
    h = WindowHarvest(counts=[1, 2])
    assert not h.all_full()
    h.record(0, _occ(1))
    assert h.is_full(0)
    assert not h.is_full(1)
    h.record(1, _occ(2))
    h.record(1, _occ(3))
    assert h.all_full()


def test_fill_status():
    h = WindowHarvest(counts=[1, 2])
    h.record(0, _occ(1))
    h.record(1, _occ(2))
    n_filled, short = h.fill_status()
    assert n_filled == 1
    assert short == {1: "1/2"}
