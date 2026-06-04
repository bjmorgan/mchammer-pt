"""Tests for SeedSearchParams validation."""

from __future__ import annotations

import pytest

from mchammer_pt.seeding.params import SeedSearchParams


def test_defaults_are_valid():
    p = SeedSearchParams()
    assert p.window_search_penalty == 2.0
    assert p.walk_sweeps == 50
    assert p.max_walks_per_window == 20
    assert p.n_workers is None


def test_explicit_values_round_trip():
    p = SeedSearchParams(
        window_search_penalty=1.5,
        walk_sweeps=10,
        max_walks_per_window=5,
        n_workers=3,
    )
    assert p.window_search_penalty == 1.5
    assert p.walk_sweeps == 10
    assert p.max_walks_per_window == 5
    assert p.n_workers == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_search_penalty": 0.0},
        {"window_search_penalty": -1.0},
        {"walk_sweeps": 0},
        {"max_walks_per_window": 0},
        {"n_workers": 0},
        {"n_workers": -2},
    ],
)
def test_rejects_non_positive(kwargs):
    with pytest.raises(ValueError):
        SeedSearchParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"walk_sweeps": 2.5},
        {"max_walks_per_window": 2.5},
        {"n_workers": 2.5},
    ],
)
def test_rejects_non_integer_int_knobs(kwargs):
    with pytest.raises(ValueError, match="integer"):
        SeedSearchParams(**kwargs)
