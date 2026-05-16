"""Unit tests for WindowResult."""

from __future__ import annotations

import pandas as pd
import pytest

from mchammer_pt.wl_result import WindowResult


def _make_mock_container(
    entropy: dict[int, float],
    histogram: dict[int, int],
    energy_spacing: float,
    fill_factor: float = 0.5,
    fill_factor_history: dict | None = None,
    entropy_history: dict | None = None,
) -> object:
    """Build a minimal mock that quacks like WangLandauDataContainer."""
    from unittest.mock import MagicMock

    container = MagicMock()
    container._last_state = {
        "entropy": dict(entropy),
        "histogram": dict(histogram),
        "fill_factor": fill_factor,
        "fill_factor_history": fill_factor_history or {},
        "entropy_history": entropy_history or {},
    }
    container.ensemble_parameters = {"energy_spacing": energy_spacing}
    container.fill_factor = fill_factor
    return container


def test_get_entropy_merges_two_walkers():
    """Entropy is averaged bin-wise across walkers."""
    c0 = _make_mock_container(
        entropy={0: 2.0, 1: 4.0},
        histogram={0: 10, 1: 5},
        energy_spacing=0.5,
    )
    c1 = _make_mock_container(
        entropy={0: 6.0, 1: 8.0},
        histogram={0: 8, 1: 3},
        energy_spacing=0.5,
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=1.0,
        energy_spacing=0.5,
        containers=(c0, c1),
    )
    df = wr.get_entropy()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"energy", "entropy"}
    # bin 0: (2+6)/2 = 4.0, bin 1: (4+8)/2 = 6.0
    # After min-shift: 4-4=0, 6-4=2
    row0 = df.loc[0]
    row1 = df.loc[1]
    assert row0["energy"] == pytest.approx(0.0)  # 0 * 0.5
    assert row1["energy"] == pytest.approx(0.5)  # 1 * 0.5
    assert row0["entropy"] == pytest.approx(0.0)  # min-shifted
    assert row1["entropy"] == pytest.approx(2.0)


def test_get_histogram_sums_two_walkers():
    """Histogram is summed bin-wise across walkers."""
    c0 = _make_mock_container(
        entropy={0: 1.0},
        histogram={0: 10, 1: 5},
        energy_spacing=0.5,
    )
    c1 = _make_mock_container(
        entropy={0: 1.0},
        histogram={0: 8, 2: 3},
        energy_spacing=0.5,
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=1.0,
        energy_spacing=0.5,
        containers=(c0, c1),
    )
    df = wr.get_histogram()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"energy", "histogram"}
    # bin 0: 10+8=18, bin 1: 5+0=5, bin 2: 0+3=3
    assert df.loc[0, "histogram"] == 18
    assert df.loc[1, "histogram"] == 5
    assert df.loc[2, "histogram"] == 3


def test_n_walkers():
    """n_walkers returns the number of containers."""
    c = _make_mock_container(entropy={}, histogram={}, energy_spacing=0.1)
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c, c, c),
    )
    assert wr.n_walkers == 3


def test_get_entropy_single_walker_matches_container():
    """W=1: get_entropy produces same values as the container would."""
    entropy = {0: 3.0, 1: 5.0, 2: 4.0}
    c = _make_mock_container(
        entropy=entropy,
        histogram={0: 10, 1: 8, 2: 6},
        energy_spacing=0.25,
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=1.0,
        energy_spacing=0.25,
        containers=(c,),
    )
    df = wr.get_entropy()
    # Entropy: {0: 3.0, 1: 5.0, 2: 4.0}
    # min=3.0, shifted: {0: 0, 1: 2, 2: 1}
    assert df.loc[0, "entropy"] == pytest.approx(0.0)
    assert df.loc[1, "entropy"] == pytest.approx(2.0)
    assert df.loc[2, "entropy"] == pytest.approx(1.0)
    assert df.loc[0, "energy"] == pytest.approx(0.0)
    assert df.loc[1, "energy"] == pytest.approx(0.25)
    assert df.loc[2, "energy"] == pytest.approx(0.50)


def test_get_entropy_empty_returns_empty_dataframe():
    """Empty entropy dicts produce an empty DataFrame."""
    c = _make_mock_container(entropy={}, histogram={}, energy_spacing=0.1)
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c,),
    )
    df = wr.get_entropy()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_get_entropy_no_entropy_key_returns_none():
    """Container with no entropy data returns None."""
    from unittest.mock import MagicMock

    c = MagicMock()
    c._last_state = {"histogram": {0: 5}}
    c.ensemble_parameters = {"energy_spacing": 0.1}
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c,),
    )
    assert wr.get_entropy() is None


def test_get_histogram_no_histogram_key_returns_none():
    """Container with no histogram data returns None."""
    from unittest.mock import MagicMock

    c = MagicMock()
    c._last_state = {"entropy": {0: 1.0}}
    c.ensemble_parameters = {"energy_spacing": 0.1}
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c,),
    )
    assert wr.get_histogram() is None


def test_get_entropy_respects_fill_factor_limit():
    """fill_factor_limit selects historical entropy per walker before merge."""
    c0 = _make_mock_container(
        entropy={0: 10.0, 1: 20.0},
        histogram={0: 5, 1: 3},
        energy_spacing=1.0,
        fill_factor=0.125,
        fill_factor_history={0: 1.0, 100: 0.5, 200: 0.25, 300: 0.125},
        entropy_history={
            100: {0: 2.0, 1: 4.0},
            200: {0: 3.0, 1: 5.0},
            300: {0: 10.0, 1: 20.0},
        },
    )
    c1 = _make_mock_container(
        entropy={0: 12.0, 1: 22.0},
        histogram={0: 7, 1: 2},
        energy_spacing=1.0,
        fill_factor=0.125,
        fill_factor_history={0: 1.0, 100: 0.5, 200: 0.25, 300: 0.125},
        entropy_history={
            100: {0: 6.0, 1: 8.0},
            200: {0: 7.0, 1: 9.0},
            300: {0: 12.0, 1: 22.0},
        },
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c0, c1),
    )
    # fill_factor_limit=0.5: first ff_history entry <= 0.5 is step 100
    # c0 entropy at step 100: {0: 2.0, 1: 4.0}
    # c1 entropy at step 100: {0: 6.0, 1: 8.0}
    # merged: {0: 4.0, 1: 6.0}, shifted: {0: 0.0, 1: 2.0}
    df = wr.get_entropy(fill_factor_limit=0.5)
    assert df.loc[0, "entropy"] == pytest.approx(0.0)
    assert df.loc[1, "entropy"] == pytest.approx(2.0)
