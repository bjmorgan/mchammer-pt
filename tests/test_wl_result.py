"""Unit tests for WindowResult."""

from __future__ import annotations

import numpy as np
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
