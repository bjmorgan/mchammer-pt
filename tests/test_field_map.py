"""Tests for mchammer_pt.analysis.field.field_map.

Unit tests build a reweight_observables-shaped frame directly; the
end-to-end tests (added later) drive the real recorder -> stitch ->
reweight -> field_map path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mchammer_pt.analysis.field import field_map


def _canonical(tag: str, n_T: int, field_per_T: np.ndarray) -> pd.DataFrame:
    """Build a minimal reweight_observables-shaped frame.

    ``field_per_T`` is an ``(n_T, S)`` array of per-pixel means. Decoy
    ``_sq_mean``/``_binder`` columns are added to prove they are ignored
    (and that the pixel-column regex does not miscount ``_sq_mean``).
    """
    _, S = field_per_T.shape
    data: dict[str, np.ndarray] = {
        "T_K": np.linspace(300.0, 600.0, n_T),
        "coverage": np.ones(n_T),
    }
    for i in range(S):
        data[f"{tag}_{i}_mean"] = field_per_T[:, i]
        data[f"{tag}_{i}_sq_mean"] = field_per_T[:, i] ** 2  # decoy
        data[f"{tag}_{i}_binder"] = np.full(n_T, -999.0)  # decoy
    return pd.DataFrame(data)


def test_field_map_reshapes_in_ravel_order() -> None:
    """Pixels 0,1,2,3 fold to [[0,1],[2,3]] (C-order), matching ravel."""
    n_T = 3
    field = np.tile(np.arange(4.0), (n_T, 1))  # (3, 4): columns 0,1,2,3
    canonical = _canonical("f", n_T, field)

    temps, values = field_map(canonical, "f", (2, 2))

    assert values.shape == (n_T, 2, 2)
    np.testing.assert_allclose(values[0], [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(temps, canonical["T_K"].to_numpy())


def test_field_map_folds_three_dimensional_shape_ignoring_decoys() -> None:
    """A 3-D shape reshapes C-order; decoy ``_sq_mean``/``_binder`` ignored.

    ``_canonical`` adds a decoy ``_sq_mean`` and ``_binder`` column for every
    pixel, so a correct run also proves those columns are neither read nor
    miscounted by the pixel-column match.
    """
    n_T = 2
    field = np.tile(np.arange(8.0), (n_T, 1))  # pixels 0..7 (+ decoy columns)
    canonical = _canonical("f", n_T, field)

    _, values = field_map(canonical, "f", (2, 2, 2))

    assert values.shape == (n_T, 2, 2, 2)
    np.testing.assert_allclose(
        values[0], [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]
    )


def test_field_map_shape_mismatch_raises() -> None:
    """math.prod(shape) != recorded pixel count is rejected."""
    field = np.tile(np.arange(4.0), (2, 1))  # 4 pixels
    canonical = _canonical("f", 2, field)

    with pytest.raises(ValueError, match="contiguous"):
        field_map(canonical, "f", (3,))  # prod == 3 != 4


def test_field_map_missing_pixel_raises() -> None:
    """A gap in the pixel-mean columns is rejected."""
    field = np.tile(np.arange(4.0), (2, 1))
    canonical = _canonical("f", 2, field).drop(columns=["f_2_mean"])

    with pytest.raises(ValueError, match="contiguous"):
        field_map(canonical, "f", (2, 2))


def test_field_map_no_temperature_column_raises() -> None:
    """A frame without ``T_K`` is rejected."""
    field = np.tile(np.arange(4.0), (2, 1))
    canonical = _canonical("f", 2, field).drop(columns=["T_K"])

    with pytest.raises(ValueError, match="T_K"):
        field_map(canonical, "f", (2, 2))


def test_field_map_unknown_tag_raises() -> None:
    """A tag with no matching pixel-mean columns is rejected."""
    field = np.tile(np.arange(4.0), (2, 1))
    canonical = _canonical("f", 2, field)

    with pytest.raises(ValueError, match="contiguous"):
        field_map(canonical, "other", (2, 2))
