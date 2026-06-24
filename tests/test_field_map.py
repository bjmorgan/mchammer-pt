"""Tests for mchammer_pt.analysis.field.field_map.

Unit tests build a reweight_observables-shaped frame directly; the
end-to-end tests (added later) drive the real recorder -> stitch ->
reweight -> field_map path.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.units import kB
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)
from mchammer.observers.base_observer import BaseObserver

from mchammer_pt.analysis.field import field_map
from mchammer_pt.analysis.observables import (
    reweight_observables,
    stitch_observable_moments,
)
from mchammer_pt.wl_observable_recorder import EnergyBinnedObservableRecorder


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


# ---------------------------------------------------------------------------
# End-to-end: a generic ndarray field through the real pipeline.
#
# Material-agnostic by construction: the observer returns a synthetic
# deterministic field (pixel i = (i + 1) * sum(atomic numbers)); no
# structure-factor or material knowledge appears here.
# ---------------------------------------------------------------------------

ENERGY_SPACING = 1.0


class _FieldObserver(BaseObserver):
    """Deterministic length-S field: pixel i = (i + 1) * sum(numbers)."""

    def __init__(self, size: int, tag: str = "fld") -> None:
        super().__init__(interval=1, return_type=list, tag=tag)
        self._size = size

    def get_observable(self, structure: Any) -> np.ndarray:
        s = float(np.sum(structure.numbers))
        return np.arange(1, self._size + 1, dtype=float) * s


def _structure(z: int) -> Atoms:
    """Two atoms of atomic number ``z``; ``sum(numbers) == 2 * z``."""
    return Atoms(numbers=[z, z], positions=[[0, 0, 0], [0, 0, 1]])


def _mock_dc(record_state: dict[str, Any]) -> Any:
    """Mock unbounded-window WL container carrying one recorder store."""
    dc = MagicMock(spec=WangLandauDataContainer)
    dc._last_state = {
        "observable_records": {record_state["tag"]: record_state}
    }
    dc.ensemble_parameters = {
        "energy_spacing": ENERGY_SPACING,
        "energy_limit_left": None,
        "energy_limit_right": None,
    }
    return dc


def test_field_rides_pipeline_end_to_end() -> None:
    """A field reweights per pixel and folds back to a (n_T, 2, 2) map."""
    size, shape = 4, (2, 2)
    rec = EnergyBinnedObservableRecorder(_FieldObserver(size, tag="fld"))

    # Record a distinct, deterministic field into each of three bins.
    bins = [0, 1, 2]
    zs = {0: 1, 1: 6, 2: 8}  # sum(numbers) = 2, 12, 16
    counts = {0: 3, 1: 5, 2: 2}
    for b in bins:
        st = _structure(zs[b])
        for _ in range(counts[b]):
            rec.record(st, bin_index=b)

    # Per-bin field (constant within a bin): pixel i = (i+1) * sum(numbers).
    field = {
        b: (np.arange(1, size + 1) * (2 * zs[b])).astype(float) for b in bins
    }

    moments = stitch_observable_moments(
        [_mock_dc(rec.to_state())], ENERGY_SPACING
    )

    # Non-flat ln g so the reweighting is non-trivial. Everything is keyed by
    # bin, so the hand calculation never assumes bins == range(len(bins)).
    energy = {b: float(b * ENERGY_SPACING) for b in bins}
    log_g = {0: 0.0, 1: 1.5, 2: 0.7}
    dos = pd.DataFrame(
        {
            "energy": [energy[b] for b in bins],
            "entropy": [log_g[b] for b in bins],
        }
    )
    temperatures = np.array([300.0, 600.0])

    canonical = reweight_observables(moments["fld"], dos, temperatures)

    # Independent Boltzmann reweighting: w_b = exp(ln g_b - E_b / (kB T)).
    expected = np.zeros((len(temperatures), size))
    for t_idx, temp in enumerate(temperatures):
        w = np.array([np.exp(log_g[b] - energy[b] / (kB * temp)) for b in bins])
        w /= w.sum()
        expected[t_idx] = sum(w_b * field[b] for w_b, b in zip(w, bins, strict=True))

    # (a) reweight produced the field as per-pixel _mean columns.
    for i in range(size):
        np.testing.assert_allclose(
            canonical[f"fld_{i}_mean"].to_numpy(), expected[:, i], rtol=1e-9
        )

    # (b) field_map folds them back to a (n_T, 2, 2) map in ravel order.
    temps, values = field_map(canonical, "fld", shape)
    np.testing.assert_allclose(temps, temperatures)
    np.testing.assert_allclose(
        values, expected.reshape(len(temperatures), *shape), rtol=1e-9
    )


def test_field_map_round_trips_flat_dos() -> None:
    """Flat DOS, identical field in both bins: map equals the recorded field."""
    size, shape = 4, (2, 2)
    rec = EnergyBinnedObservableRecorder(_FieldObserver(size, tag="fld"))
    st = _structure(7)  # sum(numbers) = 14; field = [14, 28, 42, 56]
    rec.record(st, bin_index=0)
    rec.record(st, bin_index=1)  # same field in both bins

    moments = stitch_observable_moments(
        [_mock_dc(rec.to_state())], ENERGY_SPACING
    )
    dos = pd.DataFrame({"energy": [0.0, 1.0], "entropy": [0.0, 0.0]})
    temperatures = np.array([300.0, 500.0])

    canonical = reweight_observables(moments["fld"], dos, temperatures)
    _, values = field_map(canonical, "fld", shape)

    field = (np.arange(1, size + 1) * 14.0).reshape(shape)
    for t_idx in range(len(temperatures)):
        np.testing.assert_allclose(values[t_idx], field)
