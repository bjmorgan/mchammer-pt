"""Unit tests for mchammer_pt.dos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mchammer_pt.dos import stitch_entropy


def test_stitch_entropy_two_windows_aligns_in_overlap():
    # Window A covers [-10.0, -8.0]; window B covers [-8.5, -6.5].
    # In the overlap [-8.5, -8.0], B is shifted up by +5.0; stitching
    # should remove that offset.
    energies_a = np.arange(-10.0, -8.0 + 1e-9, 0.5)
    entropy_a = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
    energies_b = np.arange(-8.5, -6.5 + 1e-9, 0.5)
    entropy_b = np.array([0.8, 0.9, 1.1, 1.4, 1.8]) + 5.0
    df_a = pd.DataFrame({"energy": energies_a, "entropy": entropy_a})
    df_b = pd.DataFrame({"energy": energies_b, "entropy": entropy_b})

    stitched, errors = stitch_entropy([df_a, df_b], energy_spacing=0.5)

    assert len(stitched) == 8
    assert stitched["entropy"].iloc[0] >= 0.0 - 1e-9
    assert errors["0-1"] < 1e-9


def test_stitch_entropy_raises_when_no_overlap():
    df_a = pd.DataFrame({
        "energy": np.arange(-10.0, -8.0 + 1e-9, 0.5),
        "entropy": np.zeros(5),
    })
    df_b = pd.DataFrame({
        "energy": np.arange(-7.0, -5.0 + 1e-9, 0.5),
        "entropy": np.zeros(5),
    })
    with pytest.raises(ValueError, match="No overlap"):
        stitch_entropy([df_a, df_b], energy_spacing=0.5)
