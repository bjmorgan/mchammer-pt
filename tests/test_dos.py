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

    stitched, errors = stitch_entropy([df_a, df_b], 0.5)

    assert len(stitched) == 8
    assert stitched["entropy"].iloc[0] >= 0.0 - 1e-9
    assert errors["0-1"] < 1e-9

    # Pin the alignment direction: in the overlap region, the stitched
    # entropy must agree with df_a (the lower-energy reference) up to
    # the global rebase to min=0. Equivalently: the stitched ln g at
    # the lowest energy must equal entropy_a[0] - entropy_a[0] = 0
    # (it is the floor of the rebased curve), and the stitched ln g
    # at -8.0 (a shared bin) must equal entropy_a's value there, also
    # rebased.
    rebased_a = entropy_a - entropy_a.min()
    at_lowest = stitched.loc[
        np.isclose(stitched["energy"], -10.0), "entropy"
    ].item()
    at_overlap = stitched.loc[
        np.isclose(stitched["energy"], -8.0), "entropy"
    ].item()
    assert np.isclose(at_lowest, rebased_a[0])
    assert np.isclose(at_overlap, rebased_a[-1])


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
        stitch_entropy([df_a, df_b], 0.5)


def test_stitch_entropy_raises_when_bin_centres_do_not_align():
    # Ranges overlap on [-8.5, -8.0] but the two windows are on disjoint
    # bin grids — no shared bin centre. Must raise rather than silently
    # produce NaN.
    df_a = pd.DataFrame({
        "energy": np.array([-10.0, -9.0, -8.0]),
        "entropy": np.array([0.0, 0.5, 1.0]),
    })
    df_b = pd.DataFrame({
        "energy": np.array([-8.5, -7.5, -6.5]),
        "entropy": np.array([0.0, 0.3, 0.7]),
    })
    with pytest.raises(ValueError, match="No shared bin centres"):
        stitch_entropy([df_a, df_b], 0.5)
