"""Unit tests for mchammer_pt.analysis.dos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ase.units import kB

from mchammer_pt.analysis.dos import reweight_canonical_from_dos, stitch_entropy


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
    # the global rebase to min=0.
    rebased_a = entropy_a - entropy_a.min()
    at_lowest = stitched.loc[
        np.isclose(stitched["energy"], -10.0), "entropy"
    ].item()
    at_overlap = stitched.loc[
        np.isclose(stitched["energy"], -8.0), "entropy"
    ].item()
    assert np.isclose(at_lowest, rebased_a[0])
    assert np.isclose(at_overlap, rebased_a[-1])


def test_stitch_entropy_fills_interior_gap_with_neg_inf():
    # Two windows that align on a 0.5 grid but together leave one
    # interior bin (-7.5) unpopulated: window A covers [-8.0, -7.0]
    # but is missing its -7.5 bin, and window B starts at -7.0. The
    # stitched output must be a complete uniform grid with -inf at the
    # forbidden bin.
    df_a = pd.DataFrame({
        "energy": np.array([-8.0, -7.0]),
        "entropy": np.array([0.0, 1.0]),
    })
    df_b = pd.DataFrame({
        "energy": np.array([-7.0, -6.5, -6.0]),
        "entropy": np.array([1.0, 1.2, 1.5]),
    })
    stitched, _ = stitch_entropy([df_a, df_b], 0.5)

    # Complete grid from -8.0 to -6.0 in 0.5 steps: 5 bins.
    expected_energies = np.array([-8.0, -7.5, -7.0, -6.5, -6.0])
    assert np.allclose(stitched["energy"].to_numpy(), expected_energies)
    # Uniform spacing throughout.
    assert np.allclose(np.diff(stitched["energy"].to_numpy()), 0.5)
    # The unpopulated interior bin carries -inf; the rest are finite.
    gap = stitched.loc[np.isclose(stitched["energy"], -7.5), "entropy"].item()
    assert gap == -np.inf
    assert np.isfinite(
        stitched.loc[~np.isclose(stitched["energy"], -7.5), "entropy"]
    ).all()


def test_stitch_entropy_no_frontier_extrapolation():
    # The complete grid spans only the populated range -- no -inf bins
    # below the lowest or above the highest populated energy.
    df_a = pd.DataFrame({
        "energy": np.array([-8.0, -7.5, -7.0]),
        "entropy": np.array([0.0, 0.5, 1.0]),
    })
    df_b = pd.DataFrame({
        "energy": np.array([-7.0, -6.5, -6.0]),
        "entropy": np.array([1.0, 1.2, 1.5]),
    })
    stitched, _ = stitch_entropy([df_a, df_b], 0.5)

    assert stitched["energy"].min() == -8.0
    assert stitched["energy"].max() == -6.0
    # Fully populated, contiguous range -> no -inf anywhere.
    assert np.isfinite(stitched["entropy"]).all()
    assert stitched["entropy"].iloc[0] == 0.0


def test_stitch_entropy_neg_inf_round_trips_through_csv(tmp_path):
    # A stitched DOS with a -inf gap must survive a CSV write/read so
    # downstream tools (mchammer-pt-coexistence) see g=0 bins intact.
    df_a = pd.DataFrame({
        "energy": np.array([-8.0, -7.0]),
        "entropy": np.array([0.0, 1.0]),
    })
    df_b = pd.DataFrame({
        "energy": np.array([-7.0, -6.5, -6.0]),
        "entropy": np.array([1.0, 1.2, 1.5]),
    })
    stitched, _ = stitch_entropy([df_a, df_b], 0.5)

    path = tmp_path / "stitched_dos.csv"
    stitched.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    assert reloaded.loc[
        np.isclose(reloaded["energy"], -7.5), "entropy"
    ].item() == -np.inf
    assert np.allclose(
        reloaded["energy"].to_numpy(), stitched["energy"].to_numpy()
    )
    finite = np.isfinite(stitched["entropy"].to_numpy())
    assert np.allclose(
        reloaded["entropy"].to_numpy()[finite],
        stitched["entropy"].to_numpy()[finite],
    )


def test_stitch_entropy_rebases_on_finite_minimum_with_neg_inf_input():
    # A window that already carries a -inf (g=0) bin -- e.g. a
    # re-stitched complete histogram. The rebase offset must be the
    # finite minimum; folding -inf in would yield NaN/inf and corrupt
    # every bin.
    df = pd.DataFrame({
        "energy": np.array([0.0, 0.5, 1.0, 1.5]),
        "entropy": np.array([2.0, -np.inf, 3.0, 5.0]),
    })
    stitched, _ = stitch_entropy([df], 0.5)
    ent = stitched["entropy"].to_numpy()
    finite = np.isfinite(ent)
    # Finite bins rebased so their minimum is zero; no NaN anywhere.
    assert np.allclose(ent[finite], [0.0, 1.0, 3.0])
    assert not np.isnan(ent).any()
    # The pre-existing g=0 bin stays -inf.
    assert np.isneginf(ent[~finite]).all()


def test_stitch_entropy_raises_when_all_bins_g_zero():
    # Every bin g=0 (all -inf): no finite minimum to rebase against.
    df = pd.DataFrame({
        "energy": np.array([0.0, 0.5, 1.0]),
        "entropy": np.array([-np.inf, -np.inf, -np.inf]),
    })
    with pytest.raises(ValueError, match="no finite entropy"):
        stitch_entropy([df], 0.5)


def test_stitch_entropy_robust_to_ulp_drift():
    # Two windows on the same logical 0.5-eV grid, but each computed by
    # an independent process so the shared bins differ at the ULP level.
    # Exact float matching would silently drop them; integer-bin matching
    # picks them up.
    bins_a = np.arange(-20, -15)         # bin indices -20, ..., -16
    bins_b = np.arange(-17, -12)         # bin indices -17, ..., -13
    e_a = bins_a * 0.5 + 1e-12           # ULP-level drift
    e_b = bins_b * 0.5 - 1e-12
    df_a = pd.DataFrame({"energy": e_a, "entropy": np.array([0.0, 0.4, 0.7, 0.9, 1.0])})
    df_b = pd.DataFrame({"energy": e_b, "entropy": np.array([0.8, 0.9, 1.1, 1.4, 1.8])})

    stitched, errors = stitch_entropy([df_a, df_b], 0.5)

    # Overlap is bins {-17, -16}: aligns cleanly despite ULP drift.
    assert errors["0-1"] < 1e-9
    # Output covers the union of bins {-20..-13}, eight points.
    assert len(stitched) == 8


def test_stitch_entropy_three_windows_accumulates_offsets():
    # Three windows sampled from a single smooth truth curve, each with
    # a different additive offset. The algorithm aligns left-to-right;
    # window C's offset must be removed relative to B's already-shifted
    # frame, not its raw input frame.
    grid = np.arange(-10.0, -5.0 + 1e-9, 0.5)
    truth = 0.1 * (grid + 10.0) ** 2  # smooth, monotone

    def window(emin: float, emax: float, offset: float) -> pd.DataFrame:
        mask = (grid >= emin - 1e-9) & (grid <= emax + 1e-9)
        return pd.DataFrame({
            "energy": grid[mask],
            "entropy": truth[mask] + offset,
        })

    df_a = window(-10.0, -8.0, 0.0)
    df_b = window(-8.5, -6.5, 5.0)
    df_c = window(-7.0, -5.0, 13.0)

    stitched, errors = stitch_entropy([df_a, df_b, df_c], 0.5)

    assert set(errors.keys()) == {"0-1", "1-2"}
    assert all(v < 1e-9 for v in errors.values())

    # truth starts at 0, so rebasing is a no-op; stitched must equal truth.
    for E, s_truth in zip(grid, truth, strict=True):
        row = stitched.loc[np.isclose(stitched["energy"], E)]
        assert len(row) == 1
        assert abs(row["entropy"].item() - s_truth) < 1e-9


def test_stitch_entropy_input_order_independent():
    # Reversing the input order must produce the same stitched curve.
    # The error-key labels reflect the original input indices.
    energies_a = np.arange(-10.0, -8.0 + 1e-9, 0.5)
    entropy_a = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
    energies_b = np.arange(-8.5, -6.5 + 1e-9, 0.5)
    entropy_b = np.array([0.8, 0.9, 1.1, 1.4, 1.8]) + 5.0
    df_a = pd.DataFrame({"energy": energies_a, "entropy": entropy_a})
    df_b = pd.DataFrame({"energy": energies_b, "entropy": entropy_b})

    stitched_fwd, errors_fwd = stitch_entropy([df_a, df_b], 0.5)
    stitched_rev, errors_rev = stitch_entropy([df_b, df_a], 0.5)

    pd.testing.assert_frame_equal(
        stitched_fwd.sort_values("energy").reset_index(drop=True),
        stitched_rev.sort_values("energy").reset_index(drop=True),
    )
    assert "0-1" in errors_fwd
    # When B is input index 0 and A is input index 1, the energy-sorted
    # pair is (A, B) = (1, 0), so the error key is "1-0".
    assert "1-0" in errors_rev


def test_stitch_entropy_overlap_averaging_with_scatter():
    # Construct an overlap region with genuine scatter so groupby.mean()
    # in the overlap is observable and overlap_errors is non-zero.
    energies_a = np.array([-10.0, -9.5, -9.0, -8.5, -8.0])
    entropy_a = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    energies_b = np.array([-8.5, -8.0, -7.5, -7.0])
    # In the shared bins {-8.5, -8.0}, A reports [0.6, 0.8] and B reports
    # values whose mean differs from A's by exactly 5.0 but with scatter.
    entropy_b = np.array([0.7, 0.7, 0.9, 1.0]) + 5.0
    df_a = pd.DataFrame({"energy": energies_a, "entropy": entropy_a})
    df_b = pd.DataFrame({"energy": energies_b, "entropy": entropy_b})

    stitched, errors = stitch_entropy([df_a, df_b], 0.5)

    # Shifted B's overlap values are 0.7, 0.7 (offset = 5.0).
    # Averaged with A's 0.6, 0.8: stitched overlap is 0.65, 0.75.
    at_m85 = stitched.loc[
        np.isclose(stitched["energy"], -8.5), "entropy"
    ].item()
    at_m80 = stitched.loc[
        np.isclose(stitched["energy"], -8.0), "entropy"
    ].item()
    assert np.isclose(at_m85, 0.65)
    assert np.isclose(at_m80, 0.75)
    # Non-zero scatter in the shifted overlap: std of [0.1, -0.1] = sqrt(0.02).
    assert np.isclose(errors["0-1"], np.std([0.1, -0.1], ddof=1))


def test_stitch_entropy_raises_when_no_overlap():
    df_a = pd.DataFrame({
        "energy": np.arange(-10.0, -8.0 + 1e-9, 0.5),
        "entropy": np.zeros(5),
    })
    df_b = pd.DataFrame({
        "energy": np.arange(-7.0, -5.0 + 1e-9, 0.5),
        "entropy": np.zeros(5),
    })
    with pytest.raises(ValueError, match="No overlapping bins"):
        stitch_entropy([df_a, df_b], 0.5)


def test_stitch_entropy_raises_when_window_off_grid():
    df_a = pd.DataFrame({
        "energy": np.array([-10.0, -9.5, -9.0]),
        "entropy": np.array([0.0, 0.5, 1.0]),
    })
    df_b = pd.DataFrame({
        # 0.05 eV off the 0.5 eV grid, far beyond ULP tolerance.
        "energy": np.array([-9.45, -8.95, -8.45]),
        "entropy": np.array([0.0, 0.3, 0.7]),
    })
    with pytest.raises(ValueError, match="off the energy_spacing"):
        stitch_entropy([df_a, df_b], 0.5)


def test_stitch_entropy_raises_when_per_window_empty():
    with pytest.raises(ValueError, match="per_window is empty"):
        stitch_entropy([], 0.5)


def test_stitch_entropy_raises_when_a_window_is_empty():
    df_a = pd.DataFrame({
        "energy": np.array([0.0, 0.5]), "entropy": np.array([0.0, 0.0]),
    })
    df_b = pd.DataFrame({
        "energy": np.array([], dtype=float),
        "entropy": np.array([], dtype=float),
    })
    with pytest.raises(ValueError, match="Window 1 has no rows"):
        stitch_entropy([df_a, df_b], 0.5)


def test_reweight_canonical_rejects_empty_dos():
    with pytest.raises(ValueError, match="dos has no rows"):
        reweight_canonical_from_dos(
            pd.DataFrame({"energy": [], "entropy": []}), np.array([300.0])
        )


def test_stitch_entropy_raises_when_spacing_non_positive():
    df_a = pd.DataFrame({
        "energy": np.array([0.0, 1.0]),
        "entropy": np.array([0.0, 0.0]),
    })
    with pytest.raises(ValueError, match="energy_spacing must be"):
        stitch_entropy([df_a], 0.0)


def test_reweight_canonical_two_level_system():
    # Two energies, equal degeneracy: <E> -> midpoint as T -> infinity.
    dos = pd.DataFrame({
        "energy": np.array([-1.0, 0.0]),
        "entropy": np.array([0.0, 0.0]),
    })
    df = reweight_canonical_from_dos(dos, np.array([1.0, 1e10]))
    assert list(df.columns) == ["T_K", "E_mean", "var_E", "Cv"]
    assert df["E_mean"].iloc[0] < -0.99
    assert abs(df["E_mean"].iloc[1] - (-0.5)) < 1e-6


def test_reweight_canonical_two_level_cv_against_analytic():
    # Two-level system, equal degeneracy, gap = 1 eV. Pick T such that
    # kBT ~ gap so Cv is well within machine precision of its analytic
    # value. Analytic: var_E = gap**2 * p_0 * p_1; Cv = var_E / (kB * T**2).
    gap = 1.0
    T = gap / kB  # kBT = 1 eV exactly
    dos = pd.DataFrame({
        "energy": np.array([-gap, 0.0]),
        "entropy": np.array([0.0, 0.0]),
    })
    df = reweight_canonical_from_dos(dos, np.array([T]))

    beta = 1.0 / (kB * T)
    p_excited = 1.0 / (1.0 + np.exp(beta * gap))
    p_ground = 1.0 - p_excited
    var_analytic = gap ** 2 * p_ground * p_excited
    cv_analytic = var_analytic / (kB * T ** 2)

    assert abs(df["var_E"].iloc[0] - var_analytic) < 1e-12
    assert abs(df["Cv"].iloc[0] - cv_analytic) < 1e-12


def test_reweight_canonical_uses_log_space_no_underflow():
    # Build a DOS whose ln g range exceeds log(realmax) so any naive
    # exp(ln g) would underflow.
    energies = np.linspace(-1000.0, 0.0, 101)
    entropy = np.linspace(0.0, 800.0, 101)
    dos = pd.DataFrame({"energy": energies, "entropy": entropy})
    df = reweight_canonical_from_dos(dos, np.array([300.0]))
    # With slope(ln g)/ΔE = 0.8 per eV and β ≈ 38.7 eV^-1 at T=300 K,
    # log_w = ln g - β E is monotonically decreasing in E (β dominates
    # the entropy slope by ~50x), so the weight concentrates at the
    # lower endpoint and <E> should sit at the lowest energy bin.
    assert np.isfinite(df["E_mean"].iloc[0])
    assert abs(df["E_mean"].iloc[0] - energies.min()) < 1e-6


def test_reweight_canonical_rejects_non_positive_temperatures():
    dos = pd.DataFrame({
        "energy": np.array([-1.0, 0.0]),
        "entropy": np.array([0.0, 0.0]),
    })
    with pytest.raises(ValueError, match="strictly positive"):
        reweight_canonical_from_dos(dos, np.array([300.0, 0.0]))
