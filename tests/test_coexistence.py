"""Unit tests for mchammer_pt.analysis.coexistence."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase.units import kB

from mchammer_pt.analysis.coexistence import (
    NoBracketError,
    NotBimodalError,
    PhaseSplit,
    _auto_bracket,
    _parabolic_vertex,
    _partition_means,
    _partition_sums,
    find_phase_split,
)
from tests._coexistence_fixtures import single_gaussian_dos, two_gaussian_dos


def test_not_bimodal_error_is_value_error():
    assert issubclass(NotBimodalError, ValueError)


def test_no_bracket_error_is_value_error():
    assert issubclass(NoBracketError, ValueError)


def test_exceptions_carry_messages():
    e1 = NotBimodalError("only one peak at T=300 K")
    e2 = NoBracketError("imbalance has same sign at both endpoints")
    assert "300" in str(e1)
    assert "imbalance" in str(e2)


def test_parabolic_vertex_recovers_known_minimum():
    # y = 2*(x - 3.5)**2 + 1  -> vertex at x = 3.5
    xs = np.array([3.0, 4.0, 5.0])
    ys = 2.0 * (xs - 3.5) ** 2 + 1.0
    x_vertex = _parabolic_vertex(xs[0], xs[1], xs[2], ys[0], ys[1], ys[2])
    assert abs(x_vertex - 3.5) < 1e-12


def test_parabolic_vertex_recovers_known_maximum():
    # y = -3*(x - 7.25)**2 + 5  -> vertex at x = 7.25
    xs = np.array([7.0, 8.0, 9.0])
    ys = -3.0 * (xs - 7.25) ** 2 + 5.0
    x_vertex = _parabolic_vertex(xs[0], xs[1], xs[2], ys[0], ys[1], ys[2])
    assert abs(x_vertex - 7.25) < 1e-12


def test_parabolic_vertex_falls_back_to_centre_when_linear():
    # Three points with distinct x but collinear in y: the local fit
    # is linear (a == 0), no parabolic vertex exists, fallback to x_c.
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([10.0, 20.0, 30.0])
    x_vertex = _parabolic_vertex(xs[0], xs[1], xs[2], ys[0], ys[1], ys[2])
    assert x_vertex == 2.0


def test_parabolic_vertex_falls_back_to_centre_on_coincident_x():
    # Two x-values coincide: denominator vanishes; fallback to x_c.
    x_vertex = _parabolic_vertex(2.0, 2.0, 3.0, 1.0, 2.0, 5.0)
    assert x_vertex == 2.0


def test_phase_split_is_frozen_dataclass():
    split = PhaseSplit(
        E_peak_low=-1.0, E_peak_high=1.0, E_star=0.0, T_K=300.0,
    )
    with pytest.raises(FrozenInstanceError):
        split.E_star = 0.5  # type: ignore[misc]


def test_find_phase_split_symmetric_two_gaussian():
    # Symmetric peaks of equal width at -1.0 eV and +1.0 eV. DOS peaks
    # sit at the bin centres exactly. The valley of P(E|T) between them
    # is close to the midpoint E* = 0, with a small T-dependent shift
    # from the linear Boltzmann tilt of phi = beta * E - ln g.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    split = find_phase_split(dos, T_K=300.0)
    assert split.T_K == 300.0
    assert abs(split.E_peak_low - (-1.0)) < 1e-3
    assert abs(split.E_peak_high - 1.0) < 1e-3
    # The Boltzmann tilt shifts the saddle of P off-centre by an amount
    # of order kT / (d^2 ln g / dE^2)|_centre; for sigma=0.1 and
    # T=300 K this is ~5e-3 eV. Allow one bin spacing of tolerance.
    assert abs(split.E_star) < 0.01


def test_find_phase_split_sub_bin_precision():
    # Peaks at non-bin-centre, non-midbin positions so neither bin
    # neighbour ties the peak bin's ln_g value (which would break
    # strict-greater peak detection). Parabolic refinement should
    # recover the analytic peak locations to better than
    # energy_spacing / 10 = 0.001 eV.
    dos = two_gaussian_dos(
        E_low=-1.003, E_high=0.997,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    split = find_phase_split(dos, T_K=1_000_000.0)
    assert abs(split.E_peak_low - (-1.003)) < 0.001
    assert abs(split.E_peak_high - 0.997) < 0.001


def test_find_phase_split_raises_on_unimodal():
    dos = single_gaussian_dos(
        E_centre=0.0, sigma=0.5,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(NotBimodalError):
        find_phase_split(dos, T_K=300.0)


def test_find_phase_split_raises_on_adjacent_peaks():
    # Two peaks separated by only 2 bins (energy_spacing=0.1 -> 0.2 eV
    # apart), below the default min_peak_separation=5.
    dos = two_gaussian_dos(
        E_low=-0.1, E_high=0.1,
        sigma_low=0.02, sigma_high=0.02,
        weight_low=1.0, weight_high=1.0,
        E_min=-1.0, E_max=1.0, energy_spacing=0.1,
    )
    with pytest.raises(NotBimodalError):
        find_phase_split(dos, T_K=300.0)


def test_find_phase_split_accepts_custom_min_peak_separation():
    # Same DOS as above, but loosen the gap requirement.
    dos = two_gaussian_dos(
        E_low=-0.1, E_high=0.1,
        sigma_low=0.02, sigma_high=0.02,
        weight_low=1.0, weight_high=1.0,
        E_min=-1.0, E_max=1.0, energy_spacing=0.1,
    )
    split = find_phase_split(dos, T_K=300.0, min_peak_separation=1)
    assert split.E_peak_low < split.E_peak_high


def test_find_phase_split_rejects_non_positive_T_K():
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(ValueError, match="T_K must be > 0"):
        find_phase_split(dos, T_K=0.0)
    with pytest.raises(ValueError, match="T_K must be > 0"):
        find_phase_split(dos, T_K=-100.0)


def test_find_phase_split_rejects_empty_dos():
    import pandas as pd

    with pytest.raises(ValueError, match="dos has no rows"):
        find_phase_split(pd.DataFrame({"energy": [], "entropy": []}), T_K=300.0)


def test_find_phase_split_rejects_invalid_min_peak_separation():
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(ValueError, match="min_peak_separation must be >= 1"):
        find_phase_split(dos, T_K=300.0, min_peak_separation=0)


def test_partition_sums_full_bins_only():
    # Two-bin DOS at E=-1 and E=+1 with ln g = 0 (equal counts).
    # With energy_spacing=2 and bin centres at -1 and +1, the bin
    # containing E* = +0.5 is bin 1 (covering [0, 2)); the fraction
    # from the left edge of that bin is f = (0.5 - 0) / 2 = 0.25,
    # so 0.25 of bin 1 goes low and 0.75 high. At very high T, the
    # Boltzmann factor exp(-beta E) is ~1 for both bins.
    energies = np.array([-1.0, 1.0])
    ln_g = np.array([0.0, 0.0])
    w_low, w_high = _partition_sums(energies, ln_g, T_K=1e10, E_star=0.5)
    # bin 0 weight ~ 1, bin 1 weight ~ 1 (after max-subtraction).
    # w_low gets bin 0 entirely + f * bin 1.
    # w_high gets (1 - f) * bin 1.
    # Ratios: w_low / (w_low + w_high) should be (1 + 0.25) / 2.0 = 0.625.
    total = w_low + w_high
    assert abs(w_low / total - 0.625) < 1e-6


def test_partition_sums_preserves_total():
    # Property: w_low + w_high must equal the sum over all bins of
    # g * exp(-beta E), for any E_star strictly inside the grid range.
    energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ln_g = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    T_K = 500.0
    beta = 1.0 / (kB * T_K)
    log_w = ln_g - beta * energies
    log_w -= log_w.max()
    total = float(np.exp(log_w).sum())
    for E_star in [-1.5, -0.3, 0.0, 0.7, 1.5]:
        w_low, w_high = _partition_sums(
            energies, ln_g, T_K=T_K, E_star=E_star,
        )
        assert abs((w_low + w_high) - total) / total < 1e-9


def test_partition_means_symmetric_at_midpoint():
    # Symmetric two-Gaussian DOS, E* at zero. T=1e4 K is chosen so that
    # kB*T (~ 0.86 eV) >> sigma (0.1 eV), meaning the Boltzmann tilt
    # within each peak is small (shift ~ beta * sigma^2 ~ 0.01 eV),
    # so the conditional mean on each side is within 0.05 eV of the
    # DOS peak centre. The two peaks are 2 eV apart with near-zero
    # weight at E=0, so each side is dominated by its own peak.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    mean_low, mean_high = _partition_means(
        energies, ln_g, T_K=1e4, E_star=0.0,
    )
    assert abs(mean_low - (-1.0)) < 0.05
    assert abs(mean_high - 1.0) < 0.05


def test_auto_bracket_brackets_the_cv_peak():
    # Two-Gaussian DOS with a clear first-order coexistence near
    # T = 500 K (kT ~ 0.043 eV; peaks separated by 2 eV). The
    # auto-bracket should return a (T_lo, T_hi) interval that
    # contains the Cv peak.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    T_lo, T_hi = _auto_bracket(dos)
    assert T_lo > 0.0
    assert T_hi > T_lo
    # The bracket must span a sensible Kelvin range — not collapse
    # to zero width.
    assert (T_hi - T_lo) / T_lo > 0.1
