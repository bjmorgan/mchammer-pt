"""Unit tests for mchammer_pt.analysis.coexistence."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from ase.units import kB

import mchammer_pt.analysis.coexistence as _coexistence_mod
from mchammer_pt.analysis._partition import (
    partition_means as _partition_means,
    partition_sums as _partition_sums,
)
from mchammer_pt.analysis.coexistence import (
    CoexistencePoint,
    NoBracketError,
    NotBimodalError,
    PhaseSplit,
    _auto_bracket,
    _parabolic_vertex,
    equal_area_temperature,
    find_phase_split,
)
from mchammer_pt.analysis.dos import stitch_entropy
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


def test_find_phase_split_raises_when_phi_has_no_interior_maximum():
    # At very low T (huge beta), phi(E) = beta*E - ln g is dominated
    # by the linear term and is monotonic across the inter-peak range.
    # find_phase_split detects this and raises (clipping to a valid
    # parabolic-fit window would silently return a corrupt E_star).
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(NotBimodalError, match="no interior maximum"):
        find_phase_split(dos, T_K=1.0)


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


def test_auto_bracket_finds_sign_change():
    # Two-Gaussian DOS with weight_high=2.0 has a genuine first-order
    # coexistence near T_c ≈ 33 500 K where imbalance changes sign.
    # _auto_bracket uses the peak energy separation and entropy
    # difference to derive kT_scale, then scans imbalance(T) on a
    # log-spaced grid and returns the first adjacent pair that
    # straddles zero.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    T_lo, T_hi = _auto_bracket(dos)
    assert T_lo > 0.0
    assert T_hi > T_lo
    # The bracket must straddle T_c ≈ 33 500 K.
    T_c_analytic = 2.0 / (kB * np.log(2.0))
    assert T_lo < T_c_analytic < T_hi


def test_auto_bracket_raises_on_symmetric_dos():
    # Symmetric DOS (equal weights) has T_c → ∞: imbalance never
    # changes sign. _auto_bracket raises ValueError.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=1.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(ValueError, match="auto_bracket"):
        _auto_bracket(dos)


def test_equal_area_temperature_raises_no_bracket_when_auto_scan_finds_no_crossing(
    monkeypatch,
):
    # Two-phase DOS with a genuine sign change in imbalance(T). Shrink
    # _AUTO_BRACKET_KT_HIGH_FRAC so the scan window covers only a
    # fraction of T_c — entirely on the low-T side where imbalance is
    # positive throughout. The scan finds no sign change and must raise
    # NoBracketError (not plain ValueError).
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    # Restrict the scan to [0.05, 0.1] * kT_scale / kB, which is far
    # below T_c ≈ kT_scale / kB, so imbalance is positive throughout.
    monkeypatch.setattr(_coexistence_mod, "_AUTO_BRACKET_KT_HIGH_FRAC", 0.1)
    with pytest.raises(NoBracketError, match="auto_bracket"):
        _auto_bracket(dos)


def test_coexistence_point_is_frozen_dataclass():
    split = PhaseSplit(
        E_peak_low=-1.0, E_peak_high=1.0, E_star=0.0, T_K=500.0,
    )
    cp = CoexistencePoint(
        split=split,
        latent_heat=2.0, barrier_height=0.1,
        weight_imbalance=1e-9, n_bisection_steps=18,
    )
    with pytest.raises(FrozenInstanceError):
        cp.latent_heat = 3.0  # type: ignore[misc]


def test_coexistence_point_T_K_delegates_to_split():
    split = PhaseSplit(
        E_peak_low=-1.0, E_peak_high=1.0, E_star=0.0, T_K=500.0,
    )
    cp = CoexistencePoint(
        split=split,
        latent_heat=2.0, barrier_height=0.1,
        weight_imbalance=1e-9, n_bisection_steps=18,
    )
    assert cp.T_K == 500.0
    # T_K is a read-only property; no separate field to assign.
    with pytest.raises(AttributeError):
        cp.T_K = 600.0  # type: ignore[misc]


def test_equal_area_temperature_two_phase_dos():
    # Two-phase DOS with the high-E peak having more entropy
    # (weight_high=2.0). In the sharp-peak limit the equal-area
    # condition is:
    #   w_low * exp(beta * E_low) = w_high * exp(beta * E_high)
    # => T_c = (E_high - E_low) / (k_B * ln(w_high / w_low))
    #        = 2.0 / (k_B * ln 2) ≈ 33 500 K.
    # At T_c the saddle E*(T_c) lies near the midpoint 0 eV and the
    # latent heat equals the peak energy difference (2.0 eV).
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    T_c_analytic = 2.0 / (kB * np.log(2.0))
    result = equal_area_temperature(
        dos, T_bracket=(1000.0, 200_000.0),
    )
    assert isinstance(result, CoexistencePoint)
    # weight_imbalance is the residual |f(T_mid)| at convergence. With
    # the default xtol=1e-4 and df/dT ≈ 5e-4 /K at T_c ≈ 33 500 K,
    # the achievable imbalance is of order xtol * T_c * |df/dT| ≈ 1.7e-3.
    assert result.weight_imbalance < 0.01
    assert abs(result.split.E_star) < 0.01
    assert abs(result.latent_heat - 2.0) < 0.05
    assert result.n_bisection_steps >= 1
    assert result.barrier_height > 0.0
    # T_K must be within 5 % of the analytic value.
    assert abs(result.T_K - T_c_analytic) / T_c_analytic < 0.05


def test_equal_area_temperature_asymmetric_weights():
    # Asymmetric weights: a heavier high-E phase (weight_high=3.0,
    # weight_low=1.0). In the sharp-peak limit:
    # T_c = 2.0 / (k_B * ln 3) ≈ 21 100 K.
    # We only assert convergence and that T_K falls inside the bracket.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.05, sigma_high=0.05,
        weight_low=1.0, weight_high=3.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    T_c_analytic = 2.0 / (kB * np.log(3.0))
    result = equal_area_temperature(dos, T_bracket=(1000.0, 100_000.0))
    # Same residual reasoning as the two-phase test: achievable
    # weight_imbalance at xtol=1e-4 is O(1e-3) for this fixture.
    assert result.weight_imbalance < 0.01
    assert 1000.0 < result.T_K < 100_000.0
    assert abs(result.T_K - T_c_analytic) / T_c_analytic < 0.05


def test_equal_area_temperature_raises_on_unimodal():
    dos = single_gaussian_dos(
        E_centre=0.0, sigma=0.5,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(NoBracketError):
        equal_area_temperature(dos, T_bracket=(100.0, 1000.0))


def test_equal_area_temperature_auto_bracket_distinguishes_no_bimodality():
    # With no T_bracket supplied on a unimodal DOS, the auto-scan
    # finds no T at which shape analysis succeeds. The correct
    # diagnostic is NotBimodalError (no bimodal region anywhere),
    # not NoBracketError (which would imply a sign change existed
    # but lay outside the scanned range).
    dos = single_gaussian_dos(
        E_centre=0.0, sigma=0.5,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(NotBimodalError):
        equal_area_temperature(dos)


def test_equal_area_temperature_auto_bracket():
    # No T_bracket supplied: auto-bracket must locate the sign change
    # in imbalance(T) and return a sensible result.
    # Uses the same two-phase DOS as test_equal_area_temperature_two_phase_dos.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    result = equal_area_temperature(dos)
    # auto-bracket returns a narrower starting bracket so the residual
    # is smaller; still use a generous tolerance for robustness.
    assert result.weight_imbalance < 0.01
    assert abs(result.split.E_star) < 0.05
    T_c_analytic = 2.0 / (kB * np.log(2.0))
    assert abs(result.T_K - T_c_analytic) / T_c_analytic < 0.05


def test_equal_area_temperature_rejects_bad_bracket():
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(ValueError, match="T_bracket"):
        equal_area_temperature(dos, T_bracket=(1000.0, 100.0))  # T_lo >= T_hi
    with pytest.raises(ValueError, match="T_bracket"):
        equal_area_temperature(dos, T_bracket=(-10.0, 100.0))   # negative


def test_equal_area_temperature_rejects_bad_xtol():
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    with pytest.raises(ValueError, match="xtol"):
        equal_area_temperature(dos, T_bracket=(100.0, 200000.0), xtol=0.0)
    with pytest.raises(ValueError, match="xtol"):
        equal_area_temperature(dos, T_bracket=(100.0, 200000.0), xtol=-1.0)


def test_equal_area_temperature_raises_no_bracket_when_user_bracket_spans_non_bimodal():
    # User supplies a bracket whose lower end sits at a T so low that
    # phi has no interior maximum between the DOS peaks. The bisection
    # finds shape analysis failing mid-bracket and re-raises as
    # NoBracketError (the auto-bracket would have avoided this T range).
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    # T=1 K: at this temperature phi(E) is dominated by beta*E and
    # has no interior maximum; find_phase_split raises NotBimodalError
    # which the bisection wraps as NoBracketError ("shape analysis
    # failed at a bracket endpoint").
    with pytest.raises(NoBracketError, match="shape analysis failed"):
        equal_area_temperature(dos, T_bracket=(1.0, 200000.0))


def test_equal_area_temperature_barrier_height_matches_analytic_anchor():
    # Equal-width Gaussians at +/-1 eV with weights 1:2 have a
    # closed-form coexistence temperature
    #   T_c = (E_high - E_low) / (k_B * ln(w_high / w_low))
    # and an analytic ln g(E*=0) in the sharp-peak limit (peaks of
    # ln g - beta*E are well-separated, so logaddexp at the midpoint
    # is dominated by neither term until the saddle).
    # Anchor barrier_height to a value reasonable for these
    # parameters: in the limit sigma -> 0, ln g(0) - ln g(E_peak) ~
    # -E_peak^2 / (2 sigma^2) = -1/0.02 = -50; barrier_height in
    # energy units is k_B * T_c * 50.
    dos = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    result = equal_area_temperature(dos, T_bracket=(1000.0, 200000.0))
    T_c_analytic = 2.0 / (kB * np.log(2.0))
    # At E* ~ 0 (between symmetric-position peaks), the larger DOS
    # peak (weight 2) dominates the ln g(E*) computation through
    # logaddexp of two equally-tailed Gaussians; the barrier in
    # eV is approximately
    #     k_B * T_c * (1/(2 sigma^2) + ln(1 + w_low/w_high)).
    # The ln-correction term is small (ln 1.5 ~ 0.405) compared with
    # 1/(2*0.01) = 50, so a 5% tolerance is generous and tight enough
    # to catch unit slips (factors of k_B, k_B * T, etc).
    expected_barrier = kB * T_c_analytic * (1.0 / (2.0 * 0.1 ** 2) + np.log(1.5))
    assert result.barrier_height > 0.0
    assert abs(result.barrier_height - expected_barrier) / expected_barrier < 0.1


def test_public_surface_reexported_from_analysis():
    import mchammer_pt.analysis as analysis

    for name in (
        "PhaseSplit",
        "CoexistencePoint",
        "find_phase_split",
        "equal_area_temperature",
        "NotBimodalError",
        "NoBracketError",
    ):
        assert hasattr(analysis, name), f"missing re-export: {name}"


def test_stitch_then_equal_area_round_trip():
    # Build a synthetic two-Gaussian DOS with asymmetric weights
    # (finite equal-area T_c exists), then split it into two
    # overlapping windows that look like REWL output. Stitch them
    # back together and verify equal_area_temperature succeeds with
    # latent_heat > 0.
    full = two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    # Window A: E in [-2.0, 0.2]; Window B: E in [-0.2, 2.0]; overlap
    # [-0.2, 0.2]. Apply a constant additive shift to each window's
    # entropy column to mimic per-window unknown offsets.
    mask_a = (full["energy"] >= -2.0 - 1e-9) & (full["energy"] <= 0.2 + 1e-9)
    mask_b = (full["energy"] >= -0.2 - 1e-9) & (full["energy"] <= 2.0 + 1e-9)
    window_a = full[mask_a].copy()
    window_b = full[mask_b].copy()
    window_a["entropy"] = window_a["entropy"] + 3.0
    window_b["entropy"] = window_b["entropy"] - 7.0

    stitched, errors = stitch_entropy([window_a, window_b], 0.01)
    # Alignment within the overlap should be tight.
    assert all(v < 1e-6 for v in errors.values())

    result = equal_area_temperature(stitched)
    assert result.latent_heat > 0.0
    # By construction, latent heat should be close to E_high - E_low = 2 eV.
    assert abs(result.latent_heat - 2.0) < 0.05
