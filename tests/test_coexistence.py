"""Unit tests for mchammer_pt.analysis.coexistence."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from ase.units import kB

from mchammer_pt.analysis._partition import partition_means as _partition_means
from mchammer_pt.analysis._partition import partition_sums as _partition_sums
from mchammer_pt.analysis.coexistence import (
    CoexistencePoint,
    NoBracketError,
    NotBimodalError,
    PhaseSplit,
    _parabolic_vertex,
    _two_dominant_peak_indices,
    equal_area_temperature,
    find_phase_split,
)
from tests._coexistence_fixtures import lattice_like_dos


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


def test_find_phase_split_rejects_non_positive_T_K():
    dos = pd.DataFrame({
        "energy": np.linspace(-1.0, 1.0, 21),
        "entropy": np.zeros(21),
    })
    with pytest.raises(ValueError, match="T_K must be > 0"):
        find_phase_split(dos, T_K=0.0)
    with pytest.raises(ValueError, match="T_K must be > 0"):
        find_phase_split(dos, T_K=-100.0)


def test_find_phase_split_rejects_empty_dos():
    with pytest.raises(ValueError, match="dos has no rows"):
        find_phase_split(pd.DataFrame({"energy": [], "entropy": []}), T_K=300.0)


def test_find_phase_split_rejects_invalid_min_peak_separation():
    dos = pd.DataFrame({
        "energy": np.linspace(-1.0, 1.0, 21),
        "entropy": np.zeros(21),
    })
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
    energies = np.linspace(-2.0, 2.0, 401)
    sigma = 0.1
    log_low = -((energies - (-1.0)) ** 2) / (2.0 * sigma ** 2)
    log_high = -((energies - 1.0) ** 2) / (2.0 * sigma ** 2)
    ln_g = np.logaddexp(log_low, log_high)
    ln_g -= ln_g.min()
    mean_low, mean_high = _partition_means(
        energies, ln_g, T_K=1e4, E_star=0.0,
    )
    assert abs(mean_low - (-1.0)) < 0.05
    assert abs(mean_high - 1.0) < 0.05


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


def test_equal_area_temperature_rejects_bad_bracket():
    dos = pd.DataFrame({
        "energy": np.linspace(-1.0, 1.0, 21),
        "entropy": np.zeros(21),
    })
    with pytest.raises(ValueError, match="T_bracket"):
        equal_area_temperature(dos, T_bracket=(1000.0, 100.0))
    with pytest.raises(ValueError, match="T_bracket"):
        equal_area_temperature(dos, T_bracket=(-10.0, 100.0))


def test_equal_area_temperature_rejects_bad_xtol():
    dos = pd.DataFrame({
        "energy": np.linspace(-1.0, 1.0, 21),
        "entropy": np.zeros(21),
    })
    with pytest.raises(ValueError, match="xtol"):
        equal_area_temperature(dos, T_bracket=(100.0, 200000.0), xtol=0.0)
    with pytest.raises(ValueError, match="xtol"):
        equal_area_temperature(dos, T_bracket=(100.0, 200000.0), xtol=-1.0)


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


def test_lattice_like_dos_is_monotone_in_energy():
    # For a=1, beta_c=10, c=1 the boundary of monotonicity is the
    # real root of E**3 - E - 2.5 = 0 at E ~ 1.65; [-1.5, 1.5] is
    # comfortably inside.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    assert (np.diff(ln_g) > 0).all()
    # First bin rebased to zero (matches stitch_entropy contract).
    assert ln_g[0] == pytest.approx(0.0, abs=1e-12)
    # Grid is uniform.
    assert np.allclose(np.diff(energies), 0.001)


def test_lattice_like_dos_phi_is_quartic_double_well_at_design_beta():
    # The fixture is built so that at beta = beta_c the canonical
    # phi(E) = beta_c * E - ln g(E) = a * (E**2 - c**2)**2 is a clean
    # quartic double-well: minima at E = +/- c with equal depth, a
    # maximum at E = 0 between them.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    beta_c = 10.0
    phi = beta_c * energies - ln_g
    i_zero = int(np.argmin(np.abs(energies)))
    i_pos = int(np.argmin(np.abs(energies - 1.0)))
    i_neg = int(np.argmin(np.abs(energies - (-1.0))))
    # Saddle higher than both wells.
    assert phi[i_zero] > phi[i_pos]
    assert phi[i_zero] > phi[i_neg]
    # Wells equally deep by construction.
    assert abs(phi[i_pos] - phi[i_neg]) < 0.01


def test_two_dominant_peak_indices_finds_two_minima():
    # phi has two minima at indices 2 and 6, with the deeper one at 6.
    phi = np.array([5.0, 3.0, 1.0, 3.0, 4.0, 3.0, 0.5, 2.0, 5.0])
    out = _two_dominant_peak_indices(phi)
    assert list(out) == [2, 6]


def test_two_dominant_peak_indices_ignores_shallow_third_minimum():
    # Three local minima at indices 1, 4, 7; the two deepest are
    # at 1 (phi=0) and 4 (phi=1); the third (index 7, phi=2) is
    # ignored.
    phi = np.array([5.0, 0.0, 5.0, 5.0, 1.0, 5.0, 5.0, 2.0, 5.0])
    out = _two_dominant_peak_indices(phi)
    assert list(out) == [1, 4]


def test_two_dominant_peak_indices_raises_on_fewer_than_two_minima():
    # Monotone phi has zero interior minima.
    phi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(NotBimodalError, match="local minima of phi"):
        _two_dominant_peak_indices(phi)


def test_find_phase_split_on_lattice_like_dos_at_design_beta():
    # Fixture parameters: a=1, beta_c=10, c=1.
    # At beta = beta_c = 10 eV^-1, phi = beta*E - ln_g is the
    # designed double-well a*(E**2 - c**2)**2 with minima at E = +/- c
    # = +/- 1 and a maximum at E = 0.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    T_K = 1.0 / (kB * 10.0)
    split = find_phase_split(dos, T_K=T_K)
    assert abs(split.E_peak_low - (-1.0)) < 0.002
    assert abs(split.E_peak_high - 1.0) < 0.002
    # E_star is the maximum of phi between the peaks at E = 0.
    assert abs(split.E_star) < 0.005


def test_find_phase_split_raises_outside_bimodal_window():
    # The bimodal-beta window has half-width
    # 8 * a * c**3 / (3 * sqrt(3)) ~ 1.54 for a=1, c=1.
    # Centred on beta_c = 10, so the window is (8.46, 11.54).
    # Outside this window, the linear term in phi has overtaken
    # one of the wells and phi has fewer than two minima.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    # Well above the window: beta = 20 eV^-1 -> T ~ 580 K.
    T_below = 1.0 / (kB * 20.0)
    with pytest.raises(NotBimodalError):
        find_phase_split(dos, T_K=T_below)
    # Well below the window: beta = 2 eV^-1 -> T ~ 5800 K.
    T_above = 1.0 / (kB * 2.0)
    with pytest.raises(NotBimodalError):
        find_phase_split(dos, T_K=T_above)
