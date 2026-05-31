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
    _cv_peak_seed,
    _parabolic_vertex,
    _smooth_ln_g,
    _two_dominant_peak_indices,
    _walk_outward_bracket,
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
        weight_imbalance=1e-9, n_iterations=18,
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
        weight_imbalance=1e-9, n_iterations=18,
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


def test_find_phase_split_rejects_adjacent_minima():
    # Two phi minima at adjacent indices: separation = 1 < default
    # min_peak_separation = 5. find_phase_split should raise.
    # Build a tiny DOS where phi has minima at indices 1 and 3
    # (separation 2): pick energies and ln_g so that, at T_K=1 K,
    # phi = beta*E - ln_g has minima there.
    # Simpler: bypass T construction by using a custom DOS whose
    # ln_g shape directly produces the desired phi at beta = 1.
    # At beta=1, phi = E - ln_g. We want phi minima at E indices 1
    # and 3 of an array of length 5: pick energies = [0, 1, 2, 3, 4]
    # and ln_g such that phi = [5, 0, 5, 0, 5] (minima at 1 and 3).
    # ln_g = E - phi = [-5, 1, -3, 3, -1]. T_K = 1 / kB.
    energies = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ln_g = np.array([-5.0, 1.0, -3.0, 3.0, -1.0])
    dos = pd.DataFrame({"energy": energies, "entropy": ln_g})
    T_K = 1.0 / kB  # beta = 1
    with pytest.raises(NotBimodalError, match="within"):
        find_phase_split(dos, T_K=T_K, min_peak_separation=5)


def test_find_phase_split_accepts_small_min_peak_separation():
    # Same DOS, but explicitly lower the separation requirement so
    # the helper accepts the two adjacent minima.
    energies = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ln_g = np.array([-5.0, 1.0, -3.0, 3.0, -1.0])
    dos = pd.DataFrame({"energy": energies, "entropy": ln_g})
    T_K = 1.0 / kB
    split = find_phase_split(dos, T_K=T_K, min_peak_separation=1)
    # Peaks fall between energies 0 and 4 (the two minima of phi
    # are at energy indices 1 and 3).
    assert split.E_peak_low < split.E_peak_high


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


def test_find_phase_split_sub_bin_refinement_on_lattice_like_dos():
    # Choose c slightly off a bin centre so the analytic peak
    # positions (E = +/- c = +/- 1.0007) fall between bins of the
    # 0.001-spaced grid. Parabolic refinement must recover the
    # sub-bin positions to better than energy_spacing / 10.
    c = 1.0007
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=c,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    T_K = 1.0 / (kB * 10.0)
    split = find_phase_split(dos, T_K=T_K)
    assert abs(split.E_peak_low - (-c)) < 1e-4
    assert abs(split.E_peak_high - c) < 1e-4


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


def test_find_phase_split_with_smoothing_ignores_narrow_dimples():
    """A narrow shot-noise-scale dimple in ``ln g`` between the two
    phase peaks shifts the saddle ``E_star`` when computed from raw
    ``phi``. ``smoothing_sigma > 0`` must recover ``E_star`` closer
    to the dimple-free reference than the raw-on-dimpled computation
    does.

    Mirrors the failure mode the kwarg targets: bin-scale noise in
    ``ln g`` defeats the local-minima detector. Larger structural
    perturbations are out of scope — the kwarg's docstring scope is
    "bin-scale shot-noise dimples".
    """
    dos_clean = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    dos_dimpled = dos_clean.copy()
    mid_idx = len(dos_dimpled) // 2
    # Shot-noise-scale dimple at one bin near the saddle.
    dos_dimpled.loc[mid_idx, "entropy"] += 0.001

    T_test = 1.0 / (10.0 * 8.617e-5)  # roughly the design Tc
    bin_width = float(dos_clean.loc[1, "energy"] - dos_clean.loc[0, "energy"])

    split_clean = find_phase_split(dos_clean, T_K=T_test)
    split_dimpled = find_phase_split(dos_dimpled, T_K=T_test)
    split_smoothed = find_phase_split(
        dos_dimpled, T_K=T_test, smoothing_sigma=2.0,
    )

    # Premise: even a single-bin shot-noise dimple shifts E_star
    # by more than a bin width. Pins the bug.
    raw_error = abs(split_dimpled.E_star - split_clean.E_star)
    assert raw_error > bin_width, (
        f"dimple did not perturb E_star ({raw_error=:.6f}, "
        f"{bin_width=:.6f}); fixture no longer exercises the kwarg"
    )

    # Fix: smoothing brings E_star closer to the clean reference
    # than the un-smoothed dimpled computation does.
    smoothed_error = abs(split_smoothed.E_star - split_clean.E_star)
    assert smoothed_error < raw_error, (
        f"smoothing did not improve E_star: {smoothed_error=:.6f}, "
        f"{raw_error=:.6f}"
    )


def test_find_phase_split_zero_sigma_unchanged():
    """smoothing_sigma=0 must reproduce the existing behaviour exactly."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    T_test = 1.0 / (10.0 * 8.617e-5)
    s0 = find_phase_split(dos, T_K=T_test, smoothing_sigma=0.0)
    s_default = find_phase_split(dos, T_K=T_test)  # default sigma should be 0
    assert s0.E_peak_low == s_default.E_peak_low
    assert s0.E_peak_high == s_default.E_peak_high
    assert s0.E_star == s_default.E_star


def test_cv_peak_seed_lands_inside_bimodal_window():
    # The Cv peak must sit inside the bimodal-P(E|T) window — that's
    # the entire physical justification for using it as a bracket
    # seed. Fixture's bimodal-T window is ~(1006, 1372) K at
    # beta_c=10, c=1; the Cv peak should be somewhere inside it.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    T_seed = _cv_peak_seed(dos)
    # Bimodal-window verification: find_phase_split must succeed at
    # T_seed.
    split = find_phase_split(dos, T_K=T_seed)
    assert split.E_peak_low < split.E_peak_high
    # And T_seed must lie inside the analytic bimodal window.
    T_bimodal_lo = 1.0 / (kB * 11.54)
    T_bimodal_hi = 1.0 / (kB * 8.46)
    assert T_bimodal_lo < T_seed < T_bimodal_hi


def test_cv_peak_seed_raises_value_error_on_flat_ln_g():
    # A DOS with constant ln g has no slope to derive kT_scale from.
    dos = pd.DataFrame({
        "energy": np.linspace(-1.0, 1.0, 21),
        "entropy": np.zeros(21),
    })
    with pytest.raises(ValueError, match="kT scale"):
        _cv_peak_seed(dos)


def test_walk_outward_bracket_returns_sign_changing_pair():
    # From a seed inside the bimodal window, walk_outward_bracket
    # must return (T_lo, T_hi) with imbalance(T_lo) and
    # imbalance(T_hi) of opposite sign (or one zero). Both endpoints
    # must still yield a valid PhaseSplit.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    T_seed = _cv_peak_seed(dos)
    T_lo, T_hi = _walk_outward_bracket(dos, T_seed)
    assert T_lo > 0.0
    assert T_hi > T_lo
    split_lo = find_phase_split(dos, T_K=T_lo)
    split_hi = find_phase_split(dos, T_K=T_hi)
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    w_lo_lo, w_hi_lo = _partition_sums(
        energies, ln_g, T_K=T_lo, E_star=split_lo.E_star,
    )
    w_lo_hi, w_hi_hi = _partition_sums(
        energies, ln_g, T_K=T_hi, E_star=split_hi.E_star,
    )
    assert (w_lo_lo - w_hi_lo) * (w_lo_hi - w_hi_hi) <= 0


def test_walk_outward_bracket_raises_not_bimodal_at_bad_seed():
    # A seed outside the bimodal window must surface as
    # NotBimodalError immediately, not as a wandering walk.
    energies = np.linspace(-2.0, 2.0, 401)
    ln_g = -(energies ** 2)
    ln_g -= ln_g.min()
    dos = pd.DataFrame({"energy": energies, "entropy": ln_g})
    # T = 100 K is well inside the kT-scale range for this DOS but
    # P(E|T) is unimodal, so find_phase_split must fail.
    with pytest.raises(NotBimodalError):
        _walk_outward_bracket(dos, T_seed=100.0)


def test_equal_area_temperature_raises_not_bimodal_on_unimodal_dos():
    # Single-bump ln g: no T anywhere yields bimodal P(E|T). The
    # auto-bracket path computes a Cv peak, then verifies bimodality
    # there — the seed verification should raise NotBimodalError.
    energies = np.linspace(-2.0, 2.0, 401)
    ln_g = -(energies ** 2)
    ln_g -= ln_g.min()
    dos = pd.DataFrame({"energy": energies, "entropy": ln_g})
    with pytest.raises(NotBimodalError, match="bimodal"):
        equal_area_temperature(dos)


def test_equal_area_temperature_on_lattice_like_dos():
    # Fixture: a=1, beta_c=10, c=1. Bimodal-beta window
    # (beta_c +/- 8 a c^3 / (3 sqrt(3))) ~ (8.46, 11.54),
    # T window ~ (1006, 1372) K. By symmetry of the fixture
    # (E_low = -c, E_high = +c, equal phi-well depths at beta_c)
    # the equal-area Tc is exactly beta_c -> T = 1 / (kB * 10).
    # The half-width 8/(3 sqrt(3)) = 1.5396... rounds to 1.54, so the
    # rounded edges 8.46 and 11.54 sit on/just outside the strict
    # bimodal window; we inset to 8.5 / 11.5 (still tight on Tc=10)
    # to guarantee both endpoints yield a valid PhaseSplit.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    T_bimodal_lo = 1.0 / (kB * 11.5)
    T_bimodal_hi = 1.0 / (kB * 8.5)
    T_c_analytic = 1.0 / (kB * 10.0)
    result = equal_area_temperature(
        dos, T_bracket=(T_bimodal_lo, T_bimodal_hi),
    )
    assert T_bimodal_lo < result.T_K < T_bimodal_hi
    assert abs(result.T_K - T_c_analytic) / T_c_analytic < 0.01
    # weight_imbalance is in the same (unnormalised) units as
    # partition_sums, so we check it relative to the total partition
    # weight at the converged Tc. Default xtol=1e-4 narrows T to a
    # window ~0.1 K wide; for this fixture the imbalance(T) slope is
    # steep, giving an absolute residual ~0.3 against a total ~1930
    # (relative ~1.4e-4).
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    w_low, w_high = _partition_sums(
        energies, ln_g, result.T_K, result.split.E_star,
    )
    assert result.weight_imbalance / (w_low + w_high) < 1e-3
    assert result.latent_heat > 0.0
    # Returned PhaseSplit must be self-consistent: re-running
    # find_phase_split at result.T_K must reproduce the same split.
    re_split = find_phase_split(dos, T_K=result.T_K)
    assert abs(re_split.E_star - result.split.E_star) < 1e-6
    # At beta = beta_c with a=c=1, phi is the designed quartic
    # double-well with wells at phi=0 (after rebasing) and saddle
    # at phi=a*c^4=1. The barrier height (in eV) is kB * Tc times
    # this phi gap: barrier_height = kB * Tc * 1.0. With Tc =
    # 1/(kB * 10), that simplifies to 0.1 eV exactly.
    assert abs(result.barrier_height - 0.1) < 5e-4


def test_equal_area_temperature_auto_bracket_on_lattice_like_dos():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    result = equal_area_temperature(dos)  # no T_bracket -> auto
    T_bimodal_lo = 1.0 / (kB * 11.54)
    T_bimodal_hi = 1.0 / (kB * 8.46)
    assert T_bimodal_lo < result.T_K < T_bimodal_hi
    # See bracketed-case note above on weight_imbalance vs. xtol.
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    w_low, w_high = _partition_sums(
        energies, ln_g, result.T_K, result.split.E_star,
    )
    assert result.weight_imbalance / (w_low + w_high) < 1e-3
    assert result.latent_heat > 0.0


def test_equal_area_temperature_raises_no_bracket_on_bad_user_range():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    # T well above the bimodal window (beta ~ 1-2, far below beta_c):
    # P is unimodal, find_phase_split fails at the bracket endpoint,
    # surfaced as NoBracketError.
    T_too_hot_lo = 1.0 / (kB * 2.0)
    T_too_hot_hi = 1.0 / (kB * 1.0)
    with pytest.raises(NoBracketError):
        equal_area_temperature(dos, T_bracket=(T_too_hot_lo, T_too_hot_hi))


def test_smooth_ln_g_zero_sigma_returns_input():
    ln_g = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    out = _smooth_ln_g(ln_g, sigma=0.0)
    np.testing.assert_array_equal(out, ln_g)


def test_smooth_ln_g_positive_sigma_smooths():
    ln_g = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    out = _smooth_ln_g(ln_g, sigma=1.0)
    # Gaussian filter preserves endpoints approximately under mode='nearest'
    # and brings interior values toward neighbouring mean.
    assert out.shape == ln_g.shape
    # Interior values shift toward neighbours but stay bounded.
    assert 0.5 < out[1] < 2.0
    # Endpoints remain close to input under nearest-value extrapolation
    # at the boundary.
    assert abs(out[0] - ln_g[0]) < 1.0


def test_smooth_ln_g_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        _smooth_ln_g(np.array([0.0, 1.0]), sigma=-0.5)
