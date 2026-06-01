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
    _walk_for_sign_change,
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
        weight_imbalance=1e-9, n_brentq_iterations=18,
        n_self_consistent_iter=0, self_consistent_converged=False,
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
        weight_imbalance=1e-9, n_brentq_iterations=18,
        n_self_consistent_iter=0, self_consistent_converged=False,
    )
    assert cp.T_K == 500.0
    # T_K is a read-only property; no separate field to assign.
    with pytest.raises(AttributeError):
        cp.T_K = 600.0  # type: ignore[misc]


def test_coexistence_point_has_self_consistent_fields():
    """The struct must expose self-consistency iteration counts and
    a convergence flag for downstream callers."""
    import dataclasses

    from mchammer_pt.analysis.coexistence import CoexistencePoint
    fields = {f.name for f in dataclasses.fields(CoexistencePoint)}
    assert "n_self_consistent_iter" in fields
    assert "self_consistent_converged" in fields


def test_coexistence_point_n_iterations_alias_deprecated():
    """``n_iterations`` is a backward-compat alias for
    ``n_brentq_iterations`` and emits DeprecationWarning naming the
    replacement attribute.
    """
    from mchammer_pt.analysis.coexistence import (
        CoexistencePoint,
        PhaseSplit,
    )
    pt = CoexistencePoint(
        split=PhaseSplit(
            E_peak_low=-1.0, E_peak_high=1.0, E_star=0.0, T_K=100.0,
        ),
        latent_heat=2.0,
        barrier_height=0.5,
        weight_imbalance=1e-6,
        n_brentq_iterations=7,
        n_self_consistent_iter=3,
        self_consistent_converged=True,
    )
    with pytest.warns(DeprecationWarning, match="n_brentq_iterations"):
        assert pt.n_iterations == 7


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


def test_equal_area_temperature_rejects_xtol_below_rtol_floor():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    with pytest.raises(ValueError, match="xtol"):
        equal_area_temperature(dos, xtol=1e-18)


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
    # real root of E**3 - E - 2.5 = 0 at E ~ 1.60; [-1.5, 1.5] is
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

    T_test = 1.0 / (10.0 * kB)  # roughly the design Tc
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
    T_test = 1.0 / (10.0 * kB)
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


def test_walk_for_sign_change_returns_sign_changing_pair():
    # From a T_start inside the bimodal window with a fixed E_star,
    # _walk_for_sign_change must return (T_lo, T_hi) with the
    # fixed-E_star imbalance of opposite sign (or one zero) at the
    # two endpoints.
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    T_seed = _cv_peak_seed(dos)
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    split = find_phase_split(dos, T_K=T_seed)
    T_lo, T_hi = _walk_for_sign_change(
        energies, ln_g, T_seed, split.E_star,
    )
    assert T_lo > 0.0
    assert T_hi > T_lo
    w_lo_lo, w_hi_lo = _partition_sums(
        energies, ln_g, T_K=T_lo, E_star=split.E_star,
    )
    w_lo_hi, w_hi_hi = _partition_sums(
        energies, ln_g, T_K=T_hi, E_star=split.E_star,
    )
    assert (w_lo_lo - w_hi_lo) * (w_lo_hi - w_hi_hi) <= 0


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
    # weight_imbalance is the normalised residual
    # |w_low - w_high| / (w_low + w_high) at the converged
    # (Tc, E_star). Default xtol=1e-4 narrows T to ~0.1 K; for this
    # fixture the imbalance(T) slope is steep, so the residual is
    # well below 1e-3.
    assert result.weight_imbalance < 1e-3
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
    # weight_imbalance is the normalised residual; see bracketed-case
    # test above for the rationale on the 1e-3 bound.
    assert result.weight_imbalance < 1e-3
    assert result.latent_heat > 0.0


def test_equal_area_temperature_raises_not_bimodal_on_unimodal_user_range():
    """A user-supplied ``T_bracket`` whose midpoint is far outside the
    bimodal window surfaces as ``NotBimodalError`` rather than
    ``NoBracketError``: saddle detection at the bracket midpoint runs
    before any sign-change walk, and the smoothed phi has only one peak
    there, so the bimodality check fails first."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )
    # T well above the bimodal window (beta ~ 1-2, far below beta_c):
    # phi is unimodal at the bracket midpoint, so the saddle-detection
    # seed step raises NotBimodalError.
    T_too_hot_lo = 1.0 / (kB * 2.0)
    T_too_hot_hi = 1.0 / (kB * 1.0)
    with pytest.raises(NotBimodalError, match="not bimodal"):
        equal_area_temperature(dos, T_bracket=(T_too_hot_lo, T_too_hot_hi))


def test_equal_area_temperature_smoothing_default_works_on_clean_dos():
    """Default smoothing_sigma=2.0 must converge to a Tc near the
    design value on a clean lattice_like_dos fixture."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    result = equal_area_temperature(dos)
    expected_Tc = 1.0 / (10.0 * kB)  # ~1160 K
    assert abs(result.T_K - expected_Tc) < 1.0  # within 1 K
    assert result.self_consistent_converged is True


def test_equal_area_temperature_tolerates_shot_noise_dimples():
    """A clean DOS with random gaussian dimples in the valley should
    still resolve to the same Tc within 1% of the dimple-free answer."""
    rng = np.random.default_rng(seed=42)
    dos_clean = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    dos_dimpled = dos_clean.copy()
    mid = len(dos_dimpled) // 2
    valley_slice = slice(mid - 30, mid + 30)
    entropy_col = dos_dimpled.columns.get_loc("entropy")
    dos_dimpled.iloc[valley_slice, entropy_col] += rng.normal(0, 0.3, size=60)

    res_clean = equal_area_temperature(dos_clean)
    res_dimpled = equal_area_temperature(dos_dimpled)
    assert abs(res_dimpled.T_K - res_clean.T_K) / res_clean.T_K < 0.01


def test_equal_area_temperature_self_consistent_iteration_logged():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    res = equal_area_temperature(dos)
    assert 1 <= res.n_self_consistent_iter <= 10, (
        f"iteration took {res.n_self_consistent_iter} passes; "
        f"expected <= 10 on a clean lattice_like_dos"
    )
    assert res.n_brentq_iterations >= 1
    assert res.self_consistent_converged is True


def test_equal_area_temperature_disabled_iteration():
    """max_self_consistent_iter=0 disables the iteration; result
    should still be in the same ballpark on clean DOS (the iteration
    refines E_star at Tc rather than provides order-of-magnitude
    correction)."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    res_iter = equal_area_temperature(dos)
    res_no_iter = equal_area_temperature(dos, max_self_consistent_iter=0)
    assert res_no_iter.n_self_consistent_iter == 0
    # Un-iterated answer uses E_star detected at the seed T, not at
    # Tc; on this symmetric fixture the seed/Tc mismatch produces a
    # few-K offset that the iteration removes.
    assert abs(res_iter.T_K - res_no_iter.T_K) < 5.0


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


def test_equal_area_temperature_reports_not_converged_when_iteration_truncated():
    """If the iteration exhausts ``max_self_consistent_iter`` without
    meeting ``self_consistent_tol_K``, the result reports
    ``self_consistent_converged=False``."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    # Tight tolerance + tiny budget guarantees truncation: a single
    # pass cannot drive |Δ Tc| below 1e-12 K.
    result = equal_area_temperature(
        dos,
        max_self_consistent_iter=1,
        self_consistent_tol_K=1e-12,
    )
    assert result.self_consistent_converged is False


def test_equal_area_temperature_weight_imbalance_is_normalised():
    """The reported ``weight_imbalance`` is the dimensionless
    normalised residual ``|w_low - w_high| / (w_low + w_high)``."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    result = equal_area_temperature(dos)
    assert 0.0 <= result.weight_imbalance < 1e-3, (
        f"normalised residual {result.weight_imbalance} unexpectedly "
        f"large on a clean lattice_like_dos; suggests Tc is not a "
        f"true root"
    )


def test_equal_area_temperature_final_brentq_pins_tc_to_real_root():
    """After convergence, the reported ``Tc`` must be a zero of
    ``imbalance(T; E_star)`` for the converged ``E_star`` — not the
    damped blend from the last iteration step. We recompute the
    normalised residual at the returned ``(Tc, split.E_star)`` and
    require it below ``xtol``."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    result = equal_area_temperature(dos)
    assert result.self_consistent_converged is True
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    w_lo, w_hi = _partition_sums(
        energies, ln_g, result.T_K, result.split.E_star,
    )
    assert abs(w_lo - w_hi) / (w_lo + w_hi) < 1e-4


def test_equal_area_temperature_non_default_damping_converges():
    """The damped iteration converges across a range of damping
    factors on a clean DOS. Pins the damping kwarg's behaviour at
    non-default values. ``damping=1.0`` (no damping) is excluded:
    undamped fixed-point iteration can oscillate on shallow-bimodal
    maps, which is precisely why the default damping is < 1."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    results = {}
    for damping in (0.25, 0.5, 0.75):
        results[damping] = equal_area_temperature(dos, damping=damping)
    # All converge.
    for damping, res in results.items():
        assert res.self_consistent_converged, (
            f"damping={damping} did not converge on clean DOS"
        )
    # All find roughly the same Tc (within 50 mK across damping
    # values on this symmetric fixture).
    Tcs = [r.T_K for r in results.values()]
    assert max(Tcs) - min(Tcs) < 0.05, (
        f"Tc varies > 50 mK across damping values: {Tcs}"
    )


def test_equal_area_temperature_tight_tol_takes_more_iterations():
    """Tighter self_consistent_tol_K requires more iteration passes
    to converge. Pins the tol_K kwarg behaviour."""
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    res_loose = equal_area_temperature(dos, self_consistent_tol_K=1e-1)
    res_tight = equal_area_temperature(dos, self_consistent_tol_K=1e-5)
    # Tighter tol -> more passes (or equal, never fewer).
    assert res_tight.n_self_consistent_iter >= res_loose.n_self_consistent_iter


def test_equal_area_temperature_validates_smoothing_sigma_negative():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    with pytest.raises(ValueError, match="smoothing_sigma"):
        equal_area_temperature(dos, smoothing_sigma=-1.0)


def test_equal_area_temperature_validates_damping_range():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    with pytest.raises(ValueError, match="damping"):
        equal_area_temperature(dos, damping=0.0)
    with pytest.raises(ValueError, match="damping"):
        equal_area_temperature(dos, damping=1.5)


def test_equal_area_temperature_validates_max_iter_non_negative():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    with pytest.raises(ValueError, match="max_self_consistent_iter"):
        equal_area_temperature(dos, max_self_consistent_iter=-1)


def test_equal_area_temperature_validates_tol_positive():
    dos = lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.005,
    )
    with pytest.raises(ValueError, match="self_consistent_tol_K"):
        equal_area_temperature(dos, self_consistent_tol_K=0.0)
    with pytest.raises(ValueError, match="self_consistent_tol_K"):
        equal_area_temperature(dos, self_consistent_tol_K=-1e-3)
