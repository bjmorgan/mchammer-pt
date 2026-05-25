"""Unit tests for mchammer_pt.analysis.coexistence."""
from __future__ import annotations

import numpy as np

from mchammer_pt.analysis.coexistence import (
    NoBracketError,
    NotBimodalError,
    _parabolic_vertex,
)


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


from dataclasses import FrozenInstanceError

import pytest

from mchammer_pt.analysis.coexistence import (
    NotBimodalError,
    PhaseSplit,
    find_phase_split,
)
from tests._coexistence_fixtures import single_gaussian_dos, two_gaussian_dos


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
