"""Synthetic DOS fixtures used by the coexistence tests.

Each builder returns a ``(energy, entropy)`` DataFrame matching the
contract of ``mchammer_pt.analysis.dos.stitch_entropy``'s output: a
uniform energy grid in eV and ``entropy`` interpreted as ``ln g(E)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def two_gaussian_dos(
    *,
    E_low: float,
    E_high: float,
    sigma_low: float,
    sigma_high: float,
    weight_low: float,
    weight_high: float,
    E_min: float,
    E_max: float,
    energy_spacing: float,
) -> pd.DataFrame:
    """Build a DOS whose g(E) is a sum of two Gaussians.

    ``ln g(E) = ln(weight_low * exp(-(E - E_low)**2 / (2 * sigma_low**2))
                + weight_high * exp(-(E - E_high)**2 / (2 * sigma_high**2)))``

    Constants in front of the Gaussians (normalisations) are absorbed
    into the weight arguments. The returned DataFrame is rebased so
    that its ``entropy`` minimum is zero (matching
    ``stitch_entropy``'s convention).
    """
    n_bins = int(round((E_max - E_min) / energy_spacing)) + 1
    energies = E_min + np.arange(n_bins) * energy_spacing
    log_low = np.log(weight_low) - (energies - E_low) ** 2 / (2.0 * sigma_low ** 2)
    log_high = np.log(weight_high) - (energies - E_high) ** 2 / (2.0 * sigma_high ** 2)
    log_g = np.logaddexp(log_low, log_high)
    log_g -= log_g.min()
    return pd.DataFrame({"energy": energies, "entropy": log_g})


def single_gaussian_dos(
    *,
    E_centre: float,
    sigma: float,
    E_min: float,
    E_max: float,
    energy_spacing: float,
) -> pd.DataFrame:
    """Build a DOS with a single Gaussian g(E) (no bimodality)."""
    n_bins = int(round((E_max - E_min) / energy_spacing)) + 1
    energies = E_min + np.arange(n_bins) * energy_spacing
    log_g = -(energies - E_centre) ** 2 / (2.0 * sigma ** 2)
    log_g -= log_g.min()
    return pd.DataFrame({"energy": energies, "entropy": log_g})


def lattice_like_dos(
    *,
    A: float,
    B: float,
    w: float,
    E_c: float,
    E_min: float,
    E_max: float,
    energy_spacing: float,
) -> pd.DataFrame:
    """Build a DOS with monotone ln g and a slope kink at E_c.

    Models the canonical shape of a lattice-system DOS: ``ln g(E)``
    rises monotonically across its support, with the slope steepening
    locally around an energy ``E_c`` that represents the first-order
    coexistence region.

    Formula::

        ln g(E) = A * E + B * arctan((E - E_c) / w)

    so that::

        d(ln g)/dE = A + (B / w) / (1 + ((E - E_c) / w) ** 2)

    is strictly positive (monotone ``ln g``) when ``A > 0`` and varies
    smoothly from the asymptotic slope ``A`` far from ``E_c`` to a
    peak slope ``A + B / w`` at ``E_c``. For any inverse temperature
    ``beta`` with ``A < beta < A + B / w``, ``d(ln g)/dE = beta`` has
    two solutions — the two minima of ``phi(E) = beta * E - ln g(E)``
    and hence the two peaks of ``P(E | T)``. Analytically the phase
    peaks sit at::

        E = E_c +/- w * sqrt(B / (w * (beta - A)) - 1)

    The returned DataFrame has ``energy`` (eV) and ``entropy``
    (``ln g``) columns on a uniform grid spanning ``[E_min, E_max]``
    with spacing ``energy_spacing``. ``entropy`` is rebased so its
    minimum is zero (matching ``stitch_entropy``'s output
    convention).
    """
    n_bins = int(round((E_max - E_min) / energy_spacing)) + 1
    energies = E_min + np.arange(n_bins) * energy_spacing
    ln_g = A * energies + B * np.arctan((energies - E_c) / w)
    ln_g -= ln_g.min()
    return pd.DataFrame({"energy": energies, "entropy": ln_g})
