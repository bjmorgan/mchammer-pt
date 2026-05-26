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
