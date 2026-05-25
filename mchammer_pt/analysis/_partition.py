"""Count-weighted microstate partition on a stitched Wang-Landau DOS.

Used by :mod:`mchammer_pt.analysis.coexistence` to evaluate the
canonical partition function and its conditional moments split
around a sub-bin dividing energy ``E_star``. The boundary bin
(the one whose half-open interval contains ``E_star``) is
apportioned linearly so the partition is exact at sub-bin
``E_star`` and ``w_low + w_high`` equals the full sum exactly.

All sums are taken in log-space with max-subtraction stability,
so a large entropy range does not underflow ``float64``.
"""
from __future__ import annotations

import numpy as np
from ase.units import kB


def _boundary_fraction(
    energies: np.ndarray, E_star: float, energy_spacing: float,
) -> tuple[int, float]:
    """Locate the boundary bin and compute its low-side fraction.

    Returns ``(i_boundary, f_low)`` where ``i_boundary`` is the index
    of the bin whose half-open interval ``[E_i - dE/2, E_i + dE/2)``
    contains ``E_star`` (clipped to the grid range), and ``f_low``
    is the fraction of that bin lying on the low side of ``E_star``.

    Bins with index < ``i_boundary`` are wholly low; bins with
    index > ``i_boundary`` are wholly high.
    """
    left_edge_0 = energies[0] - 0.5 * energy_spacing
    pos = (E_star - left_edge_0) / energy_spacing
    i_boundary = int(np.floor(pos))
    i_boundary = max(0, min(i_boundary, len(energies) - 1))
    bin_left_edge = energies[i_boundary] - 0.5 * energy_spacing
    f_low = (E_star - bin_left_edge) / energy_spacing
    f_low = max(0.0, min(1.0, f_low))
    return i_boundary, f_low


def _log_weights(
    energies: np.ndarray, ln_g: np.ndarray, T_K: float,
) -> tuple[np.ndarray, float]:
    """Return ``(log_w_shifted, log_w_max)`` for stable summation.

    ``log_w[i] = ln g[i] - beta * energies[i]``. The returned array
    has its maximum subtracted off; the caller exponentiates and
    sums to recover an unnormalised partition function up to the
    overall factor ``exp(log_w_max)``.
    """
    beta = 1.0 / (kB * T_K)
    log_w = ln_g - beta * energies
    log_w_max = float(log_w.max())
    return log_w - log_w_max, log_w_max


def partition_sums(
    energies: np.ndarray, ln_g: np.ndarray, T_K: float, E_star: float,
) -> tuple[float, float]:
    """Count-weighted partition at ``E_star``: ``(w_low, w_high)``.

    Linear apportionment of the boundary bin (fraction ``f_low`` low,
    ``1 - f_low`` high) makes the partition exact at sub-bin
    ``E_star``; ``w_low + w_high`` equals the full sum.
    """
    energy_spacing = float(energies[1] - energies[0])
    log_w, _ = _log_weights(energies, ln_g, T_K)
    w = np.exp(log_w)
    i_b, f_low = _boundary_fraction(energies, E_star, energy_spacing)
    w_low = float(w[:i_b].sum()) + f_low * float(w[i_b])
    w_high = (1.0 - f_low) * float(w[i_b]) + float(w[i_b + 1:].sum())
    return w_low, w_high


def partition_means(
    energies: np.ndarray, ln_g: np.ndarray, T_K: float, E_star: float,
) -> tuple[float, float]:
    """Conditional means ``<E>_low``, ``<E>_high`` at ``E_star``.

    Uses the same fractional bin apportionment as
    :func:`partition_sums`, so the moments are consistent with the
    weights and ``<E> = (<E>_low * w_low + <E>_high * w_high) / Z``
    holds. The boundary bin contributes its centre energy to both
    halves, weighted by ``f_low * w_bin`` and ``(1 - f_low) * w_bin``
    respectively.
    """
    energy_spacing = float(energies[1] - energies[0])
    log_w, _ = _log_weights(energies, ln_g, T_K)
    w = np.exp(log_w)
    i_b, f_low = _boundary_fraction(energies, E_star, energy_spacing)
    w_low_full = float(w[:i_b].sum())
    w_high_full = float(w[i_b + 1:].sum())
    num_low = float((w[:i_b] * energies[:i_b]).sum()) + (
        f_low * float(w[i_b]) * float(energies[i_b])
    )
    num_high = (
        (1.0 - f_low) * float(w[i_b]) * float(energies[i_b])
        + float((w[i_b + 1:] * energies[i_b + 1:]).sum())
    )
    w_low = w_low_full + f_low * float(w[i_b])
    w_high = (1.0 - f_low) * float(w[i_b]) + w_high_full
    return num_low / w_low, num_high / w_high
