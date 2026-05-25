"""First-order coexistence-point analysis on stitched Wang-Landau DOS.

Locates the equal-area coexistence temperature T_c at which the
canonical energy distribution P(E|T) has equal integrated weight on
either side of the free-energy minimum E*(T) between its two peaks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ase.units import kB

from mchammer_pt.analysis.dos import reweight_canonical_from_dos

_AUTO_BRACKET_N_T = 50
_AUTO_BRACKET_KT_LOW_FRAC = 0.3
_AUTO_BRACKET_KT_HIGH_FRAC = 3.0
_AUTO_BRACKET_PAD_FACTOR = 1.5


class NotBimodalError(ValueError):
    """Raised when the DOS does not show two well-separated peaks.

    Carries a message naming the temperature at which detection ran
    and what shape was found (one peak / adjacent peaks / no interior
    minimum).
    """


class NoBracketError(ValueError):
    """Raised when bisection cannot proceed.

    Either ``imbalance(T)`` does not change sign across the supplied
    or auto-built ``T_bracket``, or shape analysis failed at one of
    the bracket endpoints (or mid-bracket — the bracket extends
    outside the bimodal region).
    """


def _parabolic_vertex(
    x_l: float, x_c: float, x_r: float,
    y_l: float, y_c: float, y_r: float,
) -> float:
    """Return the x-coordinate of the parabolic vertex through three samples.

    Fits ``y = a x^2 + b x + c`` to the three points by Lagrange
    interpolation and returns ``-b / (2 a)``. In typical use the three
    points are the centre bin of an extremum of ``phi(E)`` and its two
    neighbours on the DOS energy grid; the function returns a sub-bin
    refined position of the extremum.

    Falls back to ``x_c`` in two cases:

    - ``denom == 0``: two of the three x-values coincide. Cannot arise
      from distinct bin centres on a uniform grid; this branch is a
      defensive guard.
    - ``a == 0``: the three points are collinear (the local fit is
      linear, no parabolic vertex exists). Returning the centre sample
      is the natural no-op.
    """
    denom = (x_l - x_c) * (x_l - x_r) * (x_c - x_r)
    if denom == 0.0:
        return x_c
    a = (
        x_r * (y_c - y_l)
        + x_c * (y_l - y_r)
        + x_l * (y_r - y_c)
    ) / denom
    b = (
        x_r * x_r * (y_l - y_c)
        + x_c * x_c * (y_r - y_l)
        + x_l * x_l * (y_c - y_r)
    ) / denom
    if a == 0.0:
        return x_c
    return -b / (2.0 * a)


@dataclass(frozen=True)
class PhaseSplit:
    """The two phase peaks and the free-energy minimum between them.

    All energies are in eV. Peak and valley positions are refined to
    sub-bin precision by local-quadratic fits.
    """

    E_peak_low: float
    E_peak_high: float
    E_star: float
    T_K: float


def find_phase_split(
    dos: pd.DataFrame,
    T_K: float,
    *,
    min_peak_separation: int = 5,
) -> PhaseSplit:
    """Locate the two phase peaks and the dividing energy at T_K.

    Phase peak positions (``E_peak_low``, ``E_peak_high``) are the
    two dominant local maxima of ``ln g(E)`` — these are properties
    of the DOS and do not depend on temperature. The valley position
    ``E_star`` is the minimum of ``P(E | T_K)``, found as the maximum
    of ``phi(E) = beta * E - ln g(E)`` between the two peaks. All
    three positions are refined to sub-bin precision by three-point
    parabolic fits.

    Args:
        dos: DataFrame with ``energy`` (eV) and ``entropy`` (``ln g``)
            columns on a uniform grid.
        T_K: temperature in Kelvin at which ``E_star`` is evaluated.
        min_peak_separation: minimum number of bins required between
            the two phase peaks. Default 5.

    Returns:
        ``PhaseSplit`` with sub-bin-refined peak and valley positions
        and the supplied ``T_K``.

    Raises:
        ValueError: if ``dos`` is empty or ``T_K <= 0``.
        NotBimodalError: if fewer than two local maxima of ``ln g``
            are found, or the two largest are within
            ``min_peak_separation`` bins of each other.
    """
    if dos.empty:
        raise ValueError("dos has no rows; need at least one energy bin")
    if T_K <= 0.0:
        raise ValueError(f"T_K must be > 0 K; got {T_K}")
    if min_peak_separation < 1:
        raise ValueError(
            f"min_peak_separation must be >= 1; got {min_peak_separation}"
        )

    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    beta = 1.0 / (kB * T_K)
    phi = beta * energies - ln_g

    # Local maxima of ln_g give the DOS phase peaks (interior bins only).
    is_max = (ln_g[1:-1] > ln_g[:-2]) & (ln_g[1:-1] > ln_g[2:])
    maxima_idx = np.flatnonzero(is_max) + 1
    if maxima_idx.size < 2:
        raise NotBimodalError(
            f"find_phase_split: fewer than two local maxima of ln g at "
            f"T={T_K} K (found {maxima_idx.size})"
        )

    # Two largest-ln_g maxima: the two dominant DOS peaks.
    two_largest = maxima_idx[np.argsort(ln_g[maxima_idx])[-2:]]
    i_left, i_right = sorted(int(x) for x in two_largest)
    if i_right - i_left < min_peak_separation:
        raise NotBimodalError(
            f"find_phase_split: two largest DOS peaks at bin indices "
            f"{i_left} and {i_right} are within "
            f"{min_peak_separation} bins of each other at T={T_K} K"
        )

    # Sub-bin refinement of each DOS peak (maximise ln_g -> minimise -ln_g).
    E_peak_low = _parabolic_vertex(
        energies[i_left - 1], energies[i_left], energies[i_left + 1],
        -ln_g[i_left - 1], -ln_g[i_left], -ln_g[i_left + 1],
    )
    E_peak_high = _parabolic_vertex(
        energies[i_right - 1], energies[i_right], energies[i_right + 1],
        -ln_g[i_right - 1], -ln_g[i_right], -ln_g[i_right + 1],
    )

    # Maximum of phi between the two peak bin indices = minimum of P(E|T).
    interior = slice(i_left, i_right + 1)
    i_valley = int(np.argmax(phi[interior])) + i_left
    # Guard against the maximum sitting at an endpoint of the
    # interior slice (degenerate case; parabolic refinement still
    # works if we clip).
    j = min(max(i_valley, 1), len(phi) - 2)
    E_star = _parabolic_vertex(
        energies[j - 1], energies[j], energies[j + 1],
        phi[j - 1], phi[j], phi[j + 1],
    )

    return PhaseSplit(
        E_peak_low=float(E_peak_low),
        E_peak_high=float(E_peak_high),
        E_star=float(E_star),
        T_K=float(T_K),
    )


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
    pos = (E_star - left_edge_0) / energy_spacing  # in bin units
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

    ``log_w[i] = ln g[i] - beta * energies[i]``. The returned shifted
    array has its maximum subtracted off; the caller exponentiates
    and sums to recover an unnormalised partition function up to the
    overall factor ``exp(log_w_max)`` (which cancels in any ratio).
    """
    beta = 1.0 / (kB * T_K)
    log_w = ln_g - beta * energies
    log_w_max = float(log_w.max())
    return log_w - log_w_max, log_w_max


def _partition_sums(
    energies: np.ndarray, ln_g: np.ndarray, T_K: float, E_star: float,
) -> tuple[float, float]:
    """Count-weighted partition at ``E_star``: ``(w_low, w_high)``.

    Uses linear apportionment of the boundary bin (fraction ``f_low``
    goes to ``w_low``, ``1 - f_low`` to ``w_high``), so the partition
    is exact at sub-bin ``E_star`` and ``w_low + w_high`` equals the
    full sum exactly.

    Outputs are unnormalised but share a common scale, so ratios and
    differences are meaningful.
    """
    energy_spacing = float(energies[1] - energies[0])
    log_w, _ = _log_weights(energies, ln_g, T_K)
    w = np.exp(log_w)
    i_b, f_low = _boundary_fraction(energies, E_star, energy_spacing)
    w_low = float(w[:i_b].sum()) + f_low * float(w[i_b])
    w_high = (1.0 - f_low) * float(w[i_b]) + float(w[i_b + 1:].sum())
    return w_low, w_high


def _partition_means(
    energies: np.ndarray, ln_g: np.ndarray, T_K: float, E_star: float,
) -> tuple[float, float]:
    """Conditional means ``<E>_low``, ``<E>_high`` at ``E_star``.

    Uses the same fractional bin apportionment as ``_partition_sums``
    so the moments are consistent with the weights and the relation
    ``<E> = (<E>_low * w_low + <E>_high * w_high) / Z`` holds exactly.

    The boundary bin contributes its centre energy weighted by
    ``f_low * w_bin`` to the low-side moment and
    ``(1 - f_low) * w_bin`` to the high-side moment. This is the
    same apportionment as for the count partition; the bin centre's
    energy is used in both halves.
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


def _auto_bracket(dos: pd.DataFrame) -> tuple[float, float]:
    """Build a T-bracket for the equal-area bisection.

    Derives a kT-scale heuristic from the DOS energy and entropy
    ranges, scans Cv on a 50-point grid spanning
    ``[0.3, 3] * kT_scale / k_B``, then returns
    ``(T_centre - pad * w, T_centre + pad * w)`` where ``T_centre``
    is the parabolic-refined Cv peak and ``w`` is the half-max
    width of Cv around it. ``pad = 1.5``.
    """
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    E_range = float(energies.max() - energies.min())
    ln_g_range = float(ln_g.max() - ln_g.min())
    if ln_g_range == 0.0:
        raise ValueError(
            "auto_bracket: entropy range is zero; cannot derive a "
            "kT scale. Supply T_bracket explicitly."
        )
    kT_scale = E_range / ln_g_range  # eV
    T_lo_scan = _AUTO_BRACKET_KT_LOW_FRAC * kT_scale / kB
    T_hi_scan = _AUTO_BRACKET_KT_HIGH_FRAC * kT_scale / kB
    Ts = np.linspace(T_lo_scan, T_hi_scan, _AUTO_BRACKET_N_T)
    canonical = reweight_canonical_from_dos(dos, Ts)
    Cv = canonical["Cv"].to_numpy()
    i_peak = int(np.argmax(Cv))
    j = min(max(i_peak, 1), len(Cv) - 2)
    T_centre = _parabolic_vertex(
        Ts[j - 1], Ts[j], Ts[j + 1],
        -Cv[j - 1], -Cv[j], -Cv[j + 1],  # vertex of -Cv = peak of Cv
    )
    half_max = 0.5 * float(Cv[i_peak])
    above = np.flatnonzero(Cv >= half_max)
    if above.size == 0:
        # Degenerate: no resolvable peak. Fall back to the scan
        # bounds, padded.
        return T_lo_scan, T_hi_scan
    width = float(Ts[above[-1]] - Ts[above[0]])
    if width == 0.0:
        # Single-point peak: pad by one grid spacing.
        width = float(Ts[1] - Ts[0])
    half = 0.5 * _AUTO_BRACKET_PAD_FACTOR * width
    return float(T_centre - half), float(T_centre + half)
