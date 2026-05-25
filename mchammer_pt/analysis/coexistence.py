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
