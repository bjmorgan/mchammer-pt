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

from mchammer_pt.analysis._partition import partition_means, partition_sums

_AUTO_BRACKET_N_T = 60
_AUTO_BRACKET_KT_LOW_FRAC = 0.05
_AUTO_BRACKET_KT_HIGH_FRAC = 20.0


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

    Fits ``y = a x^2 + b x + c`` by Lagrange interpolation and returns
    ``-b / (2 a)``. Typical use: three points centred on an extremum
    bin of ``phi(E)`` on the DOS energy grid; the return is the sub-bin
    refined extremum position.

    Returns ``x_c`` when the parabola degenerates: ``a == 0``
    (collinear samples) or coincident x-values.
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


def _parabolic_value_at(
    x_l: float, x_c: float, x_r: float,
    y_l: float, y_c: float, y_r: float,
    x: float,
) -> float:
    """Evaluate the Lagrange parabola through three samples at ``x``.

    Used to read sub-bin values of a quantity sampled on the bin grid
    (e.g. ``phi`` at the sub-bin peak and valley positions returned
    by ``find_phase_split``). Returns ``y_c`` if the three x-values
    are coincident.
    """
    denom = (x_l - x_c) * (x_l - x_r) * (x_c - x_r)
    if denom == 0.0:
        return y_c
    w_l = (x - x_c) * (x - x_r) / ((x_l - x_c) * (x_l - x_r))
    w_c = (x - x_l) * (x - x_r) / ((x_c - x_l) * (x_c - x_r))
    w_r = (x - x_l) * (x - x_c) / ((x_r - x_l) * (x_r - x_c))
    return w_l * y_l + w_c * y_c + w_r * y_r


@dataclass(frozen=True)
class PhaseSplit:
    """The two phase peaks and the free-energy minimum between them.

    Attributes:
        E_peak_low: low-energy phase peak position in eV, sub-bin
            refined from the two largest local maxima of ``ln g``.
        E_peak_high: high-energy phase peak position in eV, sub-bin
            refined.
        E_star: dividing energy in eV — the maximum of
            ``phi(E) = beta * E - ln g(E)`` between the two peaks at
            ``T_K``, sub-bin refined.
        T_K: temperature in Kelvin at which ``E_star`` was located.
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

    try:
        peak_idx = _two_dominant_peak_indices(ln_g)
    except NotBimodalError as exc:
        raise NotBimodalError(
            f"find_phase_split at T={T_K} K: {exc}"
        ) from exc
    i_left, i_right = int(peak_idx[0]), int(peak_idx[1])
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

    interior = slice(i_left, i_right + 1)
    i_valley = int(np.argmax(phi[interior])) + i_left
    if i_valley == i_left or i_valley == i_right:
        raise NotBimodalError(
            f"find_phase_split: no interior maximum of phi between the "
            f"two DOS peaks at T={T_K} K (phi monotonic across the "
            f"inter-peak range; canonical distribution lacks a saddle)"
        )
    E_star = _parabolic_vertex(
        energies[i_valley - 1], energies[i_valley], energies[i_valley + 1],
        phi[i_valley - 1], phi[i_valley], phi[i_valley + 1],
    )

    return PhaseSplit(
        E_peak_low=float(E_peak_low),
        E_peak_high=float(E_peak_high),
        E_star=float(E_star),
        T_K=float(T_K),
    )


def _two_dominant_peak_indices(ln_g: np.ndarray) -> np.ndarray:
    """Return the bin indices of the two largest local maxima of ln_g.

    Local maxima are interior bins strictly higher than both
    neighbours. The two with the largest ``ln_g`` values are returned
    in ascending bin-index order.

    Raises:
        NotBimodalError: if fewer than two local maxima exist.
    """
    is_max = (ln_g[1:-1] > ln_g[:-2]) & (ln_g[1:-1] > ln_g[2:])
    maxima_idx = np.flatnonzero(is_max) + 1
    if maxima_idx.size < 2:
        raise NotBimodalError(
            f"fewer than two local maxima of ln g "
            f"(found {maxima_idx.size})"
        )
    two_largest = maxima_idx[np.argsort(ln_g[maxima_idx])[-2:]]
    return np.array(sorted(int(x) for x in two_largest), dtype=np.int64)


def _auto_bracket(
    dos: pd.DataFrame,
    *,
    min_peak_separation: int = 5,
) -> tuple[float, float]:
    """Build a T-bracket for the equal-area bisection.

    Derives a kT-scale heuristic from the energy separation and entropy
    difference of the two DOS peaks, then scans
    ``imbalance(T) = w_low(T) - w_high(T)`` on a log-spaced grid
    spanning
    ``[_AUTO_BRACKET_KT_LOW_FRAC, _AUTO_BRACKET_KT_HIGH_FRAC] * kT_scale / k_B``.
    Returns ``(T_lo, T_hi)`` as the first adjacent pair in the scan
    where imbalance changes sign.

    The kT-scale is derived as ``ΔE_peaks / |Δln g_peaks|`` where
    ``ΔE_peaks`` is the energy separation between the two dominant DOS
    peaks and ``|Δln g_peaks|`` is their entropy difference. This
    ratio is the leading-order estimate of T_c for a bimodal DOS.

    Raises:
        NotBimodalError: if the DOS has fewer than two local maxima of
            ``ln g``.
        ValueError: if the entropy difference between the two peaks is
            zero (symmetric DOS, T_c → ∞).
        NoBracketError: if no sign change in ``imbalance(T)`` is found
            across the scan grid.
    """
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()

    try:
        peak_idx = _two_dominant_peak_indices(ln_g)
    except NotBimodalError as exc:
        raise NotBimodalError(
            f"auto_bracket: {exc}; cannot derive a kT scale. "
            "Supply T_bracket explicitly."
        ) from exc
    i_left, i_right = int(peak_idx[0]), int(peak_idx[1])
    E_peak_sep = float(energies[i_right] - energies[i_left])
    ln_g_diff = abs(float(ln_g[i_right] - ln_g[i_left]))
    if ln_g_diff < 1e-10:
        raise ValueError(
            "auto_bracket: two DOS peaks have equal entropy "
            f"(ln g diff = {ln_g_diff:.2g}); T_c → ∞ for a symmetric DOS. "
            "Supply T_bracket explicitly."
        )
    kT_scale = E_peak_sep / ln_g_diff  # eV
    T_lo_scan = max(_AUTO_BRACKET_KT_LOW_FRAC * kT_scale / kB, 1.0)
    T_hi_scan = _AUTO_BRACKET_KT_HIGH_FRAC * kT_scale / kB
    Ts = np.logspace(
        np.log10(T_lo_scan), np.log10(T_hi_scan), _AUTO_BRACKET_N_T,
    )

    prev_T: float | None = None
    prev_f: float | None = None
    n_valid = 0
    n_compared = 0
    for T in Ts:
        try:
            split = find_phase_split(
                dos, T_K=float(T), min_peak_separation=min_peak_separation,
            )
        except NotBimodalError:
            prev_T = None
            prev_f = None
            continue
        w_low, w_high = partition_sums(energies, ln_g, float(T), split.E_star)
        f = w_low - w_high
        n_valid += 1
        if prev_f is not None and prev_T is not None:
            n_compared += 1
            if prev_f * f <= 0.0:
                return float(prev_T), float(T)
        prev_T = float(T)
        prev_f = f

    if n_valid == 0:
        raise NotBimodalError(
            "auto_bracket: shape analysis failed at every scan T in "
            f"[{T_lo_scan:.1f}, {T_hi_scan:.1f}] K. The canonical "
            "distribution P(E|T) appears not to be bimodal anywhere "
            "in the scan range. Supply T_bracket explicitly if you "
            "believe a coexistence region exists outside it."
        )
    if n_compared == 0:
        raise NoBracketError(
            f"auto_bracket: only isolated bimodal scan points in "
            f"[{T_lo_scan:.1f}, {T_hi_scan:.1f}] K (no two adjacent "
            f"scan Ts were both bimodal; n_valid={n_valid}). The "
            "bimodal-P(E|T) range is narrower than the scan grid "
            "resolution. Supply T_bracket explicitly to bisect inside "
            "the bimodal region."
        )
    raise NoBracketError(
        "auto_bracket: imbalance(T) did not change sign across the "
        f"scan [{T_lo_scan:.1f}, {T_hi_scan:.1f}] K "
        f"(kT_scale = {kT_scale:.4g} eV). "
        "Supply T_bracket explicitly."
    )


@dataclass(frozen=True)
class CoexistencePoint:
    """First-order coexistence point obtained from a stitched DOS.

    Returned by :func:`equal_area_temperature`. Bundles the phase
    split at coexistence with the diagnostics that only make sense
    there (latent heat, barrier height).

    Attributes:
        split: :class:`PhaseSplit` at the coexistence temperature;
            ``split.T_K`` is the equal-area temperature in Kelvin and
            ``split.E_star`` is ``E*(split.T_K)``.
        latent_heat: ``<E>_high - <E>_low`` at coexistence, in eV.
            Positive by construction.
        barrier_height: free-energy barrier height at coexistence,
            in eV. The negative log-ratio of the saddle
            ``P(E_star | T_c)`` to the larger of the two phase-peak
            heights, multiplied by ``k_B * T_c``. Non-negative for a
            genuinely bimodal DOS.
        weight_imbalance: ``|w_low - w_high|`` at the returned
            temperature, the bisection residual.
        n_bisection_steps: number of bisection iterations executed.

    The coexistence temperature is exposed as the read-only
    ``T_K`` property, delegating to ``split.T_K``.
    """

    split: PhaseSplit
    latent_heat: float
    barrier_height: float
    weight_imbalance: float
    n_bisection_steps: int

    @property
    def T_K(self) -> float:
        """The equal-area coexistence temperature, in Kelvin."""
        return self.split.T_K


def equal_area_temperature(
    dos: pd.DataFrame,
    *,
    T_bracket: tuple[float, float] | None = None,
    xtol: float = 1e-4,
    min_peak_separation: int = 5,
) -> CoexistencePoint:
    """Equal-area coexistence temperature from a stitched DOS.

    Performs a single 1D bisection on temperature where, at each
    trial T, ``find_phase_split`` is called to locate the dividing
    energy ``E*(T)`` and ``imbalance(T) = w_low(T) - w_high(T)`` is
    computed from the count-weighted microstate partition at that
    ``E*``. The bisection terminates when the bracket has shrunk
    below ``xtol * T_mid``.

    Args:
        dos: stitched DOS as produced by
            ``mchammer_pt.analysis.dos.stitch_entropy``.
        T_bracket: ``(T_lo, T_hi)`` bracket in Kelvin. If ``None``,
            built automatically from a kT-scale heuristic derived
            from the energy and entropy difference of the two
            dominant DOS peaks; the heuristic scans ``imbalance(T)``
            on a log-spaced grid to find the first sign change.
        xtol: relative bisection tolerance on T. Default 1e-4.
        min_peak_separation: forwarded to
            :func:`find_phase_split`.

    Returns:
        A :class:`CoexistencePoint`.

    Raises:
        ValueError: on invalid inputs.
        NotBimodalError: if no T in the auto-built scan range
            yields a bimodal ``P(E|T)`` (only when ``T_bracket`` is
            ``None``).
        NoBracketError: if ``imbalance(T)`` does not change sign
            across the bracket, or if shape analysis fails at any
            bracket endpoint or mid-bracket trial.
    """
    if dos.empty:
        raise ValueError("dos has no rows; need at least one energy bin")
    if xtol <= 0.0:
        raise ValueError(f"xtol must be > 0; got {xtol}")
    if min_peak_separation < 1:
        raise ValueError(
            f"min_peak_separation must be >= 1; got {min_peak_separation}"
        )

    if T_bracket is None:
        T_lo, T_hi = _auto_bracket(dos, min_peak_separation=min_peak_separation)
    else:
        T_lo, T_hi = T_bracket
        if T_lo <= 0.0 or T_hi <= 0.0:
            raise ValueError(
                f"T_bracket entries must be > 0 K; got ({T_lo}, {T_hi})"
            )
        if T_lo >= T_hi:
            raise ValueError(
                f"T_bracket must satisfy T_lo < T_hi; got ({T_lo}, {T_hi})"
            )

    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()

    def imbalance(T: float) -> float:
        split = find_phase_split(
            dos, T_K=T, min_peak_separation=min_peak_separation,
        )
        w_low, w_high = partition_sums(energies, ln_g, T, split.E_star)
        return w_low - w_high

    try:
        f_lo = imbalance(T_lo)
        f_hi = imbalance(T_hi)
    except NotBimodalError as exc:
        raise NoBracketError(
            f"shape analysis failed at a bracket endpoint "
            f"(T_lo={T_lo}, T_hi={T_hi}): {exc}"
        ) from exc
    if f_lo * f_hi > 0.0:
        raise NoBracketError(
            f"imbalance has same sign at both endpoints "
            f"(T_lo={T_lo}: {f_lo:.3g}, T_hi={T_hi}: {f_hi:.3g}); "
            f"extend T_bracket"
        )

    n_steps = 0
    while True:
        T_mid = 0.5 * (T_lo + T_hi)
        try:
            f_mid = imbalance(T_mid)
        except NotBimodalError as exc:
            raise NoBracketError(
                f"shape analysis failed at mid-bracket T={T_mid}; the "
                f"bracket extends outside the bimodal region: {exc}"
            ) from exc
        n_steps += 1
        # Move the endpoint that shares a sign with f_mid. The
        # sign-product test handles f_lo == 0 as a sign change.
        if f_lo * f_mid < 0.0:
            T_hi = T_mid
            f_hi = f_mid
        else:
            T_lo = T_mid
            f_lo = f_mid
        if (T_hi - T_lo) < xtol * T_mid:
            T_c = T_mid
            break

    final_split = find_phase_split(
        dos, T_K=T_c, min_peak_separation=min_peak_separation,
    )
    mean_low, mean_high = partition_means(
        energies, ln_g, T_c, final_split.E_star,
    )
    latent_heat = mean_high - mean_low

    # barrier_height = k_B * T_c * (phi(E_star) - phi_peak_min), from
    # the log-space identity ln[P(E_star) / max(P_low_peak, P_high_peak)]
    # = phi_peak_min - phi(E_star) (since P = exp(-phi)). phi is
    # sampled by three-point parabolic interpolation at the sub-bin
    # energies returned by find_phase_split.
    beta_c = 1.0 / (kB * T_c)
    phi = beta_c * energies - ln_g
    energy_spacing = float(energies[1] - energies[0])

    def phi_at(E: float) -> float:
        i = int(round((E - energies[0]) / energy_spacing))
        i = max(1, min(i, len(energies) - 2))
        return float(_parabolic_value_at(
            energies[i - 1], energies[i], energies[i + 1],
            phi[i - 1], phi[i], phi[i + 1],
            E,
        ))

    phi_peak_min = min(
        phi_at(final_split.E_peak_low),
        phi_at(final_split.E_peak_high),
    )
    barrier_height = float(kB * T_c * (phi_at(final_split.E_star) - phi_peak_min))

    return CoexistencePoint(
        split=final_split,
        latent_heat=float(latent_heat),
        barrier_height=barrier_height,
        weight_imbalance=float(abs(f_mid)),
        n_bisection_steps=n_steps,
    )
