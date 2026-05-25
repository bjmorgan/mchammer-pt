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

    # Locate the two dominant DOS peaks to derive kT_scale.
    is_max = (ln_g[1:-1] > ln_g[:-2]) & (ln_g[1:-1] > ln_g[2:])
    maxima_idx = np.flatnonzero(is_max) + 1
    if maxima_idx.size < 2:
        raise NotBimodalError(
            "auto_bracket: fewer than two local maxima of ln g; cannot "
            "derive a kT scale. Supply T_bracket explicitly."
        )
    two_largest = maxima_idx[np.argsort(ln_g[maxima_idx])[-2:]]
    i_left, i_right = sorted(int(x) for x in two_largest)
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
    for T in Ts:
        try:
            split = find_phase_split(
                dos, T_K=float(T), min_peak_separation=min_peak_separation,
            )
        except NotBimodalError:
            prev_T = None
            prev_f = None
            continue
        w_low, w_high = _partition_sums(energies, ln_g, float(T), split.E_star)
        f = w_low - w_high
        n_valid += 1
        if prev_f is not None and prev_T is not None:
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
    raise NoBracketError(
        "auto_bracket: imbalance(T) did not change sign across the scan "
        f"[{T_lo_scan:.1f}, {T_hi_scan:.1f}] K "
        f"(kT_scale = {kT_scale:.4g} eV). "
        "Supply T_bracket explicitly."
    )


@dataclass(frozen=True)
class CoexistencePoint:
    """First-order coexistence point obtained from a stitched DOS.

    Returned by :func:`equal_area_temperature`. Bundles the
    equal-area temperature and the diagnostics that only make sense
    at coexistence (latent heat, barrier height).

    Attributes:
        T_K: equal-area coexistence temperature, in Kelvin.
        split: :class:`PhaseSplit` at ``T_K``; ``split.E_star`` is
            ``E*(T_K)``.
        latent_heat: ``<E>_high - <E>_low`` at ``T_K``, in eV. Positive
            by construction.
        barrier_height: free-energy barrier height at ``T_K``, in eV.
            The negative log-ratio of the saddle ``P(E_star | T_K)``
            to the larger of the two phase-peak heights, multiplied
            by ``k_B * T_K``. Non-negative for a genuinely bimodal DOS.
        weight_imbalance: ``|w_low - w_high|`` at the returned ``T_K``,
            the bisection residual. Dimensionless (unnormalised
            partition weights share a common scale).
        n_bisection_steps: number of bisection iterations executed.
    """

    T_K: float
    split: PhaseSplit
    latent_heat: float
    barrier_height: float
    weight_imbalance: float
    n_bisection_steps: int


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
    ``E*``. Converges when ``|T_new - T_old| < xtol * T_new``.

    Args:
        dos: stitched DOS as produced by
            ``mchammer_pt.analysis.dos.stitch_entropy``.
        T_bracket: ``(T_lo, T_hi)`` bracket in Kelvin. If ``None``,
            built by :func:`_auto_bracket`, which derives a kT scale
            from the energy and entropy differences of the two
            dominant DOS peaks and locates an adjacent pair of trial
            temperatures across which ``imbalance(T)`` changes sign.
        xtol: relative bisection tolerance on T. Default 1e-4.
        min_peak_separation: forwarded to
            :func:`find_phase_split`.

    Returns:
        A :class:`CoexistencePoint`.

    Raises:
        ValueError: on invalid inputs.
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
        w_low, w_high = _partition_sums(energies, ln_g, T, split.E_star)
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
        # sign-product form `f_lo * f_mid < 0` is the standard
        # bisection idiom; it handles f_lo == 0 correctly (treated
        # as a sign change to the right) where the explicit-sign
        # form would always fall through to the `else` branch.
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
    mean_low, mean_high = _partition_means(
        energies, ln_g, T_c, final_split.E_star,
    )
    latent_heat = mean_high - mean_low

    # barrier_height in eV. Using log-space identity:
    # ln(P(E_star) / max(P_low_peak, P_high_peak))
    # = (phi at peak with smaller phi) - phi(E_star)
    # where phi = beta * E - ln g.
    beta_c = 1.0 / (kB * T_c)
    phi = beta_c * energies - ln_g
    energy_spacing = float(energies[1] - energies[0])

    def nearest_index(E: float) -> int:
        i = int(round((E - energies[0]) / energy_spacing))
        return max(0, min(i, len(energies) - 1))

    i_peak_low = nearest_index(final_split.E_peak_low)
    i_peak_high = nearest_index(final_split.E_peak_high)
    i_star = nearest_index(final_split.E_star)
    phi_peak_min = min(phi[i_peak_low], phi[i_peak_high])
    # barrier_height = -k_B * T_c * ln(P(E_star) / max_P_peak)
    #                = -k_B * T_c * (phi_peak_min - phi[i_star])
    #                = k_B * T_c * (phi[i_star] - phi_peak_min)
    barrier_height = float(kB * T_c * (phi[i_star] - phi_peak_min))

    return CoexistencePoint(
        T_K=float(T_c),
        split=final_split,
        latent_heat=float(latent_heat),
        barrier_height=barrier_height,
        weight_imbalance=float(abs(f_mid)),
        n_bisection_steps=n_steps,
    )
