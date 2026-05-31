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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

from mchammer_pt.analysis._partition import partition_means, partition_sums
from mchammer_pt.analysis.dos import reweight_canonical_from_dos

# Cv-peak seed scan: log-spaced T grid over a generous kT-scale range.
# The range is intentionally wide because the kT-scale heuristic is
# only a coarse estimate; the Cv peak picks the right T inside it.
_CV_SEED_N_T = 200
_CV_SEED_KT_LOW_FRAC = 0.05
_CV_SEED_KT_HIGH_FRAC = 20.0

# Walk-outward bracket: linear step in K. A first-order bimodal
# window is typically tens of K wide on lattice systems regardless
# of where Tc sits, so 1 K samples the window at adequate resolution
# without depending on T_seed's magnitude. Step budget is generous
# enough to cross a wide bimodal window or to walk the bracket end
# to the bracketed sign change.
_WALK_STEP_K = 1.0
_WALK_MAX_STEPS = 500


def _smooth_ln_g(ln_g: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``ln g`` for topology-detection purposes.

    Wraps ``scipy.ndimage.gaussian_filter1d`` with ``mode='nearest'``
    (constant extrapolation at the boundary). Used only for finding
    phase peaks and the saddle; all weight integrals downstream
    consume raw ``ln g``.

    Args:
        ln_g: 1-D array of entropy values per bin.
        sigma: Gaussian standard deviation in bins. Must be >= 0.
            ``sigma=0`` returns the input unchanged.

    Returns:
        Smoothed array with the same shape as ``ln_g``.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0; got {sigma!r}")
    if sigma == 0.0:
        return ln_g.copy()
    return gaussian_filter1d(ln_g, sigma=sigma, mode="nearest")


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
            refined from the two deepest local minima of
            ``phi(E) = beta * E - ln g(E)`` at ``T_K``.
        E_peak_high: high-energy phase peak position in eV, sub-bin
            refined.
        E_star: dividing energy in eV — the maximum of ``phi``
            between the two peaks at ``T_K``, sub-bin refined.
        T_K: temperature in Kelvin at which the peaks and valley
            were located.
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
    smoothing_sigma: float = 0.0,
) -> PhaseSplit:
    """Locate the two phase peaks and the dividing energy at T_K.

    The two phase peak positions ``E_peak_low`` and ``E_peak_high``
    are the two deepest local minima of
    ``phi(E) = beta * E - ln g(E)``. ``E_star`` is the maximum of
    ``phi`` between them. All three positions are refined to
    sub-bin precision by three-point parabolic fits.

    Args:
        dos: DataFrame with ``energy`` (eV) and ``entropy`` (``ln g``)
            columns on a uniform grid.
        T_K: temperature in Kelvin at which the peaks and valley are
            located.
        min_peak_separation: minimum number of bins required between
            the two phase peaks. Default 5.
        smoothing_sigma: Gaussian standard deviation in bins applied to
            ``ln g`` before computing ``phi`` for topology detection.
            ``0.0`` (default) disables smoothing and reproduces the
            pre-smoothing behaviour exactly. Positive values smooth out
            bin-scale shot-noise dimples that can defeat the local-minima
            detection on under-converged DOS data.

    Returns:
        ``PhaseSplit`` with sub-bin-refined peak and valley positions
        and the supplied ``T_K``.

    Raises:
        ValueError: if ``dos`` is empty or ``T_K <= 0``.
        NotBimodalError: if ``P(E | T_K)`` is not bimodal (fewer than
            two local minima of ``phi``, or the two deepest within
            ``min_peak_separation`` bins of each other).
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
    if not (np.isfinite(energies).all() and np.isfinite(ln_g).all()):
        raise ValueError(
            "find_phase_split: dos contains non-finite (NaN/inf) "
            "values in 'energy' or 'entropy' columns"
        )
    ln_g_for_topology = _smooth_ln_g(ln_g, sigma=smoothing_sigma)
    beta = 1.0 / (kB * T_K)
    phi = beta * energies - ln_g_for_topology

    try:
        peak_idx = _two_dominant_peak_indices(phi)
    except NotBimodalError as exc:
        raise NotBimodalError(
            f"find_phase_split at T={T_K} K: {exc}"
        ) from exc
    i_left, i_right = int(peak_idx[0]), int(peak_idx[1])
    if i_right - i_left < min_peak_separation:
        raise NotBimodalError(
            f"find_phase_split: two deepest phi minima at bin indices "
            f"{i_left} and {i_right} are within "
            f"{min_peak_separation} bins of each other at T={T_K} K"
        )

    # Sub-bin refinement of each phase peak (minimum of phi).
    E_peak_low = _parabolic_vertex(
        energies[i_left - 1], energies[i_left], energies[i_left + 1],
        phi[i_left - 1], phi[i_left], phi[i_left + 1],
    )
    E_peak_high = _parabolic_vertex(
        energies[i_right - 1], energies[i_right], energies[i_right + 1],
        phi[i_right - 1], phi[i_right], phi[i_right + 1],
    )

    interior = slice(i_left, i_right + 1)
    # i_valley is strictly between i_left and i_right: both are local
    # minima of phi (strict interior minima), so phi at either endpoint
    # of the slice is smaller than at least one neighbour inside the slice.
    i_valley = int(np.argmax(phi[interior])) + i_left
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


def _two_dominant_peak_indices(phi: np.ndarray) -> np.ndarray:
    """Return the bin indices of the two deepest local minima of phi.

    Local minima are interior bins strictly lower than both
    neighbours. The two with the smallest ``phi`` values are returned
    in ascending bin-index order. These are the two dominant peaks
    of ``P(E | T) ∝ exp(-phi(E))``.

    Raises:
        NotBimodalError: if fewer than two local minima exist.
    """
    is_min = (phi[1:-1] < phi[:-2]) & (phi[1:-1] < phi[2:])
    minima_idx = np.flatnonzero(is_min) + 1
    if minima_idx.size < 2:
        raise NotBimodalError(
            f"fewer than two local minima of phi "
            f"(found {minima_idx.size})"
        )
    two_smallest = minima_idx[np.argsort(phi[minima_idx])[:2]]
    return np.array(sorted(int(x) for x in two_smallest), dtype=np.int64)


def _cv_peak_seed(dos: pd.DataFrame) -> float:
    """Temperature in K of the heat-capacity peak.

    Computes ``Cv(T) = Var_T(E) / (k_B * T**2)`` on a log-spaced T
    grid spanning the kT-scale heuristic range and returns the
    ``argmax``. ``Cv`` peaks where energy fluctuations are largest;
    for a first-order DOS, this is inside the bimodal-``P(E|T)``
    window, making the peak T a physical seed for the walk-outward
    bracket finder.

    The kT-scale heuristic is ``(E_max - E_min) / (ln_g_max -
    ln_g_min)``, the inverse of the mean slope of ``ln g``, which
    is the right order of magnitude for the coexistence temperature
    of a typical lattice system. The scan range is wide
    (``[0.05, 20] * kT_scale / k_B``) because we only need the peak
    to lie inside it, not to centre on it.

    Raises:
        ValueError: if the DOS contains non-finite values.
        ValueError: if ``ln g`` has no variation across the DOS, so
            no kT scale can be derived.
        ValueError: if the Cv peak sits at the scan-range edge,
            indicating the true peak lies outside the heuristic
            range. Caller should supply ``T_bracket`` explicitly.
    """
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    if not (np.isfinite(energies).all() and np.isfinite(ln_g).all()):
        raise ValueError(
            "cv_peak_seed: dos contains non-finite (NaN/inf) "
            "values in 'energy' or 'entropy' columns"
        )

    E_range = float(energies.max() - energies.min())
    ln_g_range = float(ln_g.max() - ln_g.min())
    if ln_g_range < 1e-10:
        raise ValueError(
            "cv_peak_seed: ln g has no variation across the DOS "
            f"(range = {ln_g_range:.2g}); cannot derive a kT scale. "
            "Supply T_bracket explicitly."
        )
    kT_scale = E_range / ln_g_range
    T_lo = max(_CV_SEED_KT_LOW_FRAC * kT_scale / kB, 1.0)
    T_hi = _CV_SEED_KT_HIGH_FRAC * kT_scale / kB
    Ts = np.logspace(np.log10(T_lo), np.log10(T_hi), _CV_SEED_N_T)
    cv_df = reweight_canonical_from_dos(dos, Ts)
    cv = cv_df["Cv"].to_numpy()
    i_peak = int(cv.argmax())
    if i_peak == 0 or i_peak == _CV_SEED_N_T - 1:
        raise ValueError(
            f"cv_peak_seed: Cv peak at scan-range edge "
            f"(T={Ts[i_peak]:.1f} K, scan range "
            f"[{T_lo:.1f}, {T_hi:.1f}] K). The true peak likely lies "
            "outside the kT-scale heuristic range. Supply T_bracket "
            "explicitly."
        )
    return float(Ts[i_peak])


def _walk_outward_bracket(
    dos: pd.DataFrame,
    T_seed: float,
    *,
    min_peak_separation: int = 5,
    step_K: float = _WALK_STEP_K,
    max_steps: int = _WALK_MAX_STEPS,
) -> tuple[float, float]:
    """Walk outward from ``T_seed`` until ``imbalance(T)`` changes sign.

    From a ``T_seed`` inside the bimodal-``P(E|T)`` window, step T
    by ``step_K`` Kelvin per iteration in the direction implied by
    the sign of ``imbalance(T_seed) = w_low - w_high`` at the seed.
    Positive imbalance (low-E phase heavier) means heating
    populates the high-E phase, so walk up; negative means cool.
    Stop when the sign flips and return the bracket.

    The walk is domain-aware: if a step lands outside the
    bimodal-``P(E|T)`` window (``find_phase_split`` raises
    ``NotBimodalError``), the walk terminates with a
    ``NoBracketError`` whose message names the failure shape.
    Imbalance not crossing zero before the window edge means the
    DOS doesn't exhibit equal-area coexistence in this T range —
    a genuine failure mode, not a bracket-search artefact.

    Args:
        dos: stitched DOS.
        T_seed: starting temperature in Kelvin, must be inside the
            bimodal window. Typically the output of
            :func:`_cv_peak_seed`.
        min_peak_separation: forwarded to
            :func:`find_phase_split`.
        step_K: linear step size in Kelvin. Default 1 K — first-
            order bimodal windows on lattice systems are typically
            tens of K wide, so 1 K samples the window at adequate
            resolution to detect a sign change or to assert that
            imbalance is one-signed across the window.
        max_steps: maximum walk steps before giving up. Default
            500.

    Returns:
        ``(T_lo, T_hi)`` with ``imbalance(T_lo) * imbalance(T_hi)
        <= 0``. Both endpoints are inside the bimodal window.

    Raises:
        NotBimodalError: if ``find_phase_split`` fails at
            ``T_seed`` itself.
        NoBracketError: if the walk hits the bimodal-window edge
            before imbalance changes sign, or if ``max_steps`` is
            exceeded.
    """
    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()

    def imbalance_at(T: float) -> float:
        split = find_phase_split(
            dos, T_K=T, min_peak_separation=min_peak_separation,
        )
        w_lo, w_hi = partition_sums(energies, ln_g, T, split.E_star)
        return w_lo - w_hi

    f_seed = imbalance_at(T_seed)  # raises NotBimodalError if seed is bad
    if f_seed == 0.0:
        # Exact zero is virtually impossible with floating point, but
        # if it happens, return a tight bracket around T_seed so the
        # downstream brentq sees a valid sign-change interval.
        return T_seed - step_K, T_seed + step_K

    step = step_K if f_seed > 0 else -step_K  # walk up if low-E heavier
    T = T_seed
    T_prev = T_seed  # last T at which we successfully sampled f
    for _ in range(max_steps):
        T = T + step
        if T <= 0.0:
            raise NoBracketError(
                f"walk_outward_bracket: walked down past T=0 K from "
                f"T_seed={T_seed:.2f} K without finding a sign change. "
                "The DOS does not exhibit equal-area coexistence at "
                "any positive T below the seed."
            )
        try:
            f = imbalance_at(T)
        except NotBimodalError as exc:
            raise NoBracketError(
                f"walk_outward_bracket: hit bimodal-window edge at "
                f"T={T:.2f} K (walking "
                f"{'up' if step > 0 else 'down'} from "
                f"T_seed={T_seed:.2f} K in {step_K:.2f} K steps; last "
                f"bimodal T sampled was {T_prev:.2f} K) without "
                f"finding a sign change in imbalance(T). Imbalance "
                f"stays {'>0' if f_seed > 0 else '<0'} across every "
                "sampled T in the bimodal window; the DOS does not "
                "exhibit equal-area coexistence in this T range."
            ) from exc
        if f * f_seed <= 0.0:
            return (T, T_seed) if step < 0 else (T_seed, T)
        T_prev = T

    raise NoBracketError(
        f"walk_outward_bracket: reached max_steps={max_steps} from "
        f"T_seed={T_seed:.2f} K (walking "
        f"{'up' if step > 0 else 'down'} in {step_K:.2f} K steps) "
        f"without finding a sign change. Final T={T:.2f} K, f={f:.3g}."
    )


@dataclass(frozen=True)
class CoexistencePoint:
    """First-order coexistence point obtained from a stitched DOS.

    Returned by :func:`equal_area_temperature`. Bundles the phase
    split at coexistence with the diagnostics that only make sense
    there (latent heat, barrier height) and the iteration counters
    that report on solver behaviour.

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
            temperature, the solver residual.
        n_brentq_iterations: total number of
            ``scipy.optimize.brentq`` iterations summed across all
            self-consistency passes.
        n_self_consistent_iter: number of passes through the
            (T_c, E_star) fixed-point iteration. Each pass is one
            brentq solve plus one re-detection of the saddle at the
            candidate T_c.
        self_consistent_converged: ``True`` if the iteration met
            the configured tolerance within the budget. ``False``
            indicates the iteration was truncated and the reported
            ``T_K`` may differ from the true fixed point by more
            than the tolerance — typically a signal of over-smoothing
            on shallow-bimodal data.

    The coexistence temperature is exposed as the read-only
    ``T_K`` property, delegating to ``split.T_K``.
    """

    split: PhaseSplit
    latent_heat: float
    barrier_height: float
    weight_imbalance: float
    n_brentq_iterations: int
    n_self_consistent_iter: int
    self_consistent_converged: bool

    @property
    def T_K(self) -> float:
        """The equal-area coexistence temperature, in Kelvin."""
        return self.split.T_K

    @property
    def n_iterations(self) -> int:
        """Deprecated alias for :attr:`n_brentq_iterations`."""
        import warnings
        warnings.warn(
            "CoexistencePoint.n_iterations is deprecated; use "
            "n_brentq_iterations instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.n_brentq_iterations


def equal_area_temperature(
    dos: pd.DataFrame,
    *,
    T_bracket: tuple[float, float] | None = None,
    xtol: float = 1e-4,
    min_peak_separation: int = 5,
) -> CoexistencePoint:
    """Equal-area coexistence temperature from a stitched DOS.

    Solves for the T at which ``imbalance(T) = w_low(T) -
    w_high(T) = 0``, where ``w_low``, ``w_high`` are the partition
    sums on either side of the dividing energy ``E*(T)`` returned
    by :func:`find_phase_split`. Uses ``scipy.optimize.brentq``
    inside an adaptive bracket.

    Bracket selection (when ``T_bracket`` is ``None``):

    1. Compute the heat-capacity peak ``T_seed`` from the stitched
       DOS (see :func:`_cv_peak_seed`). This is guaranteed to lie
       inside the bimodal-``P(E|T)`` window of a first-order DOS.
    2. From ``T_seed``, walk outward in T in the direction implied
       by the sign of ``imbalance(T_seed)`` until the sign flips
       (see :func:`_walk_outward_bracket`). The walk stays inside
       the bimodal window; hitting a window edge is a hard failure
       and surfaces as ``NoBracketError``.

    Args:
        dos: stitched DOS as produced by
            ``mchammer_pt.analysis.dos.stitch_entropy``.
        T_bracket: optional ``(T_lo, T_hi)`` bracket in Kelvin. If
            supplied, bypasses Cv-peak seeding and walk-outward;
            the user is responsible for the bracket being valid
            (positive, ordered, sign-changing, inside the bimodal
            window).
        xtol: relative tolerance on T. Default 1e-4.
        min_peak_separation: forwarded to
            :func:`find_phase_split`.

    Returns:
        A :class:`CoexistencePoint`.

    Raises:
        ValueError: on invalid inputs, or when the Cv-peak seed
            cannot be derived (flat ``ln g``, Cv peak at scan-range
            edge).
        NotBimodalError: when ``T_bracket`` is ``None`` and the Cv
            peak temperature is not in the bimodal-``P(E|T)``
            window — i.e. the DOS does not exhibit first-order
            coexistence at the Cv-peak temperature.
        NoBracketError: when the walk-outward search hits a
            bimodal-window edge before finding a sign change, or
            when a user-supplied ``T_bracket`` doesn't sign-change
            or extends outside the bimodal window.
    """
    if dos.empty:
        raise ValueError("dos has no rows; need at least one energy bin")
    if xtol <= 0.0:
        raise ValueError(f"xtol must be > 0; got {xtol}")
    if min_peak_separation < 1:
        raise ValueError(
            f"min_peak_separation must be >= 1; got {min_peak_separation}"
        )

    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()

    def imbalance(T: float) -> float:
        split = find_phase_split(
            dos, T_K=T, min_peak_separation=min_peak_separation,
        )
        w_low, w_high = partition_sums(energies, ln_g, T, split.E_star)
        return w_low - w_high

    if T_bracket is None:
        T_seed = _cv_peak_seed(dos)
        try:
            T_lo, T_hi = _walk_outward_bracket(
                dos, T_seed, min_peak_separation=min_peak_separation,
            )
        except NotBimodalError as exc:
            raise NotBimodalError(
                f"Cv peak at T={T_seed:.2f} K, but P(E|T) is not "
                f"bimodal there: {exc}. The DOS does not exhibit "
                "first-order coexistence at the Cv-peak temperature."
            ) from exc
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

    def imbalance_for_brentq(T: float) -> float:
        # brentq evaluates strictly inside [T_lo, T_hi]. If shape
        # analysis fails mid-bracket, the bracket extends outside
        # the bimodal region — surface as NoBracketError so the
        # caller sees a meaningful diagnostic instead of an opaque
        # scipy exception.
        try:
            return imbalance(T)
        except NotBimodalError as exc:
            raise NoBracketError(
                f"shape analysis failed at mid-bracket T={T:.4f} K; "
                f"the bracket extends outside the bimodal region: {exc}"
            ) from exc

    try:
        T_c, result = brentq(
            imbalance_for_brentq, T_lo, T_hi,
            xtol=xtol * 0.5 * (T_lo + T_hi),
            full_output=True, disp=True,
        )
    except ValueError as exc:
        # brentq raises ValueError("f(a) and f(b) must have different signs")
        # when the user-supplied bracket doesn't sign-change. The walk-outward
        # path always returns a valid sign-changing bracket, so this only
        # fires for a bad user-supplied T_bracket.
        if "different signs" in str(exc).lower():
            raise NoBracketError(
                f"imbalance has same sign at both endpoints "
                f"(T_lo={T_lo}, T_hi={T_hi}); extend T_bracket"
            ) from exc
        raise
    n_steps = int(result.iterations)
    f_mid = imbalance(T_c)

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
        n_brentq_iterations=n_steps,
        n_self_consistent_iter=0,
        self_consistent_converged=False,
    )
