"""First-order coexistence-point analysis on stitched Wang-Landau DOS.

Locates the equal-area coexistence temperature T_c at which the
canonical energy distribution P(E|T) has equal integrated weight on
either side of the free-energy minimum E*(T) between its two peaks.
"""
from __future__ import annotations

import warnings
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

# Default cap on (Tc, E_star) self-consistency passes. Exposed as a
# module constant so the CLI can reference it without duplicating
# the literal.
DEFAULT_MAX_SELF_CONSISTENT_ITER = 20

# Lower bound on the relative tolerance forwarded to
# ``scipy.optimize.brentq`` as ``rtol``. scipy rejects ``rtol`` below
# ``4 * eps`` with a ValueError, so reject it here with a clearer
# message.
_RTOL_FLOOR = 4.0 * np.finfo(float).eps


def _smooth_ln_g(ln_g: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``ln g`` for topology-detection purposes.

    Wraps ``scipy.ndimage.gaussian_filter1d`` with ``mode='nearest'``
    (the boundary sample value is replicated for samples beyond the
    array edge). Used only for finding phase peaks and the saddle;
    all weight integrals downstream consume raw ``ln g``.

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
    """Raised when ``imbalance(T)`` cannot be bracketed for brentq.

    Two failure modes share this exception:

    - The outward walk from the Cv-peak seed hits the T scan range
      edge before ``imbalance(T)`` changes sign (the DOS does not
      exhibit equal-area coexistence within the scanned T range).
    - A user-supplied ``T_bracket`` evaluates to the same sign of
      ``imbalance`` at both endpoints (the bracket does not
      straddle a coexistence Tc).
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


def _find_saddle_at(
    energies: np.ndarray,
    ln_g_smoothed: np.ndarray,
    T_K: float,
    min_peak_separation: int,
    fallback_T_factors: tuple[float, ...] = (
        0.99, 1.01, 0.95, 1.05, 0.90, 1.10, 0.80, 1.20,
    ),
) -> tuple[float, float, float]:
    """Find ``(E_star, E_peak_low, E_peak_high)`` from smoothed phi at T_K.

    If the smoothed phi at ``T_K`` is not bimodal, retry at
    multiplicative perturbations of T given by
    ``fallback_T_factors``. Raises :class:`NotBimodalError` if no T
    in the sweep yields a bimodal smoothed phi.

    Args:
        energies: bin energies (sorted ascending).
        ln_g_smoothed: smoothed entropy values per bin.
        T_K: temperature in K to search at first.
        min_peak_separation: minimum bin separation between the
            two phase peaks.
        fallback_T_factors: multiplicative factors applied to
            ``T_K`` to define fallback temperatures, tried in
            order if the primary T is not bimodal.

    Returns:
        ``(E_star, E_peak_low, E_peak_high)`` — sub-bin refined
        positions.
    """
    candidate_Ts = (T_K,) + tuple(T_K * f for f in fallback_T_factors)
    last_exc: NotBimodalError | None = None
    for T in candidate_Ts:
        try:
            beta = 1.0 / (kB * T)
            phi = beta * energies - ln_g_smoothed
            peak_idx = _two_dominant_peak_indices(phi)
            i_lo, i_hi = int(peak_idx[0]), int(peak_idx[1])
            if i_hi - i_lo < min_peak_separation:
                raise NotBimodalError(
                    f"two deepest phi minima at bin indices {i_lo} "
                    f"and {i_hi} are within {min_peak_separation} "
                    f"bins of each other at T={T} K"
                )
            i_valley = int(np.argmax(phi[i_lo : i_hi + 1])) + i_lo
            E_peak_low = _parabolic_vertex(
                energies[i_lo - 1], energies[i_lo], energies[i_lo + 1],
                phi[i_lo - 1], phi[i_lo], phi[i_lo + 1],
            )
            E_peak_high = _parabolic_vertex(
                energies[i_hi - 1], energies[i_hi], energies[i_hi + 1],
                phi[i_hi - 1], phi[i_hi], phi[i_hi + 1],
            )
            E_star = _parabolic_vertex(
                energies[i_valley - 1], energies[i_valley],
                energies[i_valley + 1],
                phi[i_valley - 1], phi[i_valley], phi[i_valley + 1],
            )
            return float(E_star), float(E_peak_low), float(E_peak_high)
        except NotBimodalError as exc:
            last_exc = exc
            continue
    raise NotBimodalError(
        f"smoothed phi not bimodal at T={T_K} K nor at any of "
        f"{len(fallback_T_factors)} nearby T values (last error: "
        f"{last_exc})"
    )


def _walk_for_sign_change(
    energies: np.ndarray,
    ln_g_raw: np.ndarray,
    T_start: float,
    E_star: float,
    *,
    step_K: float = _WALK_STEP_K,
    max_steps: int = _WALK_MAX_STEPS,
    T_min: float = 1.0,
    T_max_factor: float = 5.0,
) -> tuple[float, float]:
    """Walk outward from ``T_start`` in linear ``step_K`` increments
    until ``imbalance(T; E_star)`` changes sign.

    ``imbalance(T) = w_low - w_high`` is computed against the raw
    ``ln g`` with a fixed ``E_star`` — i.e. no peak-detection runs
    inside the walk.

    Direction is determined by the sign of
    ``imbalance(T_start)``: positive imbalance (low-E phase heavier
    than high-E) → walk up in T.

    Raises :class:`NoBracketError` if no sign change within the
    step budget or scan range.
    """
    def f(T: float) -> float:
        w_lo, w_hi = partition_sums(energies, ln_g_raw, T, E_star)
        return float(w_lo - w_hi)

    f_start = f(T_start)
    # An f == 0 at T_start is handled implicitly by the loop: the
    # next sample satisfies f_T * f_prev <= 0.0 and we return that
    # bracket. brentq accepts an endpoint where f == 0 (it's a root).
    direction = +1 if f_start >= 0 else -1
    T_max = T_start * T_max_factor

    T_prev = T_start
    f_prev = f_start
    for step in range(1, max_steps + 1):
        T = T_start + direction * step * step_K
        if T <= T_min or T >= T_max:
            raise NoBracketError(
                f"imbalance(T) did not change sign in walk from "
                f"T={T_start:.2f} K (direction {direction:+d}, hit "
                f"scan edge at T={T:.2f} K)"
            )
        f_T = f(T)
        if f_T * f_prev <= 0.0:
            return (T_prev, T) if direction > 0 else (T, T_prev)
        T_prev = T
        f_prev = f_T
    raise NoBracketError(
        f"imbalance(T) did not change sign within {max_steps} steps "
        f"of {step_K} K from T={T_start:.2f} K"
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
        weight_imbalance: normalised solver residual
            ``|w_low - w_high| / (w_low + w_high)`` at the converged
            temperature, evaluated against the converged ``E_star``.
            Dimensionless; near-zero for a well-solved Tc.
        n_brentq_iterations: total number of
            ``scipy.optimize.brentq`` iterations summed across all
            self-consistency passes.
        n_self_consistent_iter: number of passes through the
            (T_c, E_star) fixed-point iteration. Each pass is one
            brentq solve plus one re-detection of the saddle at the
            candidate T_c.
        self_consistent_converged: ``True`` if the (T_c, E_star)
            fixed-point iteration met ``self_consistent_tol_K``
            within ``max_self_consistent_iter`` passes. When
            ``max_self_consistent_iter=0`` (iteration disabled),
            this is ``True`` by convention — there is no
            convergence question to defer. ``False`` indicates the
            iteration was truncated (budget exhausted or a
            break-path triggered, e.g. saddle re-detection failed);
            the reported ``T_K`` may differ from the true fixed
            point by more than ``self_consistent_tol_K`` —
            typically a signal of over-smoothing on shallow-bimodal
            data.

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
    smoothing_sigma: float = 2.0,
    max_self_consistent_iter: int = DEFAULT_MAX_SELF_CONSISTENT_ITER,
    damping: float = 0.5,
    self_consistent_tol_K: float = 1e-3,
) -> CoexistencePoint:
    """Equal-area coexistence temperature from a stitched DOS.

    The algorithm decouples saddle (``E*``) detection from the
    brentq root-find on ``imbalance(T)``:

    1. Smooth ``ln g`` once with ``smoothing_sigma`` (in bins) for
       topology detection only. Weight integrals downstream use
       raw ``ln g``.
    2. Seed at the heat-capacity peak (or ``T_bracket`` midpoint
       if supplied), and locate ``E*`` from the smoothed phi at
       the seed via :func:`_find_saddle_at` (with fallback Ts if
       the seed is marginal).
    3. Walk outward from the seed on raw-data
       ``imbalance(T; E* fixed)`` until the sign changes
       (:func:`_walk_for_sign_change`).
    4. ``brentq`` ``imbalance(T; E* fixed) = 0`` inside the
       bracket.
    5. Re-detect ``E*`` at the converged ``Tc`` and re-solve, with
       linear damping (``damping`` parameter) to stabilise the
       fixed-point map on shallow-bimodal data, until the undamped
       residual ``|Tc_raw - Tc| < self_consistent_tol_K`` or
       ``max_self_consistent_iter`` is reached.

    Args:
        dos: stitched DOS as produced by
            ``mchammer_pt.analysis.dos.stitch_entropy``.
        T_bracket: optional ``(T_lo, T_hi)`` bracket in Kelvin. If
            supplied, bypasses Cv-peak seeding and walk-outward;
            the midpoint is used as the seed for ``E*`` detection.
            The user is responsible for the bracket being valid
            (positive, ordered, sign-changing for the fixed
            ``E*``).
        xtol: relative tolerance on the coexistence temperature,
            forwarded to ``scipy.optimize.brentq`` as its ``rtol``
            argument (scaled off the current root estimate). Must be
            at least ``4 * eps`` (scipy's ``rtol`` floor). Default
            1e-4.
        min_peak_separation: minimum bin separation between the
            two phase peaks. Default 5.
        smoothing_sigma: Gaussian standard deviation in bins
            applied to ``ln g`` for the topology (peak/saddle)
            detection step only. Weight integrals consume raw
            ``ln g``. Default 2.0.
        max_self_consistent_iter: maximum (Tc, E*) re-iterations
            after the initial brentq solve. 0 disables iteration
            (frozen mode). In frozen mode the reported split (peaks
            and E*) is the seed-temperature saddle detection —
            possibly at a fallback temperature near the seed if the
            exact seed T was not bimodal — while ``T_K`` is the
            single-pass equal-area root for that frozen saddle; the
            reported peaks may therefore correspond to a temperature
            slightly different from ``T_K``. Default 20.
        damping: linear-mixing factor for the (Tc, E*) update; must
            satisfy ``0 < damping <= 1``. ``damping=1`` is no
            damping. Besides stabilising the iteration (the undamped
            map can oscillate), it influences where on the
            saddle-detection plateau the iteration settles, and
            therefore has a small (~discretisation-scale) effect on
            the delivered Tc. Default 0.5.
        self_consistent_tol_K: convergence tolerance on the undamped
            fixed-point residual ``|Tc_raw - Tc|`` between successive
            self-consistency passes, in K. The achievable Tc accuracy
            is bounded below by the saddle-detection discretisation —
            the smoothed-phi argmax is piecewise-constant in T over a
            small plateau — so tightening this tolerance past that
            plateau does not refine Tc further. It controls when
            convergence is *declared*, not the ultimate accuracy.
            Default 1e-3.

    Returns:
        A :class:`CoexistencePoint`.

    Raises:
        ValueError: on invalid inputs, or when the Cv-peak seed
            cannot be derived (flat ``ln g``, Cv peak at scan-range
            edge).
        NotBimodalError: when no T near the seed yields a bimodal
            smoothed phi.
        NoBracketError: when the walk-outward search hits the scan
            edge before finding a sign change, or when a
            user-supplied ``T_bracket`` doesn't sign-change.
    """
    if dos.empty:
        raise ValueError("dos has no rows; need at least one energy bin")
    if xtol <= 0.0:
        raise ValueError(f"xtol must be > 0; got {xtol}")
    if xtol < _RTOL_FLOOR:
        raise ValueError(
            f"xtol must be >= {_RTOL_FLOOR:.2e} (scipy.optimize.brentq's "
            f"rtol floor); got {xtol}"
        )
    if min_peak_separation < 1:
        raise ValueError(
            f"min_peak_separation must be >= 1; got {min_peak_separation}"
        )
    if smoothing_sigma < 0.0:
        raise ValueError(
            f"smoothing_sigma must be >= 0; got {smoothing_sigma}"
        )
    if not (0.0 < damping <= 1.0):
        raise ValueError(
            f"damping must satisfy 0 < damping <= 1; got {damping}"
        )
    if max_self_consistent_iter < 0:
        raise ValueError(
            f"max_self_consistent_iter must be >= 0; got "
            f"{max_self_consistent_iter}"
        )
    if self_consistent_tol_K <= 0.0:
        raise ValueError(
            f"self_consistent_tol_K must be > 0; got {self_consistent_tol_K}"
        )

    energies = dos["energy"].to_numpy()
    ln_g_raw = dos["entropy"].to_numpy()
    if not (np.isfinite(energies).all() and np.isfinite(ln_g_raw).all()):
        raise ValueError(
            "equal_area_temperature: dos contains non-finite values "
            "in 'energy' or 'entropy' columns"
        )
    ln_g_sm = _smooth_ln_g(ln_g_raw, sigma=smoothing_sigma)

    # Seed T.
    if T_bracket is not None:
        T_lo_user, T_hi_user = T_bracket
        if T_lo_user <= 0.0 or T_hi_user <= 0.0:
            raise ValueError(
                f"T_bracket entries must be > 0 K; got "
                f"({T_lo_user}, {T_hi_user})"
            )
        if T_lo_user >= T_hi_user:
            raise ValueError(
                f"T_bracket must satisfy T_lo < T_hi; got "
                f"({T_lo_user}, {T_hi_user})"
            )
        T_seed = 0.5 * (T_lo_user + T_hi_user)
    else:
        T_seed = _cv_peak_seed(dos)

    # Initial E_star at the seed.
    try:
        E_star, E_peak_low_seed, E_peak_high_seed = _find_saddle_at(
            energies, ln_g_sm, T_seed, min_peak_separation,
        )
    except NotBimodalError as exc:
        seed_desc = (
            f"T_bracket midpoint T={T_seed:.2f} K"
            if T_bracket is not None
            else f"Cv-peak seed T={T_seed:.2f} K"
        )
        raise NotBimodalError(
            f"smoothed phi is not bimodal at the {seed_desc} nor at "
            f"nearby T: {exc}. The DOS does not exhibit first-order "
            "coexistence near this temperature."
        ) from exc

    # Bracket: user-supplied or walk-derived (fixed-E_star).
    if T_bracket is None:
        T_lo, T_hi = _walk_for_sign_change(
            energies, ln_g_raw, T_seed, E_star,
        )
    else:
        T_lo, T_hi = T_lo_user, T_hi_user

    def imbalance(T: float, E_star_fixed: float) -> float:
        w_lo, w_hi = partition_sums(energies, ln_g_raw, T, E_star_fixed)
        return float(w_lo - w_hi)

    # First brentq. Explicit precheck on bracket sign-change, so we
    # don't have to string-match scipy's ValueError text.
    f_lo = imbalance(T_lo, E_star)
    f_hi = imbalance(T_hi, E_star)
    if f_lo * f_hi > 0.0:
        raise NoBracketError(
            f"imbalance has same sign at both endpoints "
            f"(T_lo={T_lo}, f_lo={f_lo:.3e}; T_hi={T_hi}, "
            f"f_hi={f_hi:.3e}); the bracket does not straddle a "
            f"coexistence Tc"
        )
    Tc, brentq_result = brentq(
        lambda T: imbalance(T, E_star),
        T_lo, T_hi,
        rtol=xtol,
        full_output=True, disp=True,
    )
    n_brentq_total = int(brentq_result.iterations)

    # Self-consistency iteration on (Tc, E_star).
    self_consistent_converged = (max_self_consistent_iter == 0)
    n_sc_iter = 0
    for iter_idx in range(1, max_self_consistent_iter + 1):
        # Reset each pass so the flag can't be left stale by a future
        # break-path edit that forgets to clear an earlier True.
        self_consistent_converged = False
        n_sc_iter = iter_idx
        try:
            E_star_new, _, _ = _find_saddle_at(
                energies, ln_g_sm, Tc, min_peak_separation,
            )
        except NotBimodalError:
            # Saddle detection failed at Tc; keep the previous E_star.
            break
        try:
            T_lo_i, T_hi_i = _walk_for_sign_change(
                energies, ln_g_raw, Tc, E_star_new,
            )
        except NoBracketError:
            # Re-bracket failed; keep current Tc.
            break
        # Explicit precheck on the inner-loop bracket. If
        # _walk_for_sign_change returned a bracket whose imbalance
        # endpoints don't straddle zero, stop iterating with the
        # current (Tc, E_star); self_consistent_converged stays
        # False to signal truncation.
        f_lo_i = imbalance(T_lo_i, E_star_new)
        f_hi_i = imbalance(T_hi_i, E_star_new)
        if f_lo_i * f_hi_i > 0.0:
            break
        Tc_raw, br = brentq(
            lambda T, E=E_star_new: imbalance(T, E),
            T_lo_i, T_hi_i,
            rtol=xtol,
            full_output=True, disp=True,
        )
        n_brentq_total += int(br.iterations)
        # Honest fixed-point residual: how far Tc is from the equal-area
        # root for the freshly-detected saddle. Damping stabilises the Tc
        # trajectory below; it must not deflate this convergence measure
        # (testing the damped increment would inflate the effective
        # tolerance by 1/damping).
        residual = abs(Tc_raw - Tc)
        Tc = (1.0 - damping) * Tc + damping * Tc_raw
        if residual < self_consistent_tol_K:
            self_consistent_converged = True
            break

    # Final reporting. Two modes, distinguished by whether the
    # self-consistency loop was permitted to run at all.
    if max_self_consistent_iter == 0:
        # Frozen single-pass (--no-self-consistent): Tc from the first
        # brentq is already the root for the seed E_star. Report that
        # saddle and its seed-temperature peaks unchanged — freezing
        # E_star at its initial value, as documented. The reported
        # (T_K, split.E_star) pair is therefore a solved root and
        # weight_imbalance below is consistent with the returned split.
        E_star_report = E_star
        peak_low_report = E_peak_low_seed
        peak_high_report = E_peak_high_seed
    else:
        # Iterated: re-detect the saddle at the converged Tc and pin Tc
        # to the true root for that saddle, so the reported split, its
        # E_star, and the residual are mutually consistent.
        final_split = find_phase_split(
            dos, T_K=Tc, min_peak_separation=min_peak_separation,
            smoothing_sigma=smoothing_sigma,
        )
        E_star_report = final_split.E_star
        peak_low_report = final_split.E_peak_low
        peak_high_report = final_split.E_peak_high

        # Final un-damped brentq pass: pin the reported Tc to the true
        # zero of imbalance(T; E_star_report). The damped iteration
        # converges to (Tc, E_star), but the last Tc assignment is a
        # linear blend that is not itself a root; and we pin against
        # the *reported* saddle so the residual below is a faithful
        # solved root for the split we return.
        if self_consistent_converged:
            try:
                T_lo_final, T_hi_final = _walk_for_sign_change(
                    energies, ln_g_raw, Tc, E_star_report,
                )
                f_lo_final = imbalance(T_lo_final, E_star_report)
                f_hi_final = imbalance(T_hi_final, E_star_report)
                if f_lo_final * f_hi_final > 0.0:
                    raise NoBracketError(
                        "final un-damped re-bracket did not straddle zero"
                    )
                Tc, br_final = brentq(
                    lambda T: imbalance(T, E_star_report),
                    T_lo_final, T_hi_final,
                    rtol=xtol,
                    full_output=True, disp=True,
                )
                n_brentq_total += int(br_final.iterations)
            except (NoBracketError, ValueError):
                # If the un-damped re-solve fails, the damped Tc was
                # the best estimate. Flag as not converged so callers
                # know the reported triple isn't a clean root.
                self_consistent_converged = False

    split = PhaseSplit(
        E_peak_low=peak_low_report,
        E_peak_high=peak_high_report,
        E_star=E_star_report,
        T_K=Tc,
    )

    mean_low, mean_high = partition_means(
        energies, ln_g_raw, Tc, E_star_report,
    )
    latent_heat = float(mean_high - mean_low)

    # Normalised residual at the reported (Tc, E_star_report) pair.
    # Dimensionless; near-zero for a well-solved Tc and consistent
    # with the returned split by construction.
    w_lo_final, w_hi_final = partition_sums(
        energies, ln_g_raw, Tc, E_star_report,
    )
    total_final = w_lo_final + w_hi_final
    if total_final > 0.0:
        weight_imbalance = float(
            abs(w_lo_final - w_hi_final) / total_final
        )
    else:
        weight_imbalance = float("nan")

    # Barrier height (raw phi at smoothed peak/saddle positions).
    beta_c = 1.0 / (kB * Tc)
    phi = beta_c * energies - ln_g_raw
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
        phi_at(split.E_peak_low),
        phi_at(split.E_peak_high),
    )
    barrier_height = float(
        kB * Tc * (phi_at(split.E_star) - phi_peak_min)
    )

    return CoexistencePoint(
        split=split,
        latent_heat=latent_heat,
        barrier_height=barrier_height,
        weight_imbalance=weight_imbalance,
        n_brentq_iterations=n_brentq_total,
        n_self_consistent_iter=n_sc_iter,
        self_consistent_converged=self_consistent_converged,
    )
