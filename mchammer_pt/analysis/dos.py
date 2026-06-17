"""Density-of-states post-processing for windowed Wang-Landau output.

`stitch_entropy` merges per-window entropy curves (the
``energy``-and-``entropy`` DataFrames produced by
``WindowResult.get_entropy()``) into a single density of states via
overlap-region alignment, working entirely in log space.

`reweight_canonical_from_dos` evaluates canonical thermodynamics
from a stitched ``ln g(E)`` curve on a user-supplied temperature
grid, also entirely in log space so large entropy ranges do not
underflow ``float64``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ase.units import kB


def stitch_entropy(
    per_window: list[pd.DataFrame],
    energy_spacing: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Stitch per-window entropy curves into a single density of states.

    All windows must lie on a common ``n * energy_spacing`` grid. Bin
    centres are matched by integer index after rounding
    ``energy / energy_spacing``, so ULP-level drift between windows
    computed by independent Wang-Landau processes does not silently
    drop bins from the overlap region.

    Windows are sorted by minimum energy, shifted purely additively so
    that overlap regions align by mean entropy difference, and averaged
    where they overlap. The returned ``entropy`` column is globally
    rebased so that its minimum is zero (``ln g`` is only defined up to
    a single additive constant).

    The output is a complete histogram on the ``energy_spacing`` grid:
    every integer-multiple bin from the lowest to the highest populated
    energy is emitted. Interior bins that no window reached carry
    ``entropy = -inf`` (``g = 0`` -- a forbidden energy of the discrete
    spectrum), so the grid is uniform and self-describing. The grid
    spans only the populated range; no frontier extrapolation is done.

    Args:
        per_window: list of DataFrames each carrying ``energy`` and
            ``entropy`` columns. ``entropy`` is treated as ``ln g``.
        energy_spacing: bin spacing. All ``energy`` values must lie
            within numerical tolerance of an integer multiple.

    Returns:
        ``(stitched, overlap_errors)`` where ``stitched`` is a DataFrame
        with ``energy`` and ``entropy`` columns on a complete uniform
        grid (populated bins rebased to ``min = 0``; interior empty bins
        carried as ``entropy = -inf``) and ``overlap_errors`` is a dict
        of overlap-region entropy
        standard deviations keyed by ``"i-j"`` window-pair labels in
        the original input order. Pairs sharing only one bin report
        ``0.0`` (the sample std is undefined for a single point).

    Raises:
        ValueError: if ``per_window`` is empty, if any window is empty,
            if ``energy_spacing`` is non-positive, if any window's
            energies do not lie on the common grid, or if any pair of
            neighbouring windows shares no bins.
    """
    if energy_spacing <= 0.0:
        raise ValueError(
            f"energy_spacing must be > 0; got {energy_spacing}"
        )
    if not per_window:
        raise ValueError("per_window is empty; need at least one window")

    grid_tol = energy_spacing * 1e-6
    indexed: list[tuple[int, pd.DataFrame]] = []
    for i, df in enumerate(per_window):
        if df.empty:
            raise ValueError(f"Window {i} has no rows")
        e = df["energy"].to_numpy()
        bins = np.round(e / energy_spacing).astype(np.int64)
        residual = float(np.max(np.abs(e - bins * energy_spacing)))
        if residual > grid_tol:
            raise ValueError(
                f"Window {i} has energies off the energy_spacing="
                f"{energy_spacing} grid (max residual {residual:.3g} > "
                f"tolerance {grid_tol:.3g}). All windows must lie on a "
                f"common integer-multiple grid."
            )
        indexed.append((i, df.assign(_bin=bins)))

    ordered = sorted(indexed, key=lambda t: t[1]["energy"].min())

    errors: dict[str, float] = {}
    for k in range(1, len(ordered)):
        idx_l, df_l = ordered[k - 1]
        idx_r, df_r = ordered[k]
        shared = np.intersect1d(
            df_l["_bin"].to_numpy(), df_r["_bin"].to_numpy()
        )
        if shared.size == 0:
            raise ValueError(
                f"No overlapping bins between windows {idx_l} and "
                f"{idx_r}. Adjacent windows must share at least one "
                f"bin so they can be aligned."
            )
        s_l = df_l.set_index("_bin").loc[shared, "entropy"]
        s_r = df_r.set_index("_bin").loc[shared, "entropy"]
        diff = s_r - s_l
        offset = float(diff.mean())
        errors[f"{idx_l}-{idx_r}"] = (
            0.0 if shared.size == 1 else float(diff.std())
        )
        df_r = df_r.copy()
        df_r["entropy"] = df_r["entropy"] - offset
        ordered[k] = (idx_r, df_r)

    stacked = pd.concat([df_w for _, df_w in ordered], ignore_index=True)
    merged = stacked.groupby("_bin", sort=True)["entropy"].mean()

    # Rebase so the minimum populated ln g is zero (ln g is defined only
    # up to an additive constant). The offset is the minimum over finite
    # entries: an input window may already carry -inf (g=0) bins (e.g. a
    # re-stitched complete histogram), and folding -inf into the offset
    # would yield inf/NaN. Done before the -inf fill below so the
    # empty-bin sentinel is never folded into the offset.
    finite_entropy = merged[np.isfinite(merged)]
    if finite_entropy.empty:
        raise ValueError(
            "stitch_entropy: no finite entropy values across all windows "
            "(every bin is g=0); cannot rebase to a finite minimum."
        )
    merged = merged - finite_entropy.min()

    # Materialise a complete histogram: every integer bin from the lowest
    # to the highest populated bin. Interior bins no window reached carry
    # ln g = -inf (g = 0, a forbidden energy of the discrete spectrum)
    # rather than being dropped, so the output is a self-describing
    # uniform grid. No frontier extrapolation: the grid spans only the
    # populated range.
    full_bins = pd.RangeIndex(
        int(merged.index.min()), int(merged.index.max()) + 1, name="_bin",
    )
    merged = merged.reindex(full_bins, fill_value=-np.inf)

    stitched = pd.DataFrame({
        "energy": merged.index.to_numpy() * energy_spacing,
        "entropy": merged.to_numpy(),
    })
    return stitched, errors


def _canonical_log_weights(
    energies: np.ndarray,
    log_g: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann weights ``w(E, T)`` and partition sums ``Z(T)`` in log space.

    Forms ``log_w = ln g - E / (kB T)``, max-shifted per temperature so a
    large entropy range does not underflow ``float64``. ``-inf`` entries in
    ``log_g`` (forbidden bins, ``g = 0``) contribute zero weight.

    Args:
        energies: bin energies (eV), shape ``(n_E,)``.
        log_g: ``ln g(E)`` per bin, shape ``(n_E,)``.
        temperatures: strictly-positive temperatures (K), shape ``(n_T,)``.

    Returns:
        ``(w, Z)`` with ``w`` of shape ``(n_E, n_T)`` and ``Z`` of shape
        ``(n_T,)``. The per-temperature max-shift leaves ``w`` and ``Z``
        sharing an unknown positive constant that cancels in any ratio
        ``sum(w * f) / Z``.
    """
    beta = 1.0 / (kB * temperatures)                                # (n_T,)
    log_w = log_g[:, None] - beta[None, :] * energies[:, None]      # (n_E, n_T)
    log_w -= log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w)
    return w, w.sum(axis=0)


def reweight_canonical_from_dos(
    dos: pd.DataFrame,
    temperatures: np.ndarray,
) -> pd.DataFrame:
    """Canonical reweighting from a stitched ``ln g(E)`` curve.

    Operates entirely in log space so a large entropy range does not
    underflow ``float64``. The ``entropy`` column of ``dos`` is treated
    as ``ln g``.

    Args:
        dos: DataFrame with ``energy`` (eV) and ``entropy`` (``ln g``)
            columns.
        temperatures: array-like of temperatures in Kelvin; must be
            strictly positive.

    Returns:
        DataFrame with columns ``T_K``, ``E_mean`` (eV), ``var_E``
        (eV^2), ``Cv`` (eV/K), one row per temperature.

    Raises:
        ValueError: if ``dos`` has no rows, or if any element of
            ``temperatures`` is non-positive.
    """
    if dos.empty:
        raise ValueError("dos has no rows; need at least one energy bin")
    T_arr = np.asarray(temperatures, dtype=float)
    if np.any(T_arr <= 0.0):
        raise ValueError(
            f"temperatures must be strictly positive (K); "
            f"got min={float(T_arr.min())}"
        )
    E = dos["energy"].to_numpy()
    log_g = dos["entropy"].to_numpy()
    w, Z = _canonical_log_weights(E, log_g, T_arr)
    E_mean = (w * E[:, None]).sum(axis=0) / Z
    E2_mean = (w * (E[:, None] ** 2)).sum(axis=0) / Z
    var_E = E2_mean - E_mean ** 2
    Cv = var_E / (kB * T_arr ** 2)
    return pd.DataFrame({
        "T_K": T_arr,
        "E_mean": E_mean,
        "var_E": var_E,
        "Cv": Cv,
    })
