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

    Args:
        per_window: list of DataFrames each carrying ``energy`` and
            ``entropy`` columns. ``entropy`` is treated as ``ln g``.
        energy_spacing: bin spacing. All ``energy`` values must lie
            within numerical tolerance of an integer multiple.

    Returns:
        ``(stitched, overlap_errors)`` where ``stitched`` is a DataFrame
        with ``energy`` and ``entropy`` columns (rebased to ``min = 0``)
        and ``overlap_errors`` is a dict of overlap-region entropy
        standard deviations keyed by ``"i-j"`` window-pair labels in
        the original input order. Pairs sharing only one bin report
        ``0.0`` (the sample std is undefined for a single point).

    Raises:
        ValueError: if ``energy_spacing`` is non-positive, if any
            window's energies do not lie on the common grid, or if any
            pair of neighbouring windows shares no bins.
    """
    if energy_spacing <= 0.0:
        raise ValueError(
            f"energy_spacing must be > 0; got {energy_spacing}"
        )

    grid_tol = energy_spacing * 1e-6
    indexed: list[tuple[int, pd.DataFrame]] = []
    for i, df in enumerate(per_window):
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
    merged = (
        stacked.groupby("_bin", sort=True)["entropy"]
        .mean()
        .reset_index()
    )
    merged["energy"] = merged["_bin"].to_numpy() * energy_spacing
    merged = merged[["energy", "entropy"]]
    merged["entropy"] = merged["entropy"] - merged["entropy"].min()
    return merged, errors


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
        ValueError: if any element of ``temperatures`` is non-positive.
    """
    T_arr = np.asarray(temperatures, dtype=float)
    if np.any(T_arr <= 0.0):
        raise ValueError(
            f"temperatures must be strictly positive (K); "
            f"got min={float(T_arr.min())}"
        )
    E = dos["energy"].to_numpy()
    log_g = dos["entropy"].to_numpy()
    beta = 1.0 / (kB * T_arr)                             # (n_T,)
    log_w = log_g[:, None] - beta[None, :] * E[:, None]   # (n_E, n_T)
    log_w -= log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w)
    Z = w.sum(axis=0)
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
