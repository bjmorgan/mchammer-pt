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
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Stitch per-window entropy curves into a single density of states.

    Windows are sorted by minimum energy, shifted purely additively so
    that overlap regions align by mean entropy difference, and averaged
    where they overlap. The returned ``entropy`` column is globally
    rebased so that its minimum is zero (``ln g`` is only defined up to
    a single additive constant).

    Args:
        per_window: list of DataFrames each carrying ``energy`` and
            ``entropy`` columns. ``entropy`` is treated as ``ln g``.

    Returns:
        ``(stitched, overlap_errors)`` where ``stitched`` is a DataFrame
        with ``energy`` and ``entropy`` columns (rebased to ``min = 0``)
        and ``overlap_errors`` is a dict of overlap-region entropy
        standard deviations keyed by ``"i-j"`` window-pair labels in
        the original input order.

    Raises:
        ValueError: if any pair of neighbouring windows in the sorted
            order has no overlapping energy range, or shares no bin
            centres within the overlap.
    """
    ordered = sorted(
        enumerate(per_window),
        key=lambda t: t[1]["energy"].iloc[0],
    )

    errors: dict[str, float] = {}
    for k in range(1, len(ordered)):
        idx_l, df_l = ordered[k - 1]
        idx_r, df_r = ordered[k]
        left_lim = df_r["energy"].min()
        right_lim = df_l["energy"].max()
        if left_lim >= right_lim:
            raise ValueError(
                f"No overlap between windows {idx_l} and {idx_r}: "
                f"right edge {right_lim} <= left edge {left_lim}"
            )
        ol_l = df_l[(df_l["energy"] >= left_lim) & (df_l["energy"] <= right_lim)]
        ol_r = df_r[(df_r["energy"] >= left_lim) & (df_r["energy"] <= right_lim)]
        shared = np.intersect1d(
            ol_l["energy"].to_numpy(),
            ol_r["energy"].to_numpy(),
        )
        if shared.size == 0:
            raise ValueError(
                f"No shared bin centres between windows {idx_l} and {idx_r}: "
                f"ranges overlap on [{left_lim}, {right_lim}] but no bin centres "
                f"coincide. Are the windows on the same energy grid?"
            )
        s_l = ol_l.set_index("energy").loc[shared, "entropy"]
        s_r = ol_r.set_index("energy").loc[shared, "entropy"]
        offset = float((s_r - s_l).mean())
        errors[f"{idx_l}-{idx_r}"] = float((s_r - s_l).std())
        df_r = df_r.copy()
        df_r["entropy"] = df_r["entropy"] - offset
        ordered[k] = (idx_r, df_r)

    stacked = pd.concat([df_w for _, df_w in ordered], ignore_index=True)
    merged = (
        stacked.groupby("energy", sort=True)["entropy"]
        .mean()
        .reset_index()
    )
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
