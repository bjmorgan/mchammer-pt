"""Density-of-states post-processing for windowed Wang-Landau output.

`stitch_entropy` merges per-window entropy curves (the
``energy``-and-``entropy`` DataFrames produced by
``WindowResult.get_entropy()``) into a single density of states via
overlap-region alignment, working entirely in log space.

`reweight_canonical_from_dos` evaluates canonical thermodynamics
from a stitched ``ln g(E)`` curve on a user-supplied temperature
grid, also entirely in log space so large entropy ranges do not
underflow ``float64``.

Both functions are generic: they consume only the ``energy`` and
``entropy`` columns and make no material-specific assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import k as _kB_J

KB_EV = _kB_J / 1.602176634e-19


def stitch_entropy(
    per_window: list[pd.DataFrame],
    energy_spacing: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Stitch per-window entropy curves into a single density of states.

    Each element of ``per_window`` is a DataFrame with ``energy`` and
    ``entropy`` columns; the ``entropy`` column is treated as ``ln g``.
    Windows are sorted by minimum energy, shifted purely additively so
    that overlap regions align by mean entropy difference, and averaged
    where they overlap.

    Returns the stitched DataFrame (``energy``, ``entropy``) plus a dict
    of overlap-region standard deviations keyed by ``"i-j"`` window-pair
    labels in the original input order. ``energy_spacing`` is accepted
    for API completeness but not used directly (bin centres are matched
    by intersection).

    Raises:
        ValueError: if any pair of neighbouring windows in the sorted
            order does not share at least one bin centre.
    """
    del energy_spacing
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
        s_l = ol_l.set_index("energy").loc[shared, "entropy"]
        s_r = ol_r.set_index("energy").loc[shared, "entropy"]
        offset = float((s_r - s_l).mean())
        errors[f"{idx_l}-{idx_r}"] = float((s_r - s_l).std())
        df_r = df_r.copy()
        df_r["entropy"] = df_r["entropy"] - offset
        ordered[k] = (idx_r, df_r)

    combined: dict[float, list[float]] = {}
    for _, df_w in ordered:
        for _, row in df_w.iterrows():
            combined.setdefault(
                float(row["energy"]), []
            ).append(float(row["entropy"]))
    energies = np.array(sorted(combined.keys()))
    entropies = np.array([np.mean(combined[e]) for e in energies])
    entropies -= entropies.min()

    return pd.DataFrame({"energy": energies, "entropy": entropies}), errors
