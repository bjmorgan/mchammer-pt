"""Post-processing for per-walker observable moment stores.

`stitch_observable_moments` reads ``observable_records`` from each
container's ``_last_state``, filters each walker's contributions to
bins inside that walker's own energy window, and sums moments bin-wise
across all walkers.  Moments are extensive: the merge is plain addition
with no additive rebasing.

One ``pd.DataFrame`` per observer tag is returned, on the energy grid
with columns ``energy``, ``count``, and per-scalar ``{name}_sum``,
``{name}_sum2``, ``{name}_sum4``.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from mchammer.data_containers.base_data_container import BaseDataContainer

from mchammer_pt.analysis.dos import _canonical_log_weights

# One walker's serialised store paired with its (lo, hi) energy window.
_WalkerStore = tuple[dict[str, Any], tuple[float | None, float | None]]


def stitch_observable_moments(
    containers: list[BaseDataContainer],
    energy_spacing: float,
) -> dict[str, pd.DataFrame]:
    """Sum per-walker observable moment stores bin-wise onto the energy grid.

    For each observer tag present across the supplied containers, collects
    all walkers' ``to_state()`` stores, restricts each walker's contribution
    to bins that lie inside that walker's own energy window
    (``lo <= bin_index * energy_spacing <= hi``, with ``None`` meaning
    unbounded), and sums ``count``/``sum``/``sum2``/``sum4`` bin-wise across
    all contributing walkers.  Bins in the overlap region legitimately
    accumulate from multiple walkers.

    Moments are extensive: the merge is plain addition; no additive
    rebasing is performed (unlike ``ln g`` stitching).

    Args:
        containers: Sequence of ``BaseDataContainer`` instances whose
            ``_last_state["observable_records"]`` dicts carry per-walker
            ``EnergyBinnedObservableRecorder.to_state()`` results.
            Containers that lack the key contribute nothing.
        energy_spacing: Bin spacing (eV).  Bin energy = ``bin * energy_spacing``.

    Returns:
        A dict mapping each observer tag to a ``pd.DataFrame`` with columns
        ``energy``, ``count``, and for each scalar name ``n``:
        ``{n}_sum``, ``{n}_sum2``, ``{n}_sum4``.  The DataFrame is sorted
        by energy ascending and contains only bins with ``count > 0``.
        Returns an empty dict when no containers carry observable records.

    Raises:
        ValueError: If two walkers' stores for the same tag disagree on
            observable names or size (inconsistent observer signature).
    """
    # Collect (store, window_limits) pairs keyed by tag.
    # tag -> list of (state_dict, (lo, hi))
    per_tag: dict[str, list[_WalkerStore]] = defaultdict(list)

    for dc in containers:
        obs_records: dict[str, dict[str, Any]] = dc._last_state.get(
            "observable_records", {}
        )
        params = dc.ensemble_parameters
        lo: float | None = params.get("energy_limit_left")
        hi: float | None = params.get("energy_limit_right")
        for tag, state in obs_records.items():
            per_tag[tag].append((state, (lo, hi)))

    if not per_tag:
        return {}

    result: dict[str, pd.DataFrame] = {}
    for tag, walker_stores in per_tag.items():
        df = _merge_tag(tag, walker_stores, energy_spacing)
        if df is not None:
            result[tag] = df

    return result


def _merge_tag(
    tag: str,
    walker_stores: list[_WalkerStore],
    energy_spacing: float,
) -> pd.DataFrame | None:
    """Merge all walkers' stores for a single tag into one DataFrame.

    Args:
        tag: Observer tag (used in error messages).
        walker_stores: List of ``(state_dict, (lo, hi))`` pairs.
        energy_spacing: Bin spacing.

    Returns:
        Merged DataFrame, or ``None`` if no in-window bins had any counts.

    Raises:
        ValueError: If two walkers disagree on observable names or size.
    """
    # Determine the shared names from the first store that has them.
    names: tuple[str, ...] | None = None
    for state, _ in walker_stores:
        stored_names = state.get("names", [])
        if stored_names:
            names = tuple(stored_names)
            break

    if names is None:
        # All stores have empty names lists (no observations recorded anywhere).
        return None

    # Validate that all walkers agree on names.
    for state, _ in walker_stores:
        stored_names = tuple(state.get("names", []))
        if stored_names and stored_names != names:
            raise ValueError(
                f"tag {tag!r}: inconsistent observer signature across walkers — "
                f"expected names {names!r}, got {stored_names!r}"
            )

    s = len(names)

    # Accumulate bin-wise sums.  Keys are bin indices; values are arrays of shape (s,).
    total_count: dict[int, int] = defaultdict(int)
    total_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(s))
    total_sum2: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(s))
    total_sum4: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(s))

    for state, (lo, hi) in walker_stores:
        bins: list[int] = list(state.get("bins", []))
        counts: list[int] = list(state.get("count", []))
        sums: list[list[float]] = list(state.get("sum", []))
        sum2s: list[list[float]] = list(state.get("sum2", []))
        sum4s: list[list[float]] = list(state.get("sum4", []))

        # Window membership by bin index (round-based), matching icet's
        # `_inside_energy_window` and hence the bins the recorder actually
        # logged. A raw `energy <= hi` test could drop an edge bin that the
        # recorder counted as in-window. `None` bounds are unbounded.
        lo_bin = None if lo is None else round(lo / energy_spacing)
        hi_bin = None if hi is None else round(hi / energy_spacing)
        for b, cnt, sv, sv2, sv4 in zip(bins, counts, sums, sum2s, sum4s, strict=True):
            if lo_bin is not None and b < lo_bin:
                continue
            if hi_bin is not None and b > hi_bin:
                continue
            if cnt == 0:
                continue
            total_count[b] += cnt
            total_sum[b] = total_sum[b] + np.asarray(sv, dtype=float)
            total_sum2[b] = total_sum2[b] + np.asarray(sv2, dtype=float)
            total_sum4[b] = total_sum4[b] + np.asarray(sv4, dtype=float)

    populated_bins = [b for b, cnt in total_count.items() if cnt > 0]
    if not populated_bins:
        return None

    populated_bins.sort()

    rows: dict[str, list[Any]] = {
        "energy": [b * energy_spacing for b in populated_bins],
        "count": [total_count[b] for b in populated_bins],
    }
    for i, name in enumerate(names):
        rows[f"{name}_sum"] = [float(total_sum[b][i]) for b in populated_bins]
        rows[f"{name}_sum2"] = [float(total_sum2[b][i]) for b in populated_bins]
        rows[f"{name}_sum4"] = [float(total_sum4[b][i]) for b in populated_bins]

    return pd.DataFrame(rows)


def _infer_spacing(energies: np.ndarray) -> float:
    """Infer the uniform bin spacing from an energy grid.

    Args:
        energies: bin-centre energies (need not be sorted or unique).

    Returns:
        The smallest gap between distinct energies, which equals the
        ``energy_spacing`` for a grid of integer multiples of it.

    Raises:
        ValueError: if fewer than two distinct energies are supplied.
    """
    uniq = np.unique(np.asarray(energies, dtype=float))
    if uniq.size < 2:
        raise ValueError("cannot infer energy spacing from fewer than two bins")
    return float(np.min(np.diff(uniq)))


def reweight_observables(
    moments: pd.DataFrame,
    dos: pd.DataFrame,
    temperatures: np.ndarray,
    *,
    coverage_threshold: float = 0.99,
) -> pd.DataFrame:
    """Reweight microcanonical observable moments to canonical averages.

    For each scalar observable in ``moments``, computes ``<O>(T)``,
    ``<O^2>(T)`` and the Binder cumulant
    ``U(T) = 1 - <O^4>(T) / (3 <O^2>(T)^2)``. The microcanonical average
    per bin is ``<O^n>(E) = sum_n(E) / count(E)``; reweighting uses the
    shared Boltzmann weights ``w(E, T) = g(E) e^{-E/kT}``. Sums run over
    the bins that carry BOTH a finite ``ln g`` and at least one observable
    sample (``count > 0``), normalised by the partition sum over those same
    bins (so the result is the canonical average conditional on the sampled
    energy range).

    A coverage diagnostic reports the fraction of the full canonical weight
    captured by the sampled bins; a warning is emitted if it falls below
    ``coverage_threshold`` at any temperature, since unsampled high-weight
    bins would bias ``<O>(T)``.

    Args:
        moments: per-tag microcanonical moments with columns ``energy``,
            ``count`` and, per scalar name ``n``, ``{n}_sum``, ``{n}_sum2``,
            ``{n}_sum4`` (as written by :func:`stitch_observable_moments`).
        dos: stitched density of states with ``energy`` and ``entropy``
            (``ln g``) columns.
        temperatures: strictly-positive temperatures (K).
        coverage_threshold: warn if the sampled-bin weight fraction drops
            below this at any temperature. Defaults to 0.99.

    Returns:
        DataFrame with one row per temperature: ``T_K``, ``coverage``, and
        per scalar name ``n``: ``{n}_mean``, ``{n}_sq_mean``, ``{n}_binder``.

    Raises:
        ValueError: if ``temperatures`` has a non-positive element; if
            ``moments`` or ``dos`` is empty; if ``moments`` has no
            ``{name}_sum`` columns; or if no bin carries both a finite
            ``ln g`` and an observable sample.
    """
    if moments.empty:
        raise ValueError("moments has no rows")
    if dos.empty:
        raise ValueError("dos has no rows; need at least one energy bin")
    T_arr = np.asarray(temperatures, dtype=float)
    if np.any(T_arr <= 0.0):
        raise ValueError(
            f"temperatures must be strictly positive (K); "
            f"got min={float(T_arr.min())}"
        )

    names = [c[:-4] for c in moments.columns if c.endswith("_sum")]
    if not names:
        raise ValueError("moments has no '<name>_sum' columns")

    spacing = _infer_spacing(dos["energy"].to_numpy())
    dos_bin = np.rint(dos["energy"].to_numpy() / spacing).astype(int)
    mom_bin = np.rint(moments["energy"].to_numpy() / spacing).astype(int)

    # Full partition-function support: bins with finite ln g.
    log_g_full = dos["entropy"].to_numpy()
    finite = np.isfinite(log_g_full)
    E_all = dos["energy"].to_numpy()[finite]
    log_g_all = log_g_full[finite]
    bin_all = dos_bin[finite]

    w_all, _ = _canonical_log_weights(E_all, log_g_all, T_arr)   # (n_all, n_T)
    Z_all = w_all.sum(axis=0)

    # Map each sampled bin (count > 0) to its moments-row index.
    counts = moments["count"].to_numpy()
    sampled: dict[int, int] = {
        int(b): i for i, (b, c) in enumerate(zip(mom_bin, counts, strict=True))
        if c > 0
    }

    covered_mask = np.array([int(b) in sampled for b in bin_all], dtype=bool)
    if not covered_mask.any():
        raise ValueError(
            "no overlapping bins with both a finite ln g and an observable "
            "sample; cannot reweight"
        )

    w_cov = w_all[covered_mask]                 # (n_cov, n_T)
    Z_cov = w_cov.sum(axis=0)                    # (n_T,)
    coverage = Z_cov / Z_all                     # (n_T,)

    cov_rows = [sampled[int(b)] for b in bin_all[covered_mask]]
    cov_counts = counts[cov_rows].astype(float)  # (n_cov,)

    out: dict[str, np.ndarray] = {"T_K": T_arr, "coverage": coverage}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in names:
            o1 = moments[f"{name}_sum"].to_numpy()[cov_rows] / cov_counts
            o2 = moments[f"{name}_sum2"].to_numpy()[cov_rows] / cov_counts
            o4 = moments[f"{name}_sum4"].to_numpy()[cov_rows] / cov_counts
            o1_t = (w_cov * o1[:, None]).sum(axis=0) / Z_cov
            o2_t = (w_cov * o2[:, None]).sum(axis=0) / Z_cov
            o4_t = (w_cov * o4[:, None]).sum(axis=0) / Z_cov
            binder = 1.0 - o4_t / (3.0 * o2_t**2)
            out[f"{name}_mean"] = o1_t
            out[f"{name}_sq_mean"] = o2_t
            out[f"{name}_binder"] = binder

    if float(coverage.min()) < coverage_threshold:
        warnings.warn(
            f"observable reweighting coverage drops to "
            f"{float(coverage.min()):.3f} (< {coverage_threshold}) at some "
            f"temperature: energy bins with finite g but no observable "
            f"samples carry significant canonical weight and will bias "
            f"<O>(T). Run the measurement longer to populate those bins.",
            stacklevel=2,
        )

    return pd.DataFrame(out)
