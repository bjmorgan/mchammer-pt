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

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from mchammer.data_containers.base_data_container import BaseDataContainer

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

        for b, cnt, sv, sv2, sv4 in zip(bins, counts, sums, sum2s, sum4s, strict=True):
            energy = b * energy_spacing
            # Apply window filter: both bounds are inclusive.
            if lo is not None and energy < lo:
                continue
            if hi is not None and energy > hi:
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
