"""Wang-Landau collective coordinator: policy types and pure decision functions.

Holds the policy that decides per-block coordinator actions (collective
halving, entropy merging, BP-switch). Pure: no dependency on `wl_replica`,
backends, or IPC. Consumed by both `SerialWangLandauPool` (via
`WangLandauWindowGroup.apply_plan`) and `ProcessWangLandauPool` (via
batched IPC apply rounds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED = (
    "checkpointing is not yet supported for n_walkers_per_window > 1; "
    "pass data_container_file=None and avoid save_checkpoint() / "
    "attach_checkpoint_writer() when using multiple walkers per window."
)


@dataclass(frozen=True, slots=True)
class WalkerPostBlockState:
    """State a worker reports after each ``ADVANCE``.

    Captured from a single worker reply at one MC step so the
    coordinator can use them as a consistent snapshot. Read by the
    coordinator to decide whether to halve, merge entropies, or flip
    the BP phase.
    """

    is_flat: bool
    fill_factor: float
    entropy: dict[int, float]
    step: int
    window_entry_step: int | None
    histogram: dict[int, int]
    reached_energy_window: bool

FlatnessMode = Literal["per_walker", "pooled"]
"""Flatness gate mode for collective halving.

- ``"per_walker"``: halve when every walker is independently flat
  (published Vogel et al. 2013).
- ``"pooled"``: halve when the summed histogram across walkers is flat
  (pooled: a single combined bin sees ``W x`` as many samples as any
  individual walker's bin under the same wall-clock budget). Default.
"""


MergeCadence = Literal["at_halve", "never"]
"""Cadence at which walker entropies are merged in the halving phase.

- ``"at_halve"``: merge entropies at each collective halve event
  (Vogel et al. 2013). Default.
- ``"never"``: no mid-run merge; walkers run fully independently.

In the 1/t phase, no mid-run merge fires regardless of this setting.
"""


_VALID_FLATNESS_MODES: tuple[str, ...] = ("per_walker", "pooled")
_VALID_MERGE_CADENCES: tuple[str, ...] = ("at_halve", "never")


def _validate_flatness_mode(flatness_mode: Any) -> None:
    if flatness_mode not in _VALID_FLATNESS_MODES:
        raise ValueError(
            f"flatness_mode must be one of {_VALID_FLATNESS_MODES}; "
            f"got {flatness_mode!r}"
        )


def _validate_merge_cadence(merge_cadence: Any) -> None:
    if merge_cadence not in _VALID_MERGE_CADENCES:
        raise ValueError(
            f"merge_cadence must be one of {_VALID_MERGE_CADENCES}; "
            f"got {merge_cadence!r}"
        )


@dataclass(frozen=True, slots=True)
class SlotView:
    """Read-only view of one slot at one decision point.

    Pure data. Built by each backend's collect step from per-walker
    snapshots plus the slot's scalar configuration. Consumed by
    ``decide_block_actions``.
    """

    walker_states: tuple[WalkerPostBlockState, ...]
    phase: str
    flatness_mode: FlatnessMode
    merge_cadence: MergeCadence
    schedule: str
    flatness_limit: float

    @property
    def n_walkers(self) -> int:
        return len(self.walker_states)


@dataclass(frozen=True, slots=True)
class CoordinatorPlan:
    """Actions decided for one slot in one block.

    ``halve``: every walker's fill factor should be halved and histogram
    reset.
    ``merged_entropy``: if not None, written into every walker's
    ``_entropy``.
    ``switch_to_phase``: if not None (currently always ``"1_over_t"``
    when set), the slot's phase is flipped and per-walker
    ``_fill_factor`` is set to ``1/t``.
    """

    halve: bool
    merged_entropy: dict[int, float] | None
    switch_to_phase: str | None


def _summed_histogram_flat_from_snapshots(
    snapshots: list[WalkerPostBlockState],
    flatness_limit: float,
) -> bool:
    """Snapshot-based pooled flatness for use by the process pool.

    Pools the per-walker histograms carried by each ``WalkerPostBlockState``
    and applies the flatness criterion: every bin count must be at least
    ``flatness_limit * mean(counts)``. Returns False if any walker has not
    yet entered its window.
    """
    if not snapshots:
        return False
    if not all(s.reached_energy_window for s in snapshots):
        return False
    combined: dict[int, int] = {}
    for s in snapshots:
        for k, v in s.histogram.items():
            combined[k] = combined.get(k, 0) + v
    if not combined:
        return False
    counts = np.array(list(combined.values()))
    limit = flatness_limit * float(np.average(counts))
    return bool(np.all(counts >= limit))


def _compute_per_walker_flat_min(
    histograms: list[dict[int, int]],
) -> float | None:
    """Min over walkers of ``min(H_k) / mean(H_k)``.

    Returns ``None`` if any walker has an empty histogram or a
    zero-mean histogram. Used by ``window_stats``/``per_window_stats``
    to compute the gate-relevant flat_min for ``flatness_mode="per_walker"``.
    """
    per_walker: list[float] = []
    for h in histograms:
        if not h:
            return None
        counts = np.array(list(h.values()), dtype=float)
        mean_c = float(counts.mean())
        if mean_c <= 0:
            return None
        per_walker.append(float(counts.min()) / mean_c)
    return min(per_walker) if per_walker else None


def decide_bp_switch(
    phases: list[str], ts: list[int], fs: list[float]
) -> bool:
    """Return ``True`` iff every walker should flip to the 1/t phase.

    The collective Belardinelli-Pereyra switch fires when every walker
    is still in the halving phase and every walker satisfies
    ``1/t > f``.

    Args:
        phases: per-walker ``_phase`` strings.
        ts: per-walker ``step - _window_entry_step + 1``.
        fs: per-walker ``_fill_factor`` after the collective halve.
    """
    if not phases:
        return False
    if any(p != "halving" for p in phases):
        return False
    return all((1.0 / t) > f for t, f in zip(ts, fs, strict=True))


def merge_entropies(
    entropies: list[dict[int, float]],
) -> dict[int, float]:
    """Combine per-walker entropy dicts into a single window estimate.

    Each walker's ``_entropy`` carries a private additive constant (from
    icet's periodic min-shift in ``_update_entropy`` and from
    independent accumulation between merges). Naive averaging would
    combine values whose additive constants differ; this function
    aligns walkers first by subtracting each walker's mean computed
    over the *common* set of bins all walkers visited, then averages
    bin-wise over the walkers that visited each bin. The result is
    finally shifted so ``min(merged) == 0`` (the icet convention,
    matching ``_update_entropy``'s periodic reshift).

    Args:
        entropies: list of ``{bin_index: entropy_value}`` dicts from
            each walker. Empty dicts (walkers that have not entered
            their window) are filtered out before processing.

    Returns:
        Merged entropy dict with ``min(merged) == 0``. Empty if no
        walker has entered the window.

    Raises:
        RuntimeError: if no bin is visited by every (filtered) walker,
            so rebasing across walkers is ill-defined.
    """
    # Filter out walkers with empty entropy dicts.
    visited = [e for e in entropies if e]
    if not visited:
        return {}
    if len(visited) == 1:
        # Single-walker fast path; min-shift to icet convention.
        only = visited[0]
        shift = min(only.values())
        return {b: v - shift for b, v in only.items()}

    # Intersection of bins visited by every walker.
    common = set(visited[0].keys())
    for e in visited[1:]:
        common &= e.keys()
    if not common:
        raise RuntimeError(
            "merge_entropies: walker coverage has no common bin; "
            "cannot rebase across walkers."
        )

    # Per-walker mean over the common bins; subtract from each walker.
    rebased: list[dict[int, float]] = []
    for e in visited:
        offset = sum(e[b] for b in common) / len(common)
        rebased.append({b: v - offset for b, v in e.items()})

    # Bin-wise average over walkers that visited each bin.
    all_bins: set[int] = set()
    for r in rebased:
        all_bins.update(r.keys())
    merged: dict[int, float] = {}
    for b in all_bins:
        contributors = [r[b] for r in rebased if b in r]
        merged[b] = sum(contributors) / len(contributors)

    # Post-shift to icet convention: min(merged) == 0.
    shift = min(merged.values())
    return {b: v - shift for b, v in merged.items()}


def decide_block_actions(view: SlotView) -> CoordinatorPlan:
    """Decide the per-block coordinator actions for one slot.

    Pure function. Reads only ``view``; returns a ``CoordinatorPlan``
    describing whether to halve, what entropy to write back (if any),
    and whether to flip to the 1/t phase.

    In the 1/t phase no mid-run merge fires regardless of
    ``view.merge_cadence`` — walker entropies are reconciled only at
    end-of-run via ``finalise_for_reporting``.
    """
    if view.phase != "halving":
        return CoordinatorPlan(
            halve=False, merged_entropy=None, switch_to_phase=None
        )

    if view.flatness_mode == "per_walker":
        should_halve = all(s.is_flat for s in view.walker_states)
    else:  # pooled
        should_halve = _summed_histogram_flat_from_snapshots(
            list(view.walker_states), view.flatness_limit
        )

    if not should_halve:
        return CoordinatorPlan(
            halve=False, merged_entropy=None, switch_to_phase=None
        )

    merged_entropy: dict[int, float] | None = None
    if view.merge_cadence == "at_halve" and view.n_walkers > 1:
        merged_entropy = merge_entropies(
            [dict(s.entropy) for s in view.walker_states]
        )

    switch_to_phase: str | None = None
    if view.schedule == "1_over_t":
        unentered = any(
            s.window_entry_step is None for s in view.walker_states
        )
        if not unentered:
            phases = ["halving"] * view.n_walkers
            ts = [
                s.step - s.window_entry_step + 1  # type: ignore[operator]
                for s in view.walker_states
            ]
            post_halve_fs = [
                s.fill_factor / 2.0 for s in view.walker_states
            ]
            if decide_bp_switch(phases, ts, post_halve_fs):
                switch_to_phase = "1_over_t"

    return CoordinatorPlan(
        halve=True,
        merged_entropy=merged_entropy,
        switch_to_phase=switch_to_phase,
    )
