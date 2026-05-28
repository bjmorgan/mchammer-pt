"""Wang-Landau collective coordinator: policy types and pure decision functions.

The per-block coordinator decision (collective halving, entropy
merging, BP-switch) is expressed as a pure function over a frozen
data view. No dependency on backends, replicas, or IPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class WalkerPostBlockState:
    """Snapshot of one walker's state at a single MC step."""

    halving_criterion_met: bool
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
- ``"pooled"``: halve when the summed histogram across walkers is
  flat. Default.
"""


MergeCadence = Literal["at_halve", "never"]
"""Cadence at which walker entropies are merged in the halving phase.

- ``"at_halve"``: merge entropies at each collective halve event
  (Vogel et al. 2013). Default.
- ``"never"``: no mid-run merge; walkers run fully independently.
"""


Schedule = Literal["halving", "1_over_t"]
"""Fill-factor schedule for the underlying WL ensemble.

- ``"halving"``: classic Wang-Landau; halve ``_fill_factor`` at each
  collective halve event.
- ``"1_over_t"``: Belardinelli-Pereyra 1/t schedule; switch to
  ``_fill_factor = 1/t`` once every walker satisfies ``1/t > f``.
"""


Phase = Literal["halving", "1_over_t"]
"""Current phase of the WL run.

- ``"halving"``: the collective halve gate is active.
- ``"1_over_t"``: the BP switch has fired; no more halving.
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
    """Read-only snapshot of one slot at one decision point."""

    walker_states: tuple[WalkerPostBlockState, ...]
    phase: Phase
    flatness_mode: FlatnessMode
    merge_cadence: MergeCadence
    schedule: Schedule
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
    ``switch_to_phase``: if not None, the slot's phase is flipped and
    per-walker ``_fill_factor`` is set to ``1/t``.
    """

    halve: bool
    merged_entropy: dict[int, float] | None
    switch_to_phase: Literal["1_over_t"] | None


def _summed_histogram_halving_criterion_met(
    snapshots: list[WalkerPostBlockState],
    flatness_limit: float,
    schedule: Schedule,
) -> bool:
    """Pooled-histogram halving criterion across per-walker snapshots.

    Under ``schedule='halving'`` applies the WL flatness criterion:
    every bin count must be at least ``flatness_limit * mean(counts)``.
    Under ``schedule='1_over_t'`` applies the BP coupon-collector
    criterion: every bin count must be positive. Returns False if any
    walker has not yet entered its window.
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
    mean_count = float(np.average(counts))
    if mean_count <= 0:
        return False
    if schedule == "1_over_t":
        # Belardinelli-Pereyra coupon-collector criterion across the
        # pooled histogram. flatness_limit is not consulted under
        # this schedule.
        return bool(np.all(counts > 0))
    limit = flatness_limit * mean_count
    return bool(np.all(counts >= limit))


def _compute_per_walker_flat_min(
    histograms: list[dict[int, int]],
) -> float | None:
    """Min over walkers of ``min(H_k) / mean(H_k)``.

    Returns ``None`` if any walker has an empty histogram or a
    zero-mean histogram.
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


def decide_bp_switch(ts: list[int], fs: list[float]) -> bool:
    """Return ``True`` iff every walker satisfies the BP-switch condition.

    The collective Belardinelli-Pereyra switch fires when every walker
    satisfies ``1/t > f``.

    Args:
        ts: per-walker ``step - _window_entry_step + 1``.
        fs: per-walker ``_fill_factor`` after the collective halve.
    """
    if not ts:
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
    visited = [e for e in entropies if e]
    if not visited:
        return {}
    if len(visited) == 1:
        # Single-walker fast path; min-shift to icet convention.
        only = visited[0]
        shift = min(only.values())
        return {b: v - shift for b, v in only.items()}

    common = set(visited[0].keys())
    for e in visited[1:]:
        common &= e.keys()
    if not common:
        raise RuntimeError(
            "merge_entropies: walker coverage has no common bin; "
            "cannot rebase across walkers."
        )

    rebased: list[dict[int, float]] = []
    for e in visited:
        offset = sum(e[b] for b in common) / len(common)
        rebased.append({b: v - offset for b, v in e.items()})

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

    Pure function. Returns a ``CoordinatorPlan`` describing whether
    to halve, what entropy to write back (if any), and whether to
    flip to the 1/t phase.
    """
    if view.phase != "halving":
        return CoordinatorPlan(
            halve=False, merged_entropy=None, switch_to_phase=None
        )

    if view.flatness_mode == "per_walker":
        should_halve = all(
            s.halving_criterion_met for s in view.walker_states
        )
    else:  # pooled
        should_halve = _summed_histogram_halving_criterion_met(
            list(view.walker_states),
            view.flatness_limit,
            view.schedule,
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

    switch_to_phase: Literal["1_over_t"] | None = None
    if view.schedule == "1_over_t":
        unentered = any(
            s.window_entry_step is None for s in view.walker_states
        )
        if not unentered:
            ts = [
                s.step - s.window_entry_step + 1  # type: ignore[operator]
                for s in view.walker_states
            ]
            post_halve_fs = [
                s.fill_factor / 2.0 for s in view.walker_states
            ]
            if decide_bp_switch(ts, post_halve_fs):
                switch_to_phase = "1_over_t"

    return CoordinatorPlan(
        halve=True,
        merged_entropy=merged_entropy,
        switch_to_phase=switch_to_phase,
    )
