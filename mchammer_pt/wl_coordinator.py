"""Wang-Landau collective coordinator: policy types and pure decision functions.

The per-block coordinator decision (collective halving, entropy
merging, BP-switch) is expressed as a pure function over a frozen
data view. No dependency on backends, replicas, or IPC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

_UNSET: Any = object()
"""Sentinel for "key absent" where ``None`` is a legitimate value."""


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
    current_energy: float

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


OneOverTGate = Literal["visit_once", "flatness"]
"""1/t-schedule halving-phase gate.

- ``"visit_once"`` (default): halve once every bin has been visited,
  with the coupled BP switch.
- ``"flatness"``: halve on the WL flatness criterion
  (``min(H) >= flatness_limit * mean(H)``), bundled with the decoupled
  stall-safe switch. Only consulted under the ``"1_over_t"`` schedule.
"""


OneOverTEntry = Literal["window_clock", "f_continuous"]
"""How a walker's fill factor enters the 1/t phase at the BP switch.

- ``"window_clock"`` (default): the 1/t clock runs from
  ``_window_entry_step``; at the switch ``f`` jumps to
  ``1/(step - window_entry + 1)``.
- ``"f_continuous"``: the 1/t clock starts so that ``1/t_eff`` equals
  the fill factor halving actually reached; ``f`` is continuous across
  the switch. Applies at every switch path (canonical and stall,
  coupled and decoupled).
"""


_VALID_FLATNESS_MODES: tuple[str, ...] = ("per_walker", "pooled")
_VALID_MERGE_CADENCES: tuple[str, ...] = ("at_halve", "never")
_VALID_ONE_OVER_T_GATES: tuple[str, ...] = ("visit_once", "flatness")
_VALID_ONE_OVER_T_ENTRIES: tuple[str, ...] = ("window_clock", "f_continuous")


def _validate_one_over_t_gate(one_over_t_gate: Any) -> None:
    if one_over_t_gate not in _VALID_ONE_OVER_T_GATES:
        raise ValueError(
            f"one_over_t_gate must be one of {_VALID_ONE_OVER_T_GATES}; "
            f"got {one_over_t_gate!r}"
        )


def _validate_one_over_t_entry(one_over_t_entry: Any) -> None:
    if one_over_t_entry not in _VALID_ONE_OVER_T_ENTRIES:
        raise ValueError(
            f"one_over_t_entry must be one of {_VALID_ONE_OVER_T_ENTRIES}; "
            f"got {one_over_t_entry!r}"
        )


def _validate_bp_stall_multiple(bp_stall_multiple: Any) -> float:
    if (
        isinstance(bp_stall_multiple, bool)
        or not isinstance(bp_stall_multiple, (int, float, np.integer, np.floating))
        or not math.isfinite(bp_stall_multiple)
        or float(bp_stall_multiple) <= 0.0
    ):
        raise ValueError(
            f"bp_stall_multiple must be a finite positive number; "
            f"got {bp_stall_multiple!r}"
        )
    return float(bp_stall_multiple)


def _validate_gate_schedule(one_over_t_gate: Any, schedule: Any) -> None:
    """Reject the flatness gate selected without the 1/t schedule.

    ``one_over_t_gate='flatness'`` only affects the ``'1_over_t'`` schedule;
    under ``'halving'`` it would silently do nothing. Raising at construction
    surfaces the misconfiguration rather than letting it become a silent
    no-op run (the worst outcome for an A/B study).
    """
    if one_over_t_gate == "flatness" and schedule != "1_over_t":
        raise ValueError(
            "one_over_t_gate='flatness' requires the 1/t schedule; pass "
            "ensemble_kwargs={'schedule': '1_over_t'} to use the flatness "
            "gate, or leave one_over_t_gate='visit_once'."
        )


def _validate_entry_schedule(one_over_t_entry: Any, schedule: Any) -> None:
    """Reject f-continuous entry selected without the 1/t schedule.

    ``one_over_t_entry='f_continuous'`` only affects the ``'1_over_t'``
    schedule; under ``'halving'`` no switch ever fires, so it would
    silently do nothing. Raising at construction surfaces the
    misconfiguration rather than letting it become a silent no-op run
    (the worst outcome for an A/B study).
    """
    if one_over_t_entry == "f_continuous" and schedule != "1_over_t":
        raise ValueError(
            "one_over_t_entry='f_continuous' requires the 1/t schedule; "
            "pass ensemble_kwargs={'schedule': '1_over_t'} to use "
            "f-continuous entry, or leave one_over_t_entry='window_clock'."
        )


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
    one_over_t_gate: OneOverTGate = "visit_once"
    bp_stall_multiple: float = 4.0
    last_halve_step: int | None = None
    first_halve_duration: int | None = None

    @property
    def n_walkers(self) -> int:
        return len(self.walker_states)

    @property
    def walker_ts(self) -> list[int]:
        """Per-walker ``t = step - window_entry_step + 1`` (BP time since entry).

        Only meaningful once every walker has entered its window; callers
        must guard on ``window_entry_step is not None`` first.
        """
        return [
            s.step - s.window_entry_step + 1  # type: ignore[operator]
            for s in self.walker_states
        ]


@dataclass(frozen=True, slots=True)
class CoordinatorPlan:
    """Actions decided for one slot in one block.

    ``halve``: every walker's fill factor should be halved and histogram
    reset.
    ``merged_entropy``: if not None, written into every walker's
    ``_entropy``.
    ``switch_to_phase``: if not None, the slot's phase is flipped and
    each walker applies its 1/t entry policy
    (``WangLandauReplica.switch_to_phase``): under ``window_clock``
    ``_fill_factor`` jumps to ``1/t``; under ``f_continuous`` the
    schedule-clock origin is recorded and ``_fill_factor`` is left
    unchanged.
    """

    halve: bool
    merged_entropy: dict[int, float] | None
    switch_to_phase: Literal["1_over_t"] | None


def _summed_histogram_halving_criterion_met(
    snapshots: list[WalkerPostBlockState],
    flatness_limit: float,
    schedule: Schedule,
    one_over_t_gate: OneOverTGate = "visit_once",
) -> bool:
    """Pooled-histogram halving criterion across per-walker snapshots.

    Under ``schedule='halving'`` applies the WL flatness criterion:
    every bin count must be at least ``flatness_limit * mean(counts)``.
    Under ``schedule='1_over_t'`` the gate depends on ``one_over_t_gate``:
    ``'visit_once'`` applies the BP coupon-collector criterion (every bin
    count positive, ``flatness_limit`` not consulted); ``'flatness'``
    applies the same WL flatness criterion as the halving schedule.
    Returns False if any walker has not yet entered its window.
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
    if schedule == "1_over_t" and one_over_t_gate == "visit_once":
        # Belardinelli-Pereyra coupon-collector criterion across the
        # pooled histogram. flatness_limit is not consulted here.
        return bool(np.all(counts > 0))
    limit = flatness_limit * mean_count
    return bool(np.all(counts >= limit))


def _walker_flatness_met(
    state: WalkerPostBlockState, flatness_limit: float
) -> bool:
    """WL flatness criterion for one walker's histogram.

    ``min(H) >= flatness_limit * mean(H)`` over the walker's known bins.
    Returns ``False`` for a walker that has not entered its window or has
    an empty / zero-mean histogram (mirrors the pooled gate's guards).
    """
    if not state.reached_energy_window or not state.histogram:
        return False
    counts = np.array(list(state.histogram.values()))
    mean_count = float(np.average(counts))
    if mean_count <= 0:
        return False
    return bool(np.all(counts >= flatness_limit * mean_count))


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


def _min_over_mean(weights: dict[int, float]) -> float | None:
    """``min/mean`` over a weight dict, or ``None`` if empty/zero-mean."""
    if not weights:
        return None
    arr = np.array(list(weights.values()), dtype=float)
    mean_w = float(arr.mean())
    if mean_w <= 0:
        return None
    return float(arr.min()) / mean_w


def _compute_recency_flatness(
    weights: list[dict[int, float]],
    mode: FlatnessMode,
) -> float | None:
    """Recency flatness across per-walker EWMA weight dicts.

    ``"pooled"``: ``min/mean`` over the summed weights (collective
    sampling). ``"per_walker"``: minimum over walkers of each walker's
    ``min/mean``. Returns ``None`` if no walker has usable weights.
    """
    if not weights:
        return None
    if mode == "per_walker":
        per_walker = [_min_over_mean(w) for w in weights]
        present = [v for v in per_walker if v is not None]
        return min(present) if present else None
    summed: dict[int, float] = {}
    for w in weights:
        for b, v in w.items():
            summed[b] = summed.get(b, 0.0) + v
    return _min_over_mean(summed)


def _resolve_recency_flatness(
    stats: dict[str, Any], mode: FlatnessMode
) -> None:
    """Collapse the two candidate recency flatnesses into one value.

    Multi-walker ``window_stats`` dicts carry both
    ``recency_flatness_pooled`` and ``recency_flatness_per_walker``;
    the pool calls this once it knows the mode. Single-walker dicts
    already carry ``recency_flatness`` and are left untouched.
    """
    pooled = stats.pop("recency_flatness_pooled", _UNSET)
    per_walker = stats.pop("recency_flatness_per_walker", _UNSET)
    if pooled is _UNSET:
        return
    stats["recency_flatness"] = (
        per_walker if mode == "per_walker" else pooled
    )


def _compute_filled_bins(
    histograms: list[dict[int, int]],
    mode: FlatnessMode,
) -> int:
    """Count gate-covered bins in the current histogram for ``mode``.

    A bin is "covered" when it has a positive count. Under
    ``"pooled"`` the union across walkers is taken (a bin counts if
    any walker has a positive count for it, matching the
    summed-histogram gate); under ``"per_walker"`` the intersection
    is taken (a bin counts only if every walker has a positive count
    for it, matching the every-walker-flat gate). For a single
    histogram the two modes coincide. Returns 0 for empty input.
    """
    positive_sets = [
        {b for b, c in h.items() if c > 0} for h in histograms
    ]
    if not positive_sets:
        return 0
    if mode == "per_walker":
        return len(set.intersection(*positive_sets))
    return len(set.union(*positive_sets))


def _compute_per_walker_breakdown(
    histograms: list[dict[int, int]],
) -> list[dict[str, Any]]:
    """Per-walker ``filled``/``known``/``flat_min`` triples.

    ``filled`` is the count of bins with a positive count, ``known``
    the number of bins present, and ``flat_min`` is
    ``min(counts) / mean(counts)`` (``None`` for an empty or
    zero-mean histogram).
    """
    breakdown: list[dict[str, Any]] = []
    for h in histograms:
        flat_min: float | None
        if h:
            counts = np.array(list(h.values()), dtype=float)
            mean_c = float(counts.mean())
            flat_min = (
                float(counts.min() / mean_c) if mean_c > 0 else None
            )
        else:
            flat_min = None
        breakdown.append(
            {
                "filled": sum(1 for c in h.values() if c > 0),
                "known": len(h),
                "flat_min": flat_min,
            }
        )
    return breakdown


def _resolve_bins_filled(stats: dict[str, Any], mode: FlatnessMode) -> None:
    """Collapse the two candidate filled counts into ``bins_filled``.

    Multi-walker ``window_stats`` dicts carry both
    ``bins_filled_pooled`` and ``bins_filled_per_walker`` (the group
    is mode-agnostic); the pool calls this once it knows the mode.
    Single-walker dicts already carry ``bins_filled`` and are left
    untouched.
    """
    pooled = stats.pop("bins_filled_pooled", None)
    per_walker = stats.pop("bins_filled_per_walker", None)
    if pooled is None:
        return
    stats["bins_filled"] = per_walker if mode == "per_walker" else pooled


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


def _decoupled_switch_to_phase(
    view: SlotView, should_halve: bool
) -> Literal["1_over_t"] | None:
    """Decoupled BP switch: canonical crossing OR stall escape.

    Collective: returns ``"1_over_t"`` only if every walker qualifies.
    Blocked while any walker is unentered.

    Unlike the coupled switch, this is evaluated every block, so a window
    can enter the 1/t phase without a halve firing (via the stall escape).
    """
    states = view.walker_states
    if not states:
        return None
    if any(s.window_entry_step is None for s in states):
        return None
    # Walkers in a slot run in lockstep (one collective advance per block),
    # so they share a single step and fill factor; read them from walker 0.
    f_now = states[0].fill_factor
    ts = view.walker_ts
    # (a) canonical crossing (post-halve f when a halve fires this block).
    f_ref = (f_now / 2.0) if should_halve else f_now
    canonical = all((1.0 / t) > f_ref for t in ts)
    # (b) stall escape: only for a slot that has halved at least once.
    escape = False
    if view.last_halve_step is not None and view.first_halve_duration is not None:
        continuity = all((1.0 / t) <= f_now for t in ts)
        since = states[0].step - view.last_halve_step
        stall = since > view.bp_stall_multiple * view.first_halve_duration
        escape = continuity and stall
    return "1_over_t" if (canonical or escape) else None


def _decide_flatness_decoupled(view: SlotView) -> CoordinatorPlan:
    """1/t-schedule flatness gate plus the decoupled switch.

    Runs the flatness halving gate and delegates the switch decision to
    :func:`_decoupled_switch_to_phase` on every block.
    """
    if view.flatness_mode == "per_walker":
        should_halve = all(
            _walker_flatness_met(s, view.flatness_limit)
            for s in view.walker_states
        )
    else:
        should_halve = _summed_histogram_halving_criterion_met(
            list(view.walker_states),
            view.flatness_limit,
            view.schedule,
            one_over_t_gate="flatness",
        )
    switch_to_phase = _decoupled_switch_to_phase(view, should_halve)
    merged_entropy: dict[int, float] | None = None
    if should_halve and view.merge_cadence == "at_halve" and view.n_walkers > 1:
        merged_entropy = merge_entropies(
            [dict(s.entropy) for s in view.walker_states]
        )
    return CoordinatorPlan(
        halve=should_halve,
        merged_entropy=merged_entropy,
        switch_to_phase=switch_to_phase,
    )


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

    if view.schedule == "1_over_t" and view.one_over_t_gate == "flatness":
        return _decide_flatness_decoupled(view)

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
            ts = view.walker_ts
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


def reconstruct_stall_state(
    fill_factor_history_keys: list[Any],
    window_entry_steps: list[int | None],
) -> tuple[int | None, int | None]:
    """Rebuild ``(last_halve_step, first_halve_duration)`` from saved state.

    ``_fill_factor_history`` keys are the initial entry plus one per
    collective halve, so the sorted keys after the first are the halve
    steps. ``T1`` (first-stage duration) is the first halve step minus the
    latest walker ``window_entry_step`` in the slot. Returns ``(None, None)``
    when the slot has not halved (so the stall escape stays disarmed).
    Used on resume; the live run tracks these incrementally.

    ``window_entry_step`` is only recorded under the ``1_over_t`` schedule,
    so a slot that halved under the ``halving`` schedule legitimately has no
    entries; that yields ``(last_halve_step, None)`` (disarmed), which is
    correct because the halving schedule has no decoupled switch.

    A first-stage duration of zero is valid: a walker can enter on the
    final step of its first halving block, so the first halve and the entry
    share a step (the live backend records ``step - entry == 0`` there).

    Raises:
        ValueError: if the first halve *precedes* the latest window entry (a
            negative first-stage duration). A halve cannot be recorded before
            a walker enters, so this is physically impossible in a correctly
            round-tripped checkpoint and signals a corrupted or truncated one.
    """
    keys = sorted(int(k) for k in fill_factor_history_keys)
    halve_steps = keys[1:]
    if not halve_steps:
        return (None, None)
    entries = [int(e) for e in window_entry_steps if e is not None]
    if not entries:
        return (halve_steps[-1], None)
    t1 = halve_steps[0] - max(entries)
    if t1 < 0:
        raise ValueError(
            f"reconstruct_stall_state: first halve at step {halve_steps[0]} "
            f"precedes the latest window entry {max(entries)} "
            f"(first-stage duration {t1} < 0); corrupted checkpoint."
        )
    return (halve_steps[-1], t1)
