"""Wang-Landau ensemble subclass with halving delegated to a coordinator."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, cast

import numpy as np
from mchammer.ensembles import WangLandauEnsemble
from mchammer.observers.base_observer import BaseObserver

from mchammer_pt.wl_coordinator import Phase, Schedule
from mchammer_pt.wl_observable_recorder import EnergyBinnedObservableRecorder


def _validate_recency_visits_per_bin(value: object) -> int:
    """Return ``value`` as an int, or raise if it is not a positive integer.

    Rejects non-integer values (e.g. ``2.5``) and ``bool`` rather than
    silently truncating or coercing them, so the error message's promise
    of an integer holds. Integer-valued floats such as ``1e3`` are accepted.
    Accepts ``object`` because callers pass values read from checkpoint
    metadata (an untrusted union) as well as the constructor's ``int``.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or int(value) != value
        or int(value) <= 0
    ):
        raise ValueError(
            f"recency_visits_per_bin must be a positive integer; "
            f"got {value!r}"
        )
    return int(value)


def _validate_dos_snapshot_ratio(value: object) -> float | None:
    """Return ``value`` as a float ``> 1.0``, ``None`` to disable, else raise.

    Accepts ``None`` (1/t-regime snapshotting disabled) or any finite
    real strictly greater than ``1.0``. Rejects ``bool``, non-finite
    values, and ratios ``<= 1.0`` (a ratio of 1 would snapshot every
    step). Accepts ``object`` because callers pass values read from
    checkpoint metadata (an untrusted union) as well as the
    constructor's ``float | None``.
    """
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or float(value) <= 1.0
    ):
        raise ValueError(
            f"dos_snapshot_ratio must be None or a finite float > 1.0; "
            f"got {value!r}"
        )
    return float(value)


def _validate_schedule(value: object) -> Schedule:
    """Return ``value`` if it is a valid WL fill-factor schedule, else raise.

    Accepts only ``"halving"`` or ``"1_over_t"``. Validated here rather
    than forwarded to the base ``WangLandauEnsemble``: stock icet has no
    ``schedule`` parameter, so ``CoordinatedWangLandauEnsemble`` owns the
    schedule attribute on both stock and the patched fork.
    """
    if value == "halving":
        return "halving"
    if value == "1_over_t":
        return "1_over_t"
    raise ValueError(
        f"schedule must be 'halving' or '1_over_t'; got {value!r}"
    )


class CoordinatedWangLandauEnsemble(WangLandauEnsemble):  # type: ignore[misc]
    """`WangLandauEnsemble` with internal halving suppressed.

    Bin counters and periodic entropy reshift behave identically to
    upstream. The flatness check, halving, ``_fill_factor_history``
    recording, histogram reset, BP-phase transition, and ``_converged``
    writes in the 1/t branch are all suppressed; ``WangLandauWindowGroup``
    owns those decisions and applies them via ``WangLandauReplica``.

    Upstream icet's 1/t branch records ``_entropy_history`` snapshots
    autonomously; that is suppressed here. In the coordinator design the
    halving history is written only by ``force_halve`` -- one
    ``_entropy_history`` entry per collective halve, alongside the
    matching ``_fill_factor_history`` entry -- so the ensemble must not
    write either history dict itself; an autonomous 1/t write would add
    ``_entropy_history`` entries the coordinator never made. 1/t-regime
    DOS is instead recorded into a *separate* store
    (``_entropy_snapshots`` / ``_fill_factor_snapshots``) on a
    fill-factor rung ladder, leaving the halving history to the
    coordinator. ``dos_snapshot_ratio`` sets the ladder ratio (``None``
    disables); ``2.0`` snapshots each time ``f`` halves.

    Args:
        recency_visits_per_bin: EWMA recency window width in units of
            visits-per-bin. The decay rate is ``1 / (N *
            recency_visits_per_bin)`` where ``N`` is the current known
            bin count, so the effective averaging window grows with the
            number of discovered bins.
        dos_snapshot_ratio: ratio of the fill-factor ladder used to
            record 1/t-regime DOS snapshots. ``None`` disables
            snapshotting; ``2.0`` (default) records a snapshot each
            time ``f`` halves.
        frozen_g: when ``True``, ``_update_entropy`` holds ``_entropy``,
            ``_histogram``, and ``_fill_factor`` fixed for the duration
            of the run. The converged density of states is the sampling
            bias; nothing about it is mutated. The acceptance criterion
            still reads the frozen ``_entropy``, so the walk remains
            flat-in-energy without further DOS accumulation. Intended
            for observable-measurement passes after WL convergence.
            Defaults to ``False``.
        schedule: WL fill-factor update schedule. ``"halving"``
            (default) is the standard Wang-Landau halving scheme;
            ``"1_over_t"`` is the Belardinelli-Pereyra 1/t schedule. The
            run always starts in the halving phase; under ``"1_over_t"``
            the coordinator later flips to the 1/t phase via
            ``WangLandauReplica.switch_to_phase``.
    """

    def __init__(
        self,
        *args: Any,
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        frozen_g: bool = False,
        schedule: Schedule = "halving",
        **kwargs: Any,
    ) -> None:
        recency = _validate_recency_visits_per_bin(recency_visits_per_bin)
        ratio = _validate_dos_snapshot_ratio(dos_snapshot_ratio)
        validated_schedule = _validate_schedule(schedule)
        super().__init__(*args, **kwargs)
        # Schedule state. Stock icet's WangLandauEnsemble has no `schedule`
        # parameter and never sets these three attributes; the patched fork
        # does. Own them here so both behave identically -- the coordinator
        # drives every phase transition via WangLandauReplica.switch_to_phase.
        self._schedule: Schedule = validated_schedule
        self._phase: Phase = "halving"
        self._window_entry_step: int | None = None
        # When True, ``_update_entropy`` skips all DOS-mutating writes so
        # ``_entropy``, ``_histogram``, and ``_fill_factor`` are held
        # fixed for the duration of the run.  The acceptance criterion
        # still reads the frozen ``_entropy``, keeping the walk
        # flat-in-energy.
        self._frozen_g: bool = frozen_g
        # Bins the walker has reached via `_update_entropy` since
        # window entry. Populated only by that method (guarded on
        # `_reached_energy_window`).
        self._visited_bins: set[int] = set()
        # 1/t-regime DOS snapshot store, kept separate from the halving
        # history so the `len(_fill_factor_history) - 1` halving count
        # is untouched. Written by `_update_entropy` on a fill-factor
        # rung ladder; read back via `WindowResult.get_entropy`.
        self._dos_snapshot_ratio: float | None = ratio
        self._entropy_snapshots: dict[int, dict[int, float]] = {}
        self._fill_factor_snapshots: dict[int, float] = {}
        # In-memory rung tracker; not persisted, rebuilt on resume from
        # `_fill_factor_snapshots` (see `_rebuild_max_snapshot_rung`).
        self._max_snapshot_rung: int | None = None
        # EWMA recency state: per-bin weight and the step it was last
        # updated. Decayed lazily (only the visited bin is touched per
        # step; all known bins are decayed at read time).
        self._recency_visits_per_bin: int = recency
        self._recent_weight: dict[int, float] = {}
        self._recent_last_step: dict[int, int] = {}
        # Schedule-clock origin for the 1/t phase, recorded by
        # `WangLandauReplica.switch_to_phase` under
        # `one_over_t_entry='f_continuous'` so that
        # `t_eff = step - origin + 1 = ceil(1/f)` at the switch. A
        # clock origin, not a step index: it is negative when halving
        # reached small f in few steps. `None` (the `window_clock`
        # policy, and walkers restored from checkpoints carrying no
        # origin) falls back to `_window_entry_step`.
        self._one_over_t_origin_step: int | None = None
        # Tag-keyed energy-binned observable recorders. Driven from the
        # `_run` override after each trial step (post-`update_occupations`),
        # gated on `_reached_energy_window` and each recorder's `interval`.
        self._recorders: dict[str, EnergyBinnedObservableRecorder] = {}
        # Stores restored from a checkpoint that have not yet been
        # re-attached via `attach_observable_recorder`. Populated by
        # `WangLandauReplica.restore_state`; entries are consumed (popped)
        # when the matching observer is re-attached. Any entries still
        # present at `refresh_last_state` time are written back into the
        # checkpoint unchanged, so unbound stores are never dropped.
        self._restored_observable_records: dict[str, dict[str, Any]] = {}

    def attach_observable_recorder(self, observer: BaseObserver) -> None:
        """Attach an observer whose scalar outputs are accumulated per energy bin.

        The observer is wrapped in an :class:`EnergyBinnedObservableRecorder`
        and stored keyed by its tag. Recording is driven from the ``_run``
        override after each trial step (post-``update_occupations``), gated
        on ``self._reached_energy_window`` and ``observer.interval``.

        If a restored store for this tag exists in
        ``_restored_observable_records`` (populated by
        ``WangLandauReplica.restore_state``), the recorder is seeded from
        that store via :meth:`EnergyBinnedObservableRecorder.from_state`,
        which validates the observer's signature against the stored one and
        raises ``ValueError`` on disagreement. The store entry is consumed
        so it is not written back as unbound on the next checkpoint.

        Args:
            observer: Any ``mchammer.BaseObserver`` subclass whose
                ``get_observable`` returns a scalar, sequence, or Mapping.

        Raises:
            ValueError: if ``observer.interval`` is ``None``; if an observer
                with the same tag is already attached; or if a restored store
                for the tag has an incompatible signature.
        """
        if observer.interval is None:
            raise ValueError(
                f"observer {observer.tag!r} has interval=None; set a concrete "
                "interval (in MC trial steps) for measurement recording. "
                "mchammer's attach_observer resolves None to len(structure), "
                "but the energy-binned recorder needs an explicit cadence."
            )
        tag = observer.tag
        if tag in self._recorders:
            raise ValueError(
                f"an observable recorder with tag {tag!r} is already attached; "
                "use a distinct tag for each observer"
            )
        if tag in self._restored_observable_records:
            self._recorders[tag] = EnergyBinnedObservableRecorder.from_state(
                self._restored_observable_records.pop(tag), observer
            )
        else:
            self._recorders[tag] = EnergyBinnedObservableRecorder(observer)

    def _run(self, number_of_trial_steps: int) -> None:
        """Run for ``number_of_trial_steps``, driving recorders after each step.

        Overrides the parent loop to fire per-bin observable recorders
        immediately after ``_do_trial_step`` returns. At that point
        ``configuration.structure`` is always consistent with
        ``_potential``: accepted moves have already called
        ``update_occupations``; rejected moves leave both unchanged.

        Recorders are driven inside ``_run`` rather than inside
        ``_do_trial_step`` because recording must happen after
        ``update_occupations`` has committed the accepted move.
        ``_update_entropy`` (which updates the energy and bin) runs inside
        ``_acceptance_condition``, before ``update_occupations``: at that
        point the bin is already updated but the structure is not, so
        pairing them would record the new bin against the pre-move
        structure. ``_run`` is the per-step seam where both are
        guaranteed to be consistent. A cooperative ``_do_trial_step``
        override would also be correct, but hooking in ``_run`` keeps the
        ``_do_trial_step`` slot untouched and centralises recording in one
        place.
        """
        # Mirrors mchammer BaseEnsemble._run (loop + _step/_accepted_trials
        # counters only); recording is inserted per-step here.
        # Re-check this override if upstream _run gains other per-step work.
        for _ in range(number_of_trial_steps):
            # Capture the step value used inside this trial step BEFORE
            # the increment that `_run` is responsible for.
            step_at_call = int(self._step)
            accepted = self._do_trial_step()
            self._step += 1
            self._accepted_trials += accepted
            # Drive recorders once the step is complete and the structure
            # is consistent with ``_potential``. Recording is orthogonal to
            # ``frozen_g`` and runs in both modes.
            if self._reached_energy_window and self._recorders:
                due = [
                    rec
                    for rec in self._recorders.values()
                    if step_at_call % rec.interval == 0
                ]
                if due:
                    bin_cur = self._get_bin_index(self._potential)
                    structure = self.configuration.structure
                    for rec in due:
                        rec.record(structure, bin_cur)

    def _update_entropy(self, bin_cur: int) -> None:
        entry: int | None = self._window_entry_step
        if (
            self._schedule == "1_over_t"
            and self._reached_energy_window
            and entry is None
        ):
            self._window_entry_step = self.step
            entry = self.step

        if not self._frozen_g:
            if self._phase == "1_over_t":
                origin = self._one_over_t_origin_step
                if origin is None:
                    # Window-entry clock. By construction,
                    # ``_phase == '1_over_t'`` only after
                    # ``_window_entry_step`` has been set (the coordinator
                    # flips the phase post-entry). Narrow for the type
                    # checker.
                    origin = cast(int, entry)
                t = self.step - origin + 1
                self._fill_factor = 1.0 / t

            self._entropy[bin_cur] = (
                self._entropy.get(bin_cur, 0) + self._fill_factor
            )
            self._histogram[bin_cur] = self._histogram.get(bin_cur, 0) + 1

        if self._reached_energy_window:
            self._visited_bins.add(bin_cur)
            self._record_recency_visit(bin_cur, int(self.step))

        if (
            not self._frozen_g
            and self.step > 0
            and self.step % self._flatness_check_interval == 0
            and self._reached_energy_window
        ):
            ref = np.min(list(self._entropy.values()))
            for k in self._entropy:
                self._entropy[k] -= ref

        if (
            not self._frozen_g
            and self._phase == "1_over_t"
            and self._dos_snapshot_ratio is not None
        ):
            rung = self._snapshot_rung(self._fill_factor)
            if self._max_snapshot_rung is None or rung > self._max_snapshot_rung:
                step = int(self.step)
                self._fill_factor_snapshots[step] = float(self._fill_factor)
                # Sort once here (as `force_halve` does for the halving
                # history) so `refresh_last_state` -- called on every
                # `results()` / GET_DC -- can copy without re-sorting.
                self._entropy_snapshots[step] = OrderedDict(
                    sorted(self._entropy.items())
                )
                self._max_snapshot_rung = rung

    def _snapshot_rung(self, fill_factor: float) -> int:
        """Fill-factor rung index on the configured log ladder.

        ``floor(log(1/f) / log(ratio))``. A pure function of ``f``, so
        the ladder is drift-free; at ``ratio = 2`` the rungs fall at
        ``f = 2^-k``, coinciding with the halving ladder. Only called
        when ``_dos_snapshot_ratio`` is not ``None``.
        """
        assert self._dos_snapshot_ratio is not None
        return math.floor(
            math.log(1.0 / fill_factor) / math.log(self._dos_snapshot_ratio)
        )

    def _rebuild_max_snapshot_rung(self) -> None:
        """Recompute ``_max_snapshot_rung`` from the snapshot store.

        The rung tracker is in-memory only; on resume it is derived from
        the fill factors already in ``_fill_factor_snapshots`` using the
        configured ratio. An empty store, or disabled snapshotting,
        resets it to ``None`` (so the next 1/t step re-baselines).
        """
        if self._dos_snapshot_ratio is None or not self._fill_factor_snapshots:
            self._max_snapshot_rung = None
            return
        self._max_snapshot_rung = max(
            self._snapshot_rung(f)
            for f in self._fill_factor_snapshots.values()
        )

    def _recency_alpha(self) -> float:
        """EWMA rate ``1 / tau`` with ``tau = recency_visits_per_bin * N``.

        ``N`` is the current known-bin count (``len(self._histogram)``,
        at least 1), so early in a run the effective averaging window is
        shorter than its final value because ``N`` is still growing.
        """
        n_bins = max(1, len(self._histogram))
        return 1.0 / (self._recency_visits_per_bin * n_bins)

    def _record_recency_visit(self, bin_cur: int, step: int) -> None:
        """Decay this bin's weight to ``step``, then add one visit."""
        alpha = self._recency_alpha()
        last = self._recent_last_step.get(bin_cur, step)
        decayed = self._recent_weight.get(bin_cur, 0.0) * (
            (1.0 - alpha) ** (step - last)
        )
        self._recent_weight[bin_cur] = decayed + 1.0
        self._recent_last_step[bin_cur] = step

    def recency_effective_weights(
        self, step: int | None = None
    ) -> dict[int, float]:
        """Per-known-bin EWMA weights decayed to ``step`` (default now).

        Keys are the current known bins (``self._histogram`` keys); a
        bin with no recorded visit reads 0.0.
        """
        if step is None:
            step = int(self.step)
        alpha = self._recency_alpha()
        weights: dict[int, float] = {}
        for b in self._histogram:
            w = self._recent_weight.get(b, 0.0)
            if w:
                w *= (1.0 - alpha) ** (step - self._recent_last_step[b])
            weights[b] = w
        return weights
