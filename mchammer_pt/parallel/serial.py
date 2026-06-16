"""In-process replica pool: advances replicas sequentially in the caller."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.observers.base_observer import (
    BaseObserver,
)

from ..exchange import matching_for_boundary
from ..replica import Replica
from ..wl_coordinator import (
    FlatnessMode,
    MergeCadence,
    OneOverTEntry,
    OneOverTGate,
    SlotView,
    _resolve_bins_filled,
    _resolve_recency_flatness,
    _validate_bp_stall_multiple,
    _validate_flatness_mode,
    _validate_gate_schedule,
    _validate_merge_cadence,
    _validate_one_over_t_gate,
    decide_block_actions,
)
from ..wl_merge_diagnostics import MergeEvent
from ..wl_replica import WangLandauReplica, WangLandauSlot
from ._imports import _resolve_replicas

if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )


class SerialPool:
    """Advances replicas sequentially in the calling process.

    The pool owns a list of `Replica` instances and exposes the full
    `ObservablePool` surface. Use for debugging, for small runs, or
    when the per-cycle walltime is dominated by something other than
    MC time.
    """

    def __init__(self, replicas: Sequence[Replica]) -> None:
        self._replicas: list[Replica] = list(replicas)

    def __len__(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> list[Replica]:
        """The pool's `Replica` instances. Returns a copy."""
        return list(self._replicas)

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully qualified name of the ensemble class used by replicas."""
        cls = type(self._replicas[0].ensemble)
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def ensemble_kwargs_hash(self) -> str:
        """Best-effort hash of ensemble kwargs. Sentinel for serial pools."""
        return ""

    @property
    def temperatures(self) -> list[float]:
        return [r.temperature for r in self._replicas]

    def advance_all(self, n_steps: int) -> None:
        for replica in self._replicas:
            replica.advance(n_steps)

    def current_energies(self) -> np.ndarray:
        return np.array([r.current_energy() for r in self._replicas], dtype=np.float64)

    def current_energy(self, i: int) -> float:
        return self._replicas[i].current_energy()

    def current_occupations(self, i: int) -> np.ndarray:
        return self._replicas[i].current_occupations()

    def swap_configurations(self, i: int, j: int) -> None:
        occ_i = self._replicas[i].current_occupations()
        occ_j = self._replicas[j].current_occupations()
        self._replicas[i].set_occupations(occ_j)
        try:
            self._replicas[j].set_occupations(occ_i)
        except BaseException:
            self._replicas[i].set_occupations(occ_i)
            raise

    def n_walkers(self, i: int) -> int:
        """Single-walker rungs: always 1."""
        return 1

    def walker_energy(self, i: int, walker: int) -> float:
        """Energy of rung ``i`` (single-walker; ``walker`` is always 0)."""
        return self.current_energy(i)

    def candidate_pairs(
        self, i: int, j: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Single-walker rungs exchange the one pair ``(0, 0)``.

        Returns the fixed pair without drawing from ``rng``, so the
        exchange RNG stream is left untouched.
        """
        return [(0, 0)]

    def window_of_position(self) -> np.ndarray:
        """One position per rung: the identity mapping."""
        return np.arange(len(self._replicas), dtype=np.int64)

    def n_carriers(self) -> int:
        """One walker per rung."""
        return len(self._replicas)

    def swap_walker_configurations(self, i: int, a: int, j: int, b: int) -> None:
        """Delegate to ``swap_configurations`` (rungs are single-walker)."""
        self.swap_configurations(i, j)

    def apply_swaps(self, swaps: list[tuple[int, int, int, int]]) -> None:
        """Apply accepted walker-config swaps, one at a time."""
        for i, a, j, b in swaps:
            self.swap_walker_configurations(i, a, j, b)

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to selected replicas.

        Each replica receives its own deserialised copy of ``observer``
        via a pickle round-trip; the ``observer`` argument itself is
        never registered on any replica. If ``observer`` is not
        picklable, raises ``TypeError`` immediately and points at
        ``attach_observer_class`` as the escape hatch.
        """
        target_indices = _resolve_replicas(replicas, len(self._replicas))
        if not target_indices:
            return
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        for i in target_indices:
            self._replicas[i].attach_mchammer_observer(pickle.loads(blob))

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        replicas: Sequence[int] | Literal["all"] = "all",
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to selected replicas.

        Each selected replica receives its own ``cls(*args, **kwargs)``
        instance. A parent-side dry-run construction validates the
        arguments and the ``BaseObserver`` return type before any
        replica is touched.

        The constructor must be free of externally-visible side effects:
        it fires once in the parent (the dry-run) plus once per selected
        replica.
        """
        target_indices = _resolve_replicas(replicas, len(self._replicas))
        if not target_indices:
            return
        probe = cls(*args, **kwargs)
        if not isinstance(probe, BaseObserver):
            raise TypeError(
                f"attach_observer_class: {cls.__name__}(...) returned "
                f"{type(probe).__name__}, not a BaseObserver"
            )
        del probe
        for i in target_indices:
            self._replicas[i].attach_mchammer_observer(cls(*args, **kwargs))

    def attach_observer_factory(
        self,
        factory: Callable[[Replica], BaseObserver],
        *,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer constructed locally per replica.

        ``factory(replica)`` is called once per selected replica with that
        replica as its sole argument and must return a fresh
        ``BaseObserver``. Use this for observers whose constructors take
        icet objects (``ClusterSpace``, ``ClusterExpansion``) that do not
        pickle. The factory should reload the CE from disk inside the
        factory::

            def make_obs(replica):
                ce = ClusterExpansion.read(replica.cluster_expansion_path)
                return ClusterCountObserver(
                    ce.get_cluster_space_copy(), ..., interval=...
                )

        On ``SerialPool``, ``replica.cluster_expansion_path`` is
        ``None`` unless you passed ``cluster_expansion_path=`` to
        ``Replica.__init__``. ``ProcessPool`` auto-populates the path
        on every worker.

        A factory written for ``SerialPool`` runs unchanged on
        ``ProcessPool``, where it must additionally be a top-level
        function or class method importable by fully qualified name.
        """
        target_indices = _resolve_replicas(replicas, len(self._replicas))
        if not target_indices:
            return
        for i in target_indices:
            observer = factory(self._replicas[i])
            if not isinstance(observer, BaseObserver):
                raise TypeError(
                    f"attach_observer_factory: factory returned "
                    f"{type(observer).__name__}, not a BaseObserver"
                )
            self._replicas[i].attach_mchammer_observer(observer)

    def get_observers(self, replica_index: int) -> dict[str, BaseObserver]:
        """Return a snapshot of the observers attached to one replica.

        The returned dict is keyed by observer tag. Values are
        independent copies via ``pickle`` round-trip — mutations on
        the returned observers do not affect the pool's running
        state.

        Raises:
            IndexError: if ``replica_index`` is out of range.
            TypeError: if the observer dict cannot be round-tripped
                through pickle.
        """
        n = len(self._replicas)
        if not 0 <= replica_index < n:
            raise IndexError(
                f"replica index {replica_index} out of range "
                f"for pool of size {n}"
            )
        live = self._replicas[replica_index].ensemble.observers
        try:
            return pickle.loads(pickle.dumps(live))
        except Exception as exc:
            raise TypeError(
                f"observer dict for replica {replica_index} could not be "
                f"round-tripped through pickle ({exc})"
            ) from exc

    def data_containers(self) -> list[BaseDataContainer]:
        return [r.data_container() for r in self._replicas]

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]]:
        return [r.snapshot_for_checkpoint() for r in self._replicas]

    def shutdown(self) -> None:
        return None


class SerialWangLandauPool:
    """In-process pool of `WangLandauReplica` instances.

    Mirrors `SerialPool` for canonical replicas. Satisfies
    `WangLandauObservablePool`: the same observer-attach surface as
    `SerialPool` (`attach_observer`, `attach_observer_class`,
    `attach_observer_factory`, `get_observers`) is exposed so users
    can record per-step observables during a REWL run.

    Args:
        replicas: the per-window Wang-Landau slot instances.
        energy_spacing: bin width shared across all replicas.
        flatness_mode: ``"pooled"`` or ``"per_walker"`` halving gate.
        merge_cadence: ``"at_halve"`` or ``"never"`` entropy-merge cadence.
        one_over_t_gate: gate controlling entry into the 1/t phase.
        bp_stall_multiple: stall-escape multiple for the BP switch.
        frozen_measurement: if ``True``, ``advance_all`` advances all
            walkers as normal but skips the coordinator (no halving,
            no entropy-merge, no phase-switch). Every walker's g(E)
            is left untouched. Intended for post-convergence measurement
            passes. Default ``False``.
    """

    def __init__(
        self,
        replicas: Sequence[WangLandauSlot],
        *,
        energy_spacing: float,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
        one_over_t_gate: OneOverTGate = "visit_once",
        bp_stall_multiple: float = 4.0,
        frozen_measurement: bool = False,
    ) -> None:
        self._replicas: list[WangLandauSlot] = list(replicas)
        self._energy_spacing = float(energy_spacing)
        self._frozen_measurement: bool = frozen_measurement
        _validate_flatness_mode(flatness_mode)
        _validate_merge_cadence(merge_cadence)
        _validate_one_over_t_gate(one_over_t_gate)
        self._flatness_mode: FlatnessMode = flatness_mode
        self._merge_cadence: MergeCadence = merge_cadence
        self._one_over_t_gate: OneOverTGate = one_over_t_gate
        self._bp_stall_multiple: float = _validate_bp_stall_multiple(
            bp_stall_multiple
        )
        self._last_halve_step: list[int | None] = [None] * len(self._replicas)
        self._first_halve_duration: list[int | None] = (
            [None] * len(self._replicas)
        )
        self._merge_events: list[MergeEvent] = []
        for r in self._replicas:
            if r.energy_spacing != self._energy_spacing:
                raise ValueError(
                    f"replica energy_spacing {r.energy_spacing} does not "
                    f"match pool energy_spacing {self._energy_spacing}"
                )
            _validate_gate_schedule(one_over_t_gate, r.schedule)
            if r.one_over_t_entry != self._replicas[0].one_over_t_entry:
                raise ValueError(
                    "all slots in a SerialWangLandauPool must share the "
                    f"same one_over_t_entry; got "
                    f"{self._replicas[0].one_over_t_entry!r} on slot 0 "
                    f"and {r.one_over_t_entry!r} on a subsequent slot. "
                    "Checkpoint metadata records a single entry policy "
                    "for the run."
                )

    def __len__(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> list[WangLandauSlot]:
        return list(self._replicas)

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully qualified name of the ensemble class used by replicas."""
        cls = type(self._replicas[0].ensemble)
        return f"{cls.__module__}.{cls.__qualname__}"

    @property
    def ensemble_kwargs_hash(self) -> str:
        """Best-effort hash of ensemble kwargs. Sentinel for serial pools."""
        return ""

    @property
    def windows(self) -> list[tuple[float | None, float | None]]:
        return [r.energy_window for r in self._replicas]

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    @property
    def flatness_mode(self) -> FlatnessMode:
        """Collective-halving flatness mode this pool drives."""
        return self._flatness_mode

    @property
    def merge_cadence(self) -> MergeCadence:
        """Entropy-merge cadence this pool drives."""
        return self._merge_cadence

    @property
    def one_over_t_gate(self) -> OneOverTGate:
        """1/t-schedule halving-phase gate this pool drives."""
        return self._one_over_t_gate

    @property
    def bp_stall_multiple(self) -> float:
        """Stall-escape multiple this pool drives (consulted under flatness)."""
        return self._bp_stall_multiple

    @property
    def one_over_t_entry(self) -> OneOverTEntry:
        """1/t entry policy shared by this pool's walkers.

        Walker-side config held by the replicas; the constructor
        enforces that all slots agree, so reading slot 0 is
        representative.
        """
        return self._replicas[0].one_over_t_entry

    def _view_of(self, index: int, slot: WangLandauSlot) -> SlotView:
        """Build a SlotView from a slot's walker_states plus pool config."""
        return SlotView(
            walker_states=tuple(slot.walker_states),
            phase=slot.phase,
            flatness_mode=self._flatness_mode,
            merge_cadence=self._merge_cadence,
            schedule=slot.schedule,
            flatness_limit=slot.flatness_limit,
            one_over_t_gate=self._one_over_t_gate,
            bp_stall_multiple=self._bp_stall_multiple,
            last_halve_step=self._last_halve_step[index],
            first_halve_duration=self._first_halve_duration[index],
        )

    def _update_stall_state(
        self,
        index: int,
        step: int,
        window_entry_steps: list[int | None],
    ) -> None:
        """Record the halve step (and first-stage duration) after a halve.

        The caller guards on ``plan.halve``; this records unconditionally,
        matching the inline update in ``ProcessWangLandauPool.advance_all``.
        """
        if self._last_halve_step[index] is None:
            entries = [e for e in window_entry_steps if e is not None]
            if entries:
                self._first_halve_duration[index] = step - max(entries)
        self._last_halve_step[index] = step

    def seed_stall_state(
        self, per_slot: list[tuple[int | None, int | None]]
    ) -> None:
        """Seed per-slot ``(last_halve_step, first_halve_duration)`` on resume."""
        if len(per_slot) != len(self._replicas):
            raise ValueError(
                f"seed_stall_state expects {len(self._replicas)} entries, "
                f"got {len(per_slot)}"
            )
        for i, (last, t1) in enumerate(per_slot):
            self._last_halve_step[i] = last
            self._first_halve_duration[i] = t1

    def advance_all(self, n_steps: int) -> None:
        # ADVANCE + COLLECT: each slot advances its walkers and populates
        # its walker_states from live ensemble state.
        for slot in self._replicas:
            slot.advance(n_steps)

        # Frozen mode: walkers advance but the coordinator does not run.
        # No halving, entropy-merge, or phase-switch; g(E) is untouched.
        if self._frozen_measurement:
            return

        # DECIDE: per-slot coordinator decisions; pure-Python, no IPC.
        views = [
            self._view_of(i, s) for i, s in enumerate(self._replicas)
        ]
        plans = [decide_block_actions(v) for v in views]

        # RECORD: capture merged entropy per halving merge while the
        # master-side plan still carries it. See wl_merge_diagnostics.
        for slot_index, (view, plan) in enumerate(
            zip(views, plans, strict=True)
        ):
            if plan.merged_entropy is not None:
                self._merge_events.append(
                    MergeEvent(
                        slot_index=slot_index,
                        step=view.walker_states[0].step,
                        merged_entropy=dict(plan.merged_entropy),
                    )
                )

        # APPLY: per-slot mutation. No batching benefit in-process.
        for slot, plan in zip(self._replicas, plans, strict=True):
            slot.apply_plan(plan)

        # TRACK: the pool issues every halve, so it owns the stall state
        # the decoupled BP switch reads back through the SlotView.
        for i, (slot, plan) in enumerate(
            zip(self._replicas, plans, strict=True)
        ):
            if plan.halve:
                self._update_stall_state(
                    i,
                    step=slot.walker_states[0].step,
                    window_entry_steps=[
                        s.window_entry_step for s in slot.walker_states
                    ],
                )

    def current_energies(self) -> np.ndarray:
        return np.array(
            [r.walker_energy(0) for r in self._replicas], dtype=np.float64
        )

    def current_energy(self, i: int) -> float:
        return self._replicas[i].walker_energy(0)

    def current_occupations(self, i: int) -> np.ndarray:
        return self._replicas[i].walker_occupations(0)

    def swap_configurations(self, i: int, j: int) -> None:
        """Swap walker 0 of window ``i`` with walker 0 of window ``j``."""
        self.swap_walker_configurations(i, 0, j, 0)

    def n_walkers(self, i: int) -> int:
        """Number of walkers in window ``i``."""
        return self._replicas[i].n_walkers

    def walker_energy(self, i: int, walker: int) -> float:
        """Current energy of ``walker`` in window ``i``."""
        return self._replicas[i].walker_energy(walker)

    def walker_log_g(self, i: int, walker: int, energy: float) -> float:
        """Density-of-states ``log g(E)`` for ``walker`` in window ``i``."""
        return self._replicas[i].walker_log_g(walker, energy)

    def candidate_pairs(
        self, i: int, j: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Random matching of windows ``i`` and ``j`` walkers for exchange.

        Each returned ``(a, b)`` pairs walker ``a`` of window ``i`` with
        walker ``b`` of window ``j``; pairs are disjoint in each
        coordinate. See :func:`mchammer_pt.exchange.matching_for_boundary`.
        """
        return matching_for_boundary(self.n_walkers(i), self.n_walkers(j), rng)

    def window_of_position(self) -> np.ndarray:
        """Window index of each ``(window, walker)`` position, in order."""
        counts = [self.n_walkers(i) for i in range(len(self._replicas))]
        return np.repeat(np.arange(len(counts), dtype=np.int64), counts)

    def n_carriers(self) -> int:
        """Total number of walker positions across all windows."""
        return sum(self.n_walkers(i) for i in range(len(self._replicas)))

    def swap_walker_configurations(self, i: int, a: int, j: int, b: int) -> None:
        """Swap the configurations of walker ``a`` in window ``i`` and
        walker ``b`` in window ``j``.

        Rolls back the first assignment if the second fails, so a raising
        ``set_walker_occupations`` leaves both walkers unchanged.
        """
        occ_i = self._replicas[i].walker_occupations(a)
        occ_j = self._replicas[j].walker_occupations(b)
        self._replicas[i].set_walker_occupations(a, occ_j)
        try:
            self._replicas[j].set_walker_occupations(b, occ_i)
        except BaseException:
            self._replicas[i].set_walker_occupations(a, occ_i)
            raise

    def apply_swaps(self, swaps: list[tuple[int, int, int, int]]) -> None:
        """Apply accepted walker-config swaps, one at a time."""
        for i, a, j, b in swaps:
            self.swap_walker_configurations(i, a, j, b)

    def log_g(self, i: int, energy: float) -> float:
        return self._replicas[i].log_g(energy)

    def converged_flags(self) -> np.ndarray:
        return np.array([r.converged for r in self._replicas], dtype=bool)

    def per_window_stats(self) -> list[dict[str, Any]]:
        stats = [r.window_stats() for r in self._replicas]
        for d in stats:
            d["flatness_mode"] = self._flatness_mode
            _resolve_bins_filled(d, self._flatness_mode)
            _resolve_recency_flatness(d, self._flatness_mode)
        return stats

    @property
    def merge_events(self) -> tuple[MergeEvent, ...]:
        """Per-halving merged entropies; see :mod:`wl_merge_diagnostics`."""
        return tuple(self._merge_events)

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to selected WL replicas.

        Mirrors `SerialPool.attach_observer`: each replica receives
        its own deserialised copy of ``observer`` via a pickle
        round-trip; the ``observer`` argument itself is never
        registered on any replica. If ``observer`` is not picklable,
        raises ``TypeError`` immediately and points at
        ``attach_observer_class`` as the escape hatch.
        """
        target_indices = _resolve_replicas(replicas, len(self._replicas))
        if not target_indices:
            return
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        for i in target_indices:
            self._replicas[i].attach_mchammer_observer(pickle.loads(blob))

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        replicas: Sequence[int] | Literal["all"] = "all",
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to selected WL replicas.

        Mirrors `SerialPool.attach_observer_class`: a parent-side
        dry-run construction validates the arguments and the
        ``BaseObserver`` return type before any replica is touched.
        The constructor must be free of externally-visible side
        effects (it fires once in the parent plus once per selected
        replica).
        """
        target_indices = _resolve_replicas(replicas, len(self._replicas))
        if not target_indices:
            return
        probe = cls(*args, **kwargs)
        if not isinstance(probe, BaseObserver):
            raise TypeError(
                f"attach_observer_class: {cls.__name__}(...) returned "
                f"{type(probe).__name__}, not a BaseObserver"
            )
        del probe
        for i in target_indices:
            self._replicas[i].attach_observer_class(cls, *args, **kwargs)

    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
        *,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer constructed locally per WL replica.

        Mirrors `SerialPool.attach_observer_factory`. The factory
        runs once per selected replica with that replica as its sole
        argument and must return a fresh ``BaseObserver``. Use this
        for observers whose constructors take icet objects
        (``ClusterSpace``, ``ClusterExpansion``) that do not pickle;
        reload the CE from disk inside the factory via
        ``ClusterExpansion.read(replica.cluster_expansion_path)``.

        For ``WangLandauWindowGroup`` slots, the factory is called once
        per walker (each inner ``WangLandauReplica``), so every walker
        receives a freshly-constructed independent observer.

        On ``SerialWangLandauPool``,
        ``replica.cluster_expansion_path`` is ``None`` unless you
        passed ``cluster_expansion_path=`` to
        ``WangLandauReplica.__init__``. ``ProcessWangLandauPool``
        auto-populates the path on every worker.
        """
        target_indices = _resolve_replicas(replicas, len(self._replicas))
        if not target_indices:
            return
        for i in target_indices:
            self._replicas[i].attach_observer_factory(factory)

    def get_observers(self, replica_index: int) -> dict[str, BaseObserver]:
        """Return a snapshot of the observers attached to one WL replica.

        Mirrors `SerialPool.get_observers`: the returned dict is
        keyed by observer tag; values are independent copies via
        ``pickle`` round-trip so mutations on the returned objects
        do not affect the pool's running state.

        For ``WangLandauWindowGroup`` slots, only the observers from
        the first walker in the group are returned.

        Raises:
            IndexError: if ``replica_index`` is out of range.
            TypeError: if the observer dict cannot be round-tripped
                through pickle.
        """
        n = len(self._replicas)
        if not 0 <= replica_index < n:
            raise IndexError(
                f"replica index {replica_index} out of range "
                f"for pool of size {n}"
            )
        slot = self._replicas[replica_index]
        live = slot.ensemble.observers
        try:
            return pickle.loads(pickle.dumps(live))
        except Exception as exc:
            raise TypeError(
                f"observer dict for replica {replica_index} could not be "
                f"round-tripped through pickle ({exc})"
            ) from exc

    def data_containers(self) -> list[WangLandauDataContainer]:
        """Flat per-walker containers in window-major / walker-minor order.

        For W=1 slots this is the bare replica's single container;
        for W>1 slots this expands to all walkers in the group.
        """
        from ..wl_window_group import WangLandauWindowGroup

        out: list[WangLandauDataContainer] = []
        for slot in self._replicas:
            if isinstance(slot, WangLandauWindowGroup):
                out.extend(r.data_container() for r in slot._replicas)
            else:
                out.append(slot.data_container())
        return out

    def per_window_data_containers(self) -> list[list[WangLandauDataContainer]]:
        """All data containers grouped by window slot.

        Refreshes ``_last_state`` on every walker's container before
        returning, so callers always see current entropy/histogram.

        Returns a list of length n_windows; each entry is a list of
        WangLandauDataContainer instances — one per walker for
        WangLandauWindowGroup slots, one for WangLandauReplica slots.
        """
        for r in self._replicas:
            r.refresh_last_state()
        return [r.all_data_containers() for r in self._replicas]

    def finalise_for_reporting(self) -> None:
        """Delegate to each slot.

        Multi-walker slots (``WangLandauWindowGroup``) merge their
        walkers' entropies in-place. Single-walker slots
        (``WangLandauReplica``) are no-ops — there is nothing to
        merge.
        """
        for r in self._replicas:
            r.finalise_for_reporting()

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        """Snapshot per-walker and group-level checkpoint state.

        Returns:
            Dict with:
                "per_walker": flat list of per-walker snapshot dicts in
                    window-major / walker-minor order (length M = sum
                    of walkers_per_window across slots).
                "group_state": list of length N (one per window slot).
                    Each entry is a dict with ``rng_state`` and
                    ``phase`` for W>1 slots, or ``None`` for
                    bare-replica W=1 slots.
        """
        from ..wl_window_group import WangLandauWindowGroup

        per_walker: list[dict[str, Any]] = []
        group_state: list[dict[str, Any] | None] = []
        for slot in self._replicas:
            if isinstance(slot, WangLandauWindowGroup):
                snap = slot.snapshot_for_checkpoint()
                per_walker.extend(snap["per_walker"])
                group_state.append(snap["group_state"])
            else:
                per_walker.append(slot.snapshot_for_checkpoint())
                group_state.append(None)
        return {"per_walker": per_walker, "group_state": group_state}

    def restore_replica_state(
        self,
        containers: list[BaseDataContainer],
        per_walker_extras: list[dict[str, Any]],
        group_state: list[dict[str, Any] | None],
    ) -> None:
        """Push saved per-walker and group state into each slot.

        Args:
            containers: flat list of M containers in window-major /
                walker-minor order. Length must equal the pool's total
                walker count (sum over slots of ``len(slot._replicas)``
                for ``WangLandauWindowGroup`` slots, or 1 for bare-replica
                slots).
            per_walker_extras: flat list of M per-walker extras dicts,
                same order as ``containers``.
            group_state: list of length N (one per slot). ``None`` for
                bare-replica W=1 slots; a dict for
                ``WangLandauWindowGroup`` slots.

        Raises:
            ValueError: any of the three inputs has a mismatched length,
                or a W=1 slot receives a non-None group_state entry (or
                vice versa).
        """
        from ..wl_window_group import WangLandauWindowGroup

        expected_m = sum(
            len(slot._replicas) if isinstance(slot, WangLandauWindowGroup) else 1
            for slot in self._replicas
        )
        if len(containers) != expected_m:
            raise ValueError(
                f"restore_replica_state expects {expected_m} containers, "
                f"got {len(containers)}"
            )
        if len(per_walker_extras) != expected_m:
            raise ValueError(
                f"restore_replica_state expects {expected_m} per_walker_extras, "
                f"got {len(per_walker_extras)}"
            )
        if len(group_state) != len(self._replicas):
            raise ValueError(
                f"restore_replica_state expects {len(self._replicas)} "
                f"group_state entries, got {len(group_state)}"
            )

        offset = 0
        for slot, gs in zip(self._replicas, group_state, strict=True):
            if isinstance(slot, WangLandauWindowGroup):
                n = len(slot._replicas)
                if gs is None:
                    raise ValueError(
                        "group_state entry is None for a multi-walker slot"
                    )
                slot.restore_state(
                    containers=containers[offset:offset + n],
                    per_walker_extras=per_walker_extras[offset:offset + n],
                    group_state=gs,
                )
                offset += n
            elif isinstance(slot, WangLandauReplica):
                if gs is not None:
                    raise ValueError(
                        "group_state entry is non-None for a bare-replica slot"
                    )
                slot.restore_state(
                    containers[offset],
                    sites_by_species=per_walker_extras[offset]["sites_by_species"],
                )
                offset += 1
            else:
                raise TypeError(
                    f"unexpected slot type {type(slot).__name__!r} in "
                    f"restore_replica_state; expected WangLandauReplica or "
                    f"WangLandauWindowGroup"
                )

    def shutdown(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        """Serial pool has no shutdown state; always open."""
        return True
