"""Multi-walker Wang-Landau window group."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from mchammer.observers.base_observer import BaseObserver

from .wl_coordinator import (
    _MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED,
    CoordinatorPlan,
    Phase,
    Schedule,
    WalkerPostBlockState,
    _compute_per_walker_flat_min,
    merge_entropies,
)
from .wl_replica import WangLandauReplica

if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )


class WangLandauWindowGroup:
    """A group of independent Wang-Landau walkers sharing one energy window.

    Runs W walkers in a single energy window, snapshots their state
    after each block, and applies coordinator plans (halve, merge
    entropy, phase switch) to all walkers in lockstep. Policy decisions
    (flatness mode, merge cadence) live at the pool level; the pool
    consults the group's per-walker snapshots, decides what to do, and
    hands the group a ``CoordinatorPlan`` containing the resulting
    actions (not the policy inputs). The group executes the plan
    mechanically.

    All replicas must share the same energy window, energy spacing,
    schedule, and flatness limit. The constructor enforces these
    invariants because coordinator decisions are taken against
    walker 0's cached values and applied uniformly across the group.

    Args:
        replicas: pre-constructed WangLandauReplica instances, all with
            the same energy window, energy spacing, schedule, and
            flatness limit.
        random_seed: seed for the exchange-walker selection RNG.
    """

    def __init__(
        self,
        replicas: list[WangLandauReplica],
        *,
        random_seed: int,
    ) -> None:
        if len(replicas) < 2:
            raise ValueError(
                "WangLandauWindowGroup requires at least two replicas; "
                "use a bare WangLandauReplica for single-walker windows"
            )
        w0 = replicas[0].energy_window
        s0 = replicas[0].energy_spacing
        sched0 = replicas[0].ensemble._schedule
        fl0 = replicas[0].ensemble._flatness_limit
        for r in replicas[1:]:
            if r.energy_window != w0 or r.energy_spacing != s0:
                raise ValueError(
                    "all replicas in a WangLandauWindowGroup must share "
                    "the same energy window and spacing"
                )
            if r.ensemble._schedule != sched0:
                raise ValueError(
                    "all replicas in a WangLandauWindowGroup must share "
                    f"the same schedule; got {sched0!r} on replica 0 and "
                    f"{r.ensemble._schedule!r} on a subsequent replica"
                )
            if r.ensemble._flatness_limit != fl0:
                raise ValueError(
                    "all replicas in a WangLandauWindowGroup must share "
                    f"the same flatness_limit; got {fl0!r} on replica 0 "
                    f"and {r.ensemble._flatness_limit!r} on a subsequent "
                    "replica"
                )
        self._replicas = list(replicas)
        self._rng = np.random.default_rng(int(random_seed))
        self._exchange_idx: int = 0
        self._schedule: Schedule = cast(Schedule, sched0)
        self.walker_states: list[WalkerPostBlockState] = [
            WalkerPostBlockState(
                is_flat=False,
                fill_factor=1.0,
                entropy={},
                step=0,
                window_entry_step=None,
                histogram={},
                reached_energy_window=False,
            )
            for _ in self._replicas
        ]

    def advance(self, n_steps: int) -> None:
        """Advance all W replicas; refresh walker_states.

        Coordinator decide/apply runs in
        ``SerialWangLandauPool.advance_all`` after this returns.
        """
        for r in self._replicas:
            r.advance(int(n_steps))
        self.walker_states = [
            self._snapshot_replica(r) for r in self._replicas
        ]

    def _snapshot_replica(self, replica: WangLandauReplica) -> WalkerPostBlockState:
        """Read live ensemble state into a WalkerPostBlockState."""
        return replica._snapshot()

    def apply_plan(self, plan: CoordinatorPlan) -> None:
        """Apply the coordinator's plan to every walker.

        Order: halve (zeroes histograms, halves fill factors) -> write
        merged entropy -> set phase.
        """
        if plan.halve:
            for r in self._replicas:
                r.force_halve()
        if plan.merged_entropy is not None:
            merged = dict(plan.merged_entropy)
            for r in self._replicas:
                r.ensemble._entropy = dict(merged)
        if plan.switch_to_phase is not None:
            phase = plan.switch_to_phase
            for r in self._replicas:
                r.ensemble._phase = phase
                if phase == "1_over_t":
                    entry = r.ensemble._window_entry_step
                    if entry is not None:
                        t = r.ensemble.step - entry + 1
                        r.ensemble._fill_factor = 1.0 / t

    def reroll_exchange_idx(self) -> None:
        """Re-select the exchange-representative walker."""
        self._exchange_idx = int(self._rng.integers(0, len(self._replicas)))

    def finalise_for_reporting(self) -> None:
        """Merge per-walker entropies; write the result into every walker.

        Called once at the end of a WL run. Writes the merged dict into
        every walker's ``_entropy`` and refreshes ``_last_state``, so any
        downstream reader (``WindowResult``, data containers) sees a
        consistent estimate regardless of which walker it samples from.
        """
        merged = merge_entropies(
            [dict(r.ensemble._entropy) for r in self._replicas]
        )
        for r in self._replicas:
            r.ensemble._entropy = dict(merged)
        self.refresh_last_state()

    @property
    def ensemble(self) -> Any:
        """The mchammer ensemble for the first walker.

        Used by pool-level metadata queries (ensemble class, observer
        snapshot). All walkers share the same ensemble class and kwargs.
        For read-only metadata queries only; use `attach_mchammer_observer`,
        `attach_observer_class`, or `attach_observer_factory` to add
        observers to all walkers.
        """
        return self._replicas[0].ensemble

    @property
    def energy_window(self) -> tuple[float | None, float | None]:
        return self._replicas[0].energy_window

    @property
    def energy_spacing(self) -> float:
        return self._replicas[0].energy_spacing

    @property
    def cluster_expansion_path(self) -> str | None:
        return self._replicas[0].cluster_expansion_path

    @property
    def phase(self) -> Phase:
        return cast(Phase, self._replicas[0].ensemble._phase)

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def flatness_limit(self) -> float:
        return float(self._replicas[0].ensemble._flatness_limit)

    def current_energy(self) -> float:
        return self._replicas[self._exchange_idx].current_energy()

    def current_occupations(self) -> np.ndarray:
        return self._replicas[self._exchange_idx].current_occupations()

    def set_occupations(self, occupations: np.ndarray) -> None:
        self._replicas[self._exchange_idx].set_occupations(occupations)

    def log_g(self, energy: float) -> float:
        return self._replicas[0].log_g(energy)

    @property
    def converged(self) -> bool:
        return all(r.converged for r in self._replicas)

    def is_flat(self) -> bool:
        """Return ``True`` iff every walker in the group is flat."""
        return all(r.is_flat() for r in self._replicas)

    def data_container(self) -> WangLandauDataContainer:
        """Data container for the first walker in the group."""
        return self._replicas[0].data_container()

    def all_data_containers(self) -> list[WangLandauDataContainer]:
        """All per-walker data containers."""
        return [r.data_container() for r in self._replicas]

    def window_stats(self) -> dict[str, Any]:
        """Per-window convergence metrics.

        Returns:
            ``fill_factor`` and ``halvings`` from replica 0 (all in
            sync after advance); ``histogram`` is the sum across all
            walkers; ``bins_visited`` is the size of the union of
            walker ``_visited_bins`` (bins any walker has reached
            via MC since window entry); ``bins_known`` is the size
            of the union of ``_histogram`` keys across walkers (i.e.
            ``len(combined_hist)``). ``converged`` requires every
            walker to be converged; ``per_walker_flat_min`` is min
            over walkers of ``min(H_k) / mean(H_k)``, or ``None`` if
            any walker has not yet built a histogram.
            ``flatness_mode`` is not included here; the pool injects
            it (pool-level policy).
        """
        e0 = self._replicas[0].ensemble
        combined_hist: dict[int, int] = {}
        visited_union: set[int] = set()
        for r in self._replicas:
            for k, v in r.ensemble._histogram.items():
                combined_hist[k] = combined_hist.get(k, 0) + v
            visited_union |= r.ensemble._visited_bins
        per_walker_flat_min = _compute_per_walker_flat_min(
            [r.ensemble._histogram for r in self._replicas]
        )
        return {
            "fill_factor": float(e0._fill_factor),
            "halvings": max(0, len(e0._fill_factor_history) - 1),
            "histogram": combined_hist,
            "bins_visited": len(visited_union),
            "bins_known": len(combined_hist),
            "converged": self.converged,
            "per_walker_flat_min": per_walker_flat_min,
        }

    def refresh_last_state(self) -> None:
        """Refresh ``_last_state`` on every walker's container."""
        for r in self._replicas:
            r.refresh_last_state()

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError(_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED)

    def attach_mchammer_observer(self, observer: BaseObserver) -> None:
        """Attach observer to all W replicas; each receives its own copy."""
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        for r in self._replicas:
            r.attach_mchammer_observer(pickle.loads(blob))

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to every walker in the group.

        Each of the W replicas receives its own ``cls(*args, **kwargs)``
        instance, ensuring independent per-walker state.
        """
        for r in self._replicas:
            r.attach_mchammer_observer(cls(*args, **kwargs))

    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
    ) -> None:
        """Attach an observer constructed per walker via ``factory``.

        ``factory(replica)`` is called once per walker and must return a
        fresh ``BaseObserver``.
        """
        for r in self._replicas:
            observer = factory(r)
            if not isinstance(observer, BaseObserver):
                raise TypeError(
                    f"attach_observer_factory: factory returned "
                    f"{type(observer).__name__}, not a BaseObserver"
                )
            r.attach_mchammer_observer(observer)
