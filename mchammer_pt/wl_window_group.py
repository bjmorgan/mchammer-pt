"""Multi-walker Wang-Landau window group."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from mchammer.observers.base_observer import BaseObserver

from .wl_coordinator import (
    _MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED,
    CoordinatorPlan,
    FlatnessMode,
    MergeCadence,
    WalkerPostBlockState,
    _compute_per_walker_flat_min,
    _validate_flatness_mode,
    _validate_merge_cadence,
    merge_entropies,
)
from .wl_replica import WangLandauReplica

if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )


class WangLandauWindowGroup:
    """A group of independent Wang-Landau walkers sharing one energy window.

    Owns the collective halving decision: the halve gate fires when
    either every walker is independently flat (``flatness_mode="per_walker"``,
    published Vogel et al. 2013) or the summed histogram across walkers
    is flat (``flatness_mode="pooled"``, default). At a collective halve
    all walkers' fill factors halve in lockstep and histograms reset.

    Entropy merge cadence is controlled by ``merge_cadence``:

    - ``"at_halve"`` (default): merge entropies at each collective halve
      (Vogel et al. 2013 cadence).
    - ``"never"``: no mid-run merge; walkers run fully independently.

    In the 1/t phase no mid-run merge fires regardless of cadence.

    Args:
        replicas: pre-constructed WangLandauReplica instances, all with
            the same energy window and energy spacing.
        random_seed: seed for the exchange-walker selection RNG.
        flatness_mode: ``"per_walker"`` or ``"pooled"`` (default).
        merge_cadence: ``"at_halve"`` (default) or ``"never"``.
    """

    def __init__(
        self,
        replicas: list[WangLandauReplica],
        *,
        random_seed: int,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
    ) -> None:
        if len(replicas) < 1:
            raise ValueError(
                "WangLandauWindowGroup requires at least one replica"
            )
        if len(replicas) > 1:
            w0 = replicas[0].energy_window
            s0 = replicas[0].energy_spacing
            for r in replicas[1:]:
                if r.energy_window != w0 or r.energy_spacing != s0:
                    raise ValueError(
                        "all replicas in a WangLandauWindowGroup must share "
                        "the same energy window and spacing"
                    )
        _validate_flatness_mode(flatness_mode)
        _validate_merge_cadence(merge_cadence)
        self._replicas = list(replicas)
        self._rng = np.random.default_rng(int(random_seed))
        self._exchange_idx: int = 0
        self._flatness_mode: FlatnessMode = flatness_mode
        self._merge_cadence: MergeCadence = merge_cadence
        # All walkers in a group share _schedule (set via ensemble_kwargs).
        self._schedule: str = self._replicas[0].ensemble._schedule
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
        e = replica.ensemble
        return WalkerPostBlockState(
            is_flat=replica.is_flat(),
            fill_factor=float(e._fill_factor),
            entropy=dict(e._entropy),
            step=int(e.step),
            window_entry_step=(
                None if e._window_entry_step is None
                else int(e._window_entry_step)
            ),
            histogram=dict(e._histogram),
            reached_energy_window=bool(e._reached_energy_window),
        )

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
        """Merge per-walker entropies into a single window estimate.

        Called once at the end of a WL run, regardless of
        ``merge_cadence``. Writes the merged dict into every walker's
        ``_entropy`` and refreshes ``_last_state``, so any downstream
        reader (``WindowResult``, data containers) sees a consistent
        estimate regardless of which walker it samples from.

        No-op for single-walker groups.
        """
        if len(self._replicas) <= 1:
            return
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
    def phase(self) -> str:
        return str(self._replicas[0].ensemble._phase)

    @property
    def schedule(self) -> str:
        return str(self._schedule)

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
            ``fill_factor`` and ``halvings`` from replica 0 (all in sync
            after advance); ``histogram`` is the sum across all walkers
            (the gate-relevant quantity under ``flatness_mode="pooled"``);
            ``converged`` requires every walker to be converged.
            ``flatness_mode`` is this group's mode (used by the progress
            reporter to display the gate-relevant flat_min);
            ``per_walker_flat_min`` is min over walkers of
            ``min(H_k) / mean(H_k)``, or ``None`` if any walker has not
            yet built a histogram.
        """
        e0 = self._replicas[0].ensemble
        combined_hist: dict[int, int] = {}
        for r in self._replicas:
            for k, v in r.ensemble._histogram.items():
                combined_hist[k] = combined_hist.get(k, 0) + v
        per_walker_flat_min = _compute_per_walker_flat_min(
            [r.ensemble._histogram for r in self._replicas]
        )
        return {
            "fill_factor": float(e0._fill_factor),
            "halvings": max(0, len(e0._fill_factor_history) - 1),
            "histogram": combined_hist,
            "converged": self.converged,
            "flatness_mode": self._flatness_mode,
            "per_walker_flat_min": per_walker_flat_min,
        }

    def refresh_last_state(self) -> None:
        """Refresh ``_last_state`` on every walker's container."""
        for r in self._replicas:
            r.refresh_last_state()

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        if len(self._replicas) > 1:
            raise NotImplementedError(_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED)
        return self._replicas[0].snapshot_for_checkpoint()

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
