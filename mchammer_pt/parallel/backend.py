"""Protocols for replica pools and observable replica pools.

A `ReplicaPool` owns the state of every replica in a parallel-tempering
run and exposes the operations the orchestrator needs: advance, query
energies, atomically swap configurations, and return native mchammer
data containers at end-of-run.

Pool implementations vary in whether they can carry user-supplied
`mchammer.BaseObserver` instances across their execution boundary.
Those that can satisfy the `_ObserverAttach` mixin — combined with
`CanonicalPool` this gives `ObservablePool`, combined with
`WangLandauPool` it gives `WangLandauObservablePool`. Pools that
cannot carry observers satisfy only the relevant base protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..replica import Replica
    from ..wl_coordinator import (
        FlatnessMode,
        MergeCadence,
        OneOverTEntry,
        OneOverTGate,
    )

import numpy as np
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.observers.base_observer import (
    BaseObserver,
)

from ..wl_merge_diagnostics import MergeEvent


@runtime_checkable
class ReplicaPool(Protocol):
    """Owns replica state; exposes operations the orchestrator needs."""

    def __len__(self) -> int: ...

    def advance_all(self, n_steps: int) -> None:
        """Advance every replica by ``n_steps`` MC trial steps."""
        ...

    def current_energies(self) -> np.ndarray:
        """Snapshot total CE energy for every replica.

        Returned array has shape ``(len(self),)`` and dtype float64.
        """
        ...

    def current_energy(self, i: int) -> float:
        """Single-replica energy query, used inside the exchange loop.

        Kept separate from `current_energies()` so per-pair exchange
        evaluation avoids the full-vector fetch overhead.
        """
        ...

    def current_occupations(self, i: int) -> np.ndarray:
        """Copy of replica ``i``'s current occupation vector (atomic numbers).

        Useful for end-of-run analysis where the caller wants the final
        configuration of a specific replica. Not used by the orchestrator
        itself. Integer dtype; shape ``(n_sites,)``.
        """
        ...

    def swap_configurations(self, i: int, j: int) -> None:
        """Atomically exchange the configurations of replicas i and j.

        After a successful return, ``current_energy(i)`` yields what
        ``current_energy(j)`` returned before the call, and vice versa.

        On raise, the pool rolls back replica *i* to its pre-swap
        configuration so both replicas remain consistent. The
        enclosing run propagates the exception unchanged.
        """
        ...

    def n_walkers(self, i: int) -> int:
        """Number of walkers carried by position ``i`` (1 for canonical)."""
        ...

    def walker_energy(self, i: int, walker: int) -> float:
        """Current energy of ``walker`` at position ``i``."""
        ...

    def candidate_pairs(
        self, i: int, j: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Random walker matching across the boundary of positions ``i`` and ``j``."""
        ...

    def window_of_position(self) -> np.ndarray:
        """Window index of each ``(window, walker)`` carrier, in order."""
        ...

    def n_carriers(self) -> int:
        """Total number of walker carriers across all positions."""
        ...

    def swap_walker_configurations(self, i: int, a: int, j: int, b: int) -> None:
        """Swap walker ``a`` of window ``i`` with walker ``b`` of window ``j``.

        Not guaranteed atomic across a multiprocessing backend; on a worker
        failure the pool is shut down rather than left half-swapped.
        """
        ...

    def apply_swaps(self, swaps: list[tuple[int, int, int, int]]) -> None:
        """Apply a batch of accepted ``(i, a, j, b)`` walker-config swaps."""
        ...

    def data_containers(self) -> list[BaseDataContainer]:
        """One native mchammer ``BaseDataContainer`` per replica.

        Populated by mchammer's native trajectory logging regardless of
        whether user observers have been attached. A pool that is not
        an ``ObservablePool`` still returns containers with mchammer's
        default trajectory data. May involve inter-process
        communication for remote-state pools; call once at end-of-run.
        """
        ...

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]] | dict[str, Any]:
        """Per-replica state snapshot for the checkpoint payload.

        Refreshes each replica's ``BaseDataContainer._last_state`` as a
        side effect. Call this before ``data_containers()`` — mchammer's
        ``_restart_ensemble`` reads ``_last_state`` on resume and the
        checkpoint write path depends on it being populated.

        Canonical pools (``SerialPool``, ``ProcessPool``) return a flat
        ``list[dict[str, Any]]``, one entry per replica. Wang-Landau pools
        (``SerialWangLandauPool``, ``ProcessWangLandauPool``) return a
        ``dict`` with ``"per_walker"`` and ``"group_state"`` keys.
        ``_write_checkpoint`` dispatches on the return type.
        """
        ...

    def shutdown(self) -> None:
        """Release any resources (worker processes, file handles, ...)."""
        ...


@runtime_checkable
class CanonicalPool(ReplicaPool, Protocol):
    """A ReplicaPool that carries a temperature per replica.

    Canonical-ensemble PT specialises on temperatures for the
    Metropolis exchange ratio; this subprotocol pins that
    expectation. Pools that drive non-canonical ensembles (e.g.
    Wang-Landau) implement a different subprotocol with
    ensemble-specific parameters instead.
    """

    @property
    def temperatures(self) -> Sequence[float]: ...

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully qualified name of the ensemble class driven by this pool."""
        ...

    @property
    def ensemble_kwargs_hash(self) -> str:
        """Deterministic hash of extra ensemble kwargs, or ``""`` if unavailable."""
        ...


@runtime_checkable
class WangLandauPool(ReplicaPool, Protocol):
    """A ReplicaPool driving Wang-Landau ensembles.

    Each replica owns a fixed energy window. The orchestrator queries
    per-window log-density-of-states via `log_g`; per-walker exchange
    acceptance reads `walker_log_g` and `walker_energy`.
    `converged_flags` reports per-replica convergence so the
    orchestrator can stop on global convergence.
    """

    @property
    def windows(self) -> Sequence[tuple[float | None, float | None]]: ...

    @property
    def energy_spacing(self) -> float: ...

    @property
    def flatness_mode(self) -> FlatnessMode:
        """Collective-halving flatness mode this pool drives."""
        ...

    @property
    def merge_cadence(self) -> MergeCadence:
        """Entropy-merge cadence this pool drives."""
        ...

    @property
    def one_over_t_gate(self) -> OneOverTGate:
        """1/t-schedule halving-phase gate this pool drives."""
        ...

    @property
    def bp_stall_multiple(self) -> float:
        """Stall-escape multiple this pool drives (consulted under flatness)."""
        ...

    @property
    def one_over_t_entry(self) -> OneOverTEntry:
        """1/t entry policy applied by this pool's walkers."""
        ...

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully qualified name of the ensemble class driven by this pool."""
        ...

    @property
    def ensemble_kwargs_hash(self) -> str:
        """Deterministic hash of extra ensemble kwargs, or ``""`` if unavailable."""
        ...

    def log_g(self, i: int, energy: float) -> float:
        """Return ln g at the given energy for replica i, or -inf out of window."""
        ...

    def walker_log_g(self, i: int, walker: int, energy: float) -> float:
        """Return ln g at ``energy`` for ``walker`` in window ``i`` (-inf if out)."""
        ...

    def converged_flags(self) -> np.ndarray:
        """Per-replica convergence (bool array, length len(self))."""
        ...

    def per_window_stats(self) -> list[dict[str, Any]]:
        """Per-window convergence metrics for monitoring callbacks.

        Returns one dict per window. The following keys are always
        present:

        * ``"fill_factor"`` (float): current WL fill factor.
        * ``"halvings"`` (int): number of fill-factor halvings
          completed (each marks one flattened histogram phase).
        * ``"histogram"`` (dict[int, int]): current-phase histogram
          keyed by bin index (summed across walkers for multi-walker
          windows). A halving zeroes the counts but retains the keys,
          so ``len(histogram)`` equals ``bins_known`` and does not
          reset.
        * ``"bins_visited"`` (int): number of distinct bins the
          window has reached via MC since window entry. Monotone over
          the run and unaffected by halvings.
        * ``"bins_filled"`` (int): number of bins with a positive
          count in the current histogram -- bins covered since the
          last halving (resets on each halve). For multi-walker
          windows this is the union of per-walker positive bins under
          ``flatness_mode="pooled"`` and the intersection under
          ``"per_walker"``.
        * ``"bins_known"`` (int): number of bins present in the
          histogram (seeded or visited).
        * ``"converged"`` (bool): whether this window has converged.
        * ``"phase"`` (str): current WL phase, ``"halving"`` or
          ``"1_over_t"``.
        * ``"recency_flatness"`` (float | None): ``min / mean`` over
          known bins of an EWMA recency-weighted visit histogram,
          measuring current sampling uniformity. The schedule-aware
          progress display shows it for ``1_over_t`` windows. ``None``
          before any bin has usable recent weight.
        * ``"schedule"`` (str): the WL schedule driving the window,
          ``"halving"`` or ``"1_over_t"``.

        Multi-walker windows additionally carry:

        * ``"flatness_mode"`` (str): ``"pooled"`` or ``"per_walker"``.
          The serial backend also attaches this to single-walker
          windows.
        * ``"per_walker_flat_min"`` (float | None): minimum over
          walkers of ``min(H_k) / mean(H_k)``, or ``None`` if any
          walker has no histogram yet.
        * ``"per_walker_breakdown"`` (list[dict]): one entry per
          walker with ``"filled"``, ``"known"``, and ``"flat_min"``
          (the last ``None`` for an empty or zero-mean histogram).
        """
        ...

    @property
    def merge_events(self) -> tuple[MergeEvent, ...]:
        """Per-halving merged-entropy series produced by the coordinator.

        See :class:`mchammer_pt.wl_merge_diagnostics.MergeEvent`. Empty
        for runs with no multi-walker slots, ``merge_cadence="never"``,
        or runs that have not yet triggered a halving merge. Resumed
        runs start with an empty list (the series is not persisted in
        checkpoints).
        """
        ...

    def per_window_data_containers(
        self,
    ) -> list[list[BaseDataContainer]]:
        """All data containers grouped by window.

        Returns one inner list per window, each containing one
        ``WangLandauDataContainer`` per walker.
        """
        ...

    def finalise_for_reporting(self) -> None:
        """End-of-run merge of per-walker entropies into a per-window estimate.

        Called at ``WL.run()`` exit. Writes the merged dict into every
        walker's ``_entropy`` so downstream readers see a consistent
        estimate regardless of which walker they sample. No-op for
        single-walker windows.
        """
        ...

    @property
    def is_open(self) -> bool:
        """Whether the pool can still accept commands.

        Returns ``False`` once :meth:`shutdown` has been called (process
        pool) or ``True`` always for in-process pools that do not model
        shutdown. ``WangLandauParallelTempering.run`` gates its
        end-of-run :meth:`finalise_for_reporting` call on this property
        so that an already-shutdown pool (e.g. after a worker error
        propagated through :meth:`advance_all`) does not raise a
        secondary ``RuntimeError`` masking the original failure.
        """
        ...


@runtime_checkable
class _ObserverAttach(Protocol):
    """Pool-agnostic observer-attach surface.

    Both `ObservablePool` (canonical) and `WangLandauObservablePool`
    (REWL) extend this. The orchestrator's `attach_observer` checks
    against this protocol so it accepts either pool kind.
    """

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to selected replicas."""
        ...

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        replicas: Sequence[int] | Literal["all"] = "all",
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer per selected replica.

        Escape hatch for observers whose instances do not pickle.
        """
        ...

    def attach_observer_factory(
        self,
        factory: Callable[[Replica], BaseObserver],
        *,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer constructed locally per replica.

        Required for observers whose constructors take icet objects
        (`ClusterSpace`, `ClusterExpansion`) that do not pickle and
        therefore cannot travel via `attach_observer` or
        `attach_observer_class`. The factory should reload the CE
        from disk via
        ``ClusterExpansion.read(replica.cluster_expansion_path)``,
        which yields an unmutated ``ClusterSpace``.
        """
        ...

    def get_observers(self, replica_index: int) -> dict[str, BaseObserver]:
        """Return a snapshot of the observers attached to one replica.

        Keyed by observer tag; values are independent copies via
        pickle round-trip so mutations on the returned objects do
        not affect the pool's running state.

        Raises:
            IndexError: if ``replica_index`` is out of range.
        """
        ...


@runtime_checkable
class ObservablePool(CanonicalPool, _ObserverAttach, Protocol):
    """A `CanonicalPool` that can have mchammer observers attached.

    Separate protocol because not every pool implementation can carry
    observer instances across its execution boundary. Implementations
    that satisfy this protocol get observer-attach for free via the
    `_ObserverAttach` mixin; those that don't (e.g. a hypothetical
    pool whose workers can't serialise observers) satisfy only
    `CanonicalPool`.
    """


@runtime_checkable
class _RecorderAttach(Protocol):
    """Pool-agnostic recorder-attach surface for REWL pools.

    `WangLandauObservablePool` extends this. The orchestrator's
    `record_observable` checks against this protocol so it can guard
    that the pool supports per-bin moment recording.
    """

    def record_observable(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer for per-bin microcanonical moment accumulation."""
        ...


@runtime_checkable
class WangLandauObservablePool(
    WangLandauPool, _ObserverAttach, _RecorderAttach, Protocol
):
    """A `WangLandauPool` that can have mchammer observers attached.

    Mirrors `ObservablePool` for REWL. Observers fire inside each
    replica's WL `run(block_size)` between exchange proposals, so a
    user can record any per-step quantity over the WL trajectory and
    feed it into post-processing (e.g. via
    `mchammer.data_containers.wang_landau_data_container.get_average_observables_wl`
    for thermodynamic averages against the stitched density of states).
    """
