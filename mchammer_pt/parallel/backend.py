"""Protocols for replica pools and observable replica pools.

A `ReplicaPool` owns the state of every replica in a parallel-tempering
run and exposes the operations the orchestrator needs: advance, query
energies, atomically swap configurations, and return native mchammer
data containers at end-of-run.

Pool implementations vary in whether they can carry user-supplied
`mchammer.BaseObserver` instances across their execution boundary.
Those that can implement the `ObservablePool` extension; those that
cannot (e.g. process-parallel pools whose workers live in separate
interpreters) satisfy only `ReplicaPool`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..replica import Replica

import numpy as np
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)


@runtime_checkable
class ReplicaPool(Protocol):
    """Owns replica state; exposes operations the orchestrator needs."""

    def __len__(self) -> int: ...

    @property
    def temperatures(self) -> Sequence[float]: ...

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

        Failure semantics: if any worker-boundary operation raises
        mid-swap, the pool is left in an undefined partial-swap state
        and the enclosing run should abort. Resumable-from-partial-swap
        is out of v0.1 scope; the orchestrator propagates the
        exception unchanged.
        """
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

    def shutdown(self) -> None:
        """Release any resources (worker processes, file handles, ...)."""
        ...


@runtime_checkable
class ObservablePool(ReplicaPool, Protocol):
    """A `ReplicaPool` that can have mchammer observers attached.

    Separate protocol because not every pool implementation can carry
    observer instances across its execution boundary. Pool implementations
    that support observer forwarding implement this protocol; those that
    don't satisfy only `ReplicaPool` and force the user to use a different
    pool type to attach observers.
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
        `attach_observer_class`.
        """
        ...
