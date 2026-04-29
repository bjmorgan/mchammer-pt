"""In-process replica pool: advances replicas sequentially in the caller."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from ..replica import Replica
from ._imports import _resolve_replicas


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
        self._replicas[j].set_occupations(occ_i)

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

        On ``SerialPool``, ``replica.cluster_expansion_path`` is ``None``
        unless you passed ``cluster_expansion_path=`` to ``Replica.__init__``.
        ``ProcessPool`` auto-populates the path on every worker.

        Do **not** reach for
        ``replica.ensemble.calculator.cluster_expansion``: the
        calculator mutates it during runs, and observers that store a
        reference will see wrong-length cluster vectors at observation
        time.

        A factory written for ``SerialPool`` runs unchanged on
        ``ProcessPool``, where it must additionally be a top-level function
        or class method importable by fully qualified name.
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

    def shutdown(self) -> None:
        return None
