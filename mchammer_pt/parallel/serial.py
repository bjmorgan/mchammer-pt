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

from ..replica import Replica
from ..wl_replica import WangLandauReplica
from ._imports import _resolve_replicas

if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )

    from ..wl_window_group import WangLandauWindowGroup


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
    """

    def __init__(
        self,
        replicas: Sequence[WangLandauReplica | WangLandauWindowGroup],
        *,
        energy_spacing: float,
    ) -> None:
        self._replicas: list[WangLandauReplica | WangLandauWindowGroup] = list(replicas)
        self._energy_spacing = float(energy_spacing)
        for r in self._replicas:
            if r.energy_spacing != self._energy_spacing:
                raise ValueError(
                    f"replica energy_spacing {r.energy_spacing} does not "
                    f"match pool energy_spacing {self._energy_spacing}"
                )

    def __len__(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> list[WangLandauReplica | WangLandauWindowGroup]:
        return list(self._replicas)

    def _slot_ensemble(self, slot: WangLandauReplica | WangLandauWindowGroup) -> Any:
        """Return the mchammer ensemble for a slot.

        Works for both WangLandauReplica and WangLandauWindowGroup slots.
        """
        from ..wl_window_group import WangLandauWindowGroup

        if isinstance(slot, WangLandauWindowGroup):
            return slot._replicas[0].ensemble
        return slot.ensemble

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully qualified name of the ensemble class used by replicas."""
        cls = type(self._slot_ensemble(self._replicas[0]))
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

    def advance_all(self, n_steps: int) -> None:
        for r in self._replicas:
            r.advance(n_steps)

    def current_energies(self) -> np.ndarray:
        return np.array(
            [r.current_energy() for r in self._replicas], dtype=np.float64
        )

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

    def log_g(self, i: int, energy: float) -> float:
        return self._replicas[i].log_g(energy)

    def log_g_pair(
        self, i: int, j: int, E_i: float, E_j: float,
    ) -> tuple[float, float, float, float]:
        r_i, r_j = self._replicas[i], self._replicas[j]
        return (
            r_i.log_g(E_i),
            r_i.log_g(E_j),
            r_j.log_g(E_i),
            r_j.log_g(E_j),
        )

    def converged_flags(self) -> np.ndarray:
        return np.array([r.converged for r in self._replicas], dtype=bool)

    def per_window_stats(self) -> list[dict[str, Any]]:
        return [r.window_stats() for r in self._replicas]

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
        from ..wl_window_group import WangLandauWindowGroup

        for i in target_indices:
            slot = self._replicas[i]
            if isinstance(slot, WangLandauWindowGroup):
                slot.attach_observer_class(cls, *args, **kwargs)
            else:
                slot.attach_mchammer_observer(cls(*args, **kwargs))

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
        from ..wl_window_group import WangLandauWindowGroup

        for i in target_indices:
            slot = self._replicas[i]
            if isinstance(slot, WangLandauWindowGroup):
                slot.attach_observer_factory(factory)
            else:
                observer = factory(slot)
                if not isinstance(observer, BaseObserver):
                    raise TypeError(
                        f"attach_observer_factory: factory returned "
                        f"{type(observer).__name__}, not a BaseObserver"
                    )
                slot.attach_mchammer_observer(observer)

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
        live = self._slot_ensemble(slot).observers
        try:
            return pickle.loads(pickle.dumps(live))
        except Exception as exc:
            raise TypeError(
                f"observer dict for replica {replica_index} could not be "
                f"round-tripped through pickle ({exc})"
            ) from exc

    def data_containers(self) -> list[WangLandauDataContainer]:
        return [r.data_container() for r in self._replicas]

    def per_window_data_containers(self) -> list[list[WangLandauDataContainer]]:
        """All data containers grouped by window slot.

        Returns a list of length n_windows; each entry is a list of
        WangLandauDataContainer instances — one per walker for
        WangLandauWindowGroup slots, one for WangLandauReplica slots.
        """
        from ..wl_window_group import WangLandauWindowGroup

        return [
            r.all_data_containers() if isinstance(r, WangLandauWindowGroup)
            else [r.data_container()]
            for r in self._replicas
        ]

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]]:
        return [r.snapshot_for_checkpoint() for r in self._replicas]

    def shutdown(self) -> None:
        return None
