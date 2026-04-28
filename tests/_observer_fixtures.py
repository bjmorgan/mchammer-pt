"""Test fixtures: BaseObserver subclasses for observer-support tests.

These live in a module file (not inside test functions) so that
ProcessPool spawn workers can re-import them by fully qualified name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

if TYPE_CHECKING:
    from mchammer_pt.replica import Replica


class StatefulCounter(BaseObserver):
    """Observer that counts its own invocations.

    Used to verify per-replica independence: if the same instance is
    shared across replicas the counter accumulates across replicas; if
    each replica has its own copy, each counter advances at its own
    pace.
    """

    def __init__(self, interval: int, tag: str = "counter") -> None:
        super().__init__(interval=interval, return_type=int, tag=tag)
        self.n_calls = 0

    def get_observable(self, structure: Any) -> int:
        self.n_calls += 1
        return self.n_calls


class TaggedObserver(BaseObserver):
    """Observer that stores an extra ``label`` constructor kwarg.

    Used to verify that ``attach_observer_class`` forwards positional
    and keyword arguments through to the worker-side constructor
    unchanged.
    """

    def __init__(self, interval: int, *, label: str, tag: str = "tagged") -> None:
        super().__init__(interval=interval, return_type=int, tag=tag)
        self.label = label

    def get_observable(self, structure: Any) -> int:
        return 0


class BadInitObserver(BaseObserver):
    """Observer whose constructor always raises.

    Used to verify that ``attach_observer_class``'s parent-side dry-run
    surfaces the user's own exception with a parent-process traceback,
    not a worker-wrapped ``RuntimeError``.
    """

    def __init__(self, interval: int) -> None:
        raise ValueError("deliberate constructor failure")

    def get_observable(self, structure: Any) -> int:
        return 0


class NotAnObserver:
    """A class that is *not* a BaseObserver subclass.

    Used to verify that ``attach_observer_class`` rejects classes whose
    instances are not ``BaseObserver``s, with a clear ``TypeError``.
    """

    def __init__(self, interval: int) -> None:
        self.interval = interval


def stateful_counter_factory(replica: "Replica") -> BaseObserver:
    """Factory used by ProcessPool factory-path tests.

    Returns a ``StatefulCounter`` whose tag encodes the replica's
    temperature, so test assertions can confirm each worker really
    saw its own replica.
    """
    return StatefulCounter(interval=10, tag=f"counter_T{int(replica.temperature)}")


def cluster_count_factory(replica: "Replica") -> BaseObserver:
    """Factory using icet objects only available inside the worker.

    The cluster-space and structure references come from the worker's
    own Replica; neither needs to pickle.
    """
    from mchammer.observers import ClusterCountObserver  # type: ignore[import-untyped]

    cs = replica.ensemble.calculator.cluster_expansion.get_cluster_space_copy()
    return ClusterCountObserver(
        cluster_space=cs,
        structure=replica.ensemble.structure,
        interval=20,
    )


def factory_returning_non_observer(replica: "Replica") -> object:
    """Factory whose return type is wrong; pins the isinstance check."""
    return "not an observer"
