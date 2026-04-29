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


class ConstructionCounter(BaseObserver):
    """Observer whose every construction is recorded.

    Class-level ``n_constructions`` increments on each ``__init__``;
    each instance stores its own ``construction_id`` (1, 2, 3, ...).
    Used to pin that ``attach_observer_class`` constructs ``1 + N``
    times (one dry-run probe + one per selected replica) and that
    the *probe* (id=1) is discarded -- only later-id instances should
    be registered on replicas.
    """

    n_constructions = 0

    def __init__(self, interval: int, tag: str = "construction_counter") -> None:
        super().__init__(interval=interval, return_type=int, tag=tag)
        ConstructionCounter.n_constructions += 1
        self.construction_id = ConstructionCounter.n_constructions
        self.n_calls = 0

    def get_observable(self, structure: Any) -> int:
        self.n_calls += 1
        # Encode this instance's construction_id in the observed value
        # so a registered probe (id=1) shows up clearly in the data.
        return self.construction_id


class NotAnObserver:
    """A class that is *not* a BaseObserver subclass.

    Used to verify that ``attach_observer_class`` rejects classes whose
    instances are not ``BaseObserver``s, with a clear ``TypeError``.
    """

    def __init__(self, interval: int) -> None:
        self.interval = interval


class LambdaAccumulatingObs(BaseObserver):
    """Observer that stashes a lambda on its first call.

    Picklable before ``get_observable`` runs; non-picklable afterwards.
    Used to test that ``get_observers`` raises ``TypeError`` when the
    live observer dict cannot be serialised.
    """

    def __init__(self, interval: int) -> None:
        super().__init__(interval=interval, return_type=int, tag="stash")
        self.junk: Any = None

    def get_observable(self, structure: Any) -> int:
        self.junk = lambda x: x  # noqa: E731
        return 0


def stateful_counter_factory(replica: Replica) -> BaseObserver:
    """Factory used by ProcessPool factory-path tests.

    Returns a ``StatefulCounter`` whose tag encodes the replica's
    temperature, so test assertions can confirm each worker really
    saw its own replica.
    """
    return StatefulCounter(interval=10, tag=f"counter_T{int(replica.temperature)}")


def cluster_count_factory(replica: Replica) -> BaseObserver:
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


def factory_returning_non_observer(replica: Replica) -> object:
    """Factory whose return type is wrong; pins the isinstance check."""
    return "not an observer"


def factory_fails_on_400k(replica: Replica) -> BaseObserver:
    """Factory that fails only on the 400 K replica.

    Used to exercise the mixed-success drain path: with workers at
    [300, 400, 500], worker 0 returns OK, worker 1 raises a TypeError
    (caught by the outer try/except and reported as ERR), and worker
    2 has an OK queued that the parent must drain before SHUTDOWN.
    """
    if replica.temperature == 400.0:
        return "intentional failure"  # type: ignore[return-value]
    return StatefulCounter(interval=10)
