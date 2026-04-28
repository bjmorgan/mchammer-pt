"""Test fixtures: BaseObserver subclasses for observer-support tests.

These live in a module file (not inside test functions) so that
ProcessPool spawn workers can re-import them by fully qualified name.
"""

from __future__ import annotations

from typing import Any

from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)


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
