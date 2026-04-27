"""Test fixtures: CanonicalEnsemble subclasses for issue-6 tests.

These live in a module file (not inside test functions) so that
ProcessPool spawn workers can re-import them by fully qualified name.
"""

from __future__ import annotations

from typing import Any

from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]


class TaggedCanonicalEnsemble(CanonicalEnsemble):
    """CanonicalEnsemble subclass that stores an extra ``tag`` kwarg.

    Used to verify that ``ensemble_kwargs`` are forwarded through
    Replica / orchestrator / pool construction sites unchanged.
    """

    def __init__(self, *, tag: str, **kwargs: Any) -> None:
        self.tag = tag
        super().__init__(**kwargs)


class HighAcceptanceCanonicalEnsemble(CanonicalEnsemble):
    """CanonicalEnsemble subclass that accepts every proposed swap.

    Overrides `_acceptance_condition` to short-circuit the Metropolis
    check. Used by the Task 10 integration test to confirm that a
    custom `CanonicalEnsemble` subclass — issue #6's stated use case
    — rides the parallel-tempering machinery end-to-end.
    """

    def _acceptance_condition(self, potential_diff: float) -> bool:
        return True
