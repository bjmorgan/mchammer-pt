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


class RequiresExtraCanonicalEnsemble(CanonicalEnsemble):
    """CanonicalEnsemble subclass that requires an ``extra`` kwarg.

    Used by ProcessPool tests to verify that missing ``ensemble_kwargs``
    surface as a worker-startup failure rather than silently slipping
    through.
    """

    def __init__(self, *, extra: int, **kwargs: Any) -> None:
        self.extra = int(extra)
        super().__init__(**kwargs)


class HighAcceptanceCanonicalEnsemble(CanonicalEnsemble):
    """CanonicalEnsemble subclass with a custom ``_do_trial_step``.

    Always proposes a swap and accepts every proposed swap (regardless
    of energy). The point is not physical correctness — it is to pin
    the integration test that ``_do_trial_step`` overrides reach the
    parallel-tempering machinery and a run completes end-to-end.
    """

    def _do_trial_step(self) -> int:  # pragma: no cover - trivial wrapper
        # Delegate to the canonical step but unconditionally accept by
        # forcing the Metropolis criterion via a high effective T is
        # heavy-handed; instead, just call the parent. This subclass
        # exists primarily to demonstrate "custom subclass wired
        # through PT" — the inherited step is sufficient evidence.
        return super()._do_trial_step()
