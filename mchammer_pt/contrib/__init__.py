"""Optional glue for third-party WL ensemble subclasses.

The classes here combine ``CoordinatedWangLandauEnsemble`` (which
suppresses internal halving so the coordinator can own it) with WL
ensemble subclasses from other libraries. They live in ``contrib`` so
that ``mchammer_pt`` does not hard-depend on those libraries — each
class is defined only if its third-party dependency imports cleanly.

To use a contrib class with ``WangLandauParallelTempering``, pass it
as ``ensemble_cls``:

.. code-block:: python

    from mchammer_pt.contrib import CoordinatedCustomWangLandauEnsemble

    pt = WangLandauParallelTempering(
        ...,
        ensemble_cls=CoordinatedCustomWangLandauEnsemble,
        ensemble_kwargs={"moves": [(my_move, 1.0)]},
    )

Each class is a normal module-level subclass and so pickles cleanly
across the process-pool spawn boundary.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from mchammer_moves import CustomWangLandauEnsemble

    from ..wl_ensemble import CoordinatedWangLandauEnsemble

    class CoordinatedCustomWangLandauEnsemble(
        CoordinatedWangLandauEnsemble, CustomWangLandauEnsemble
    ):
        """``CustomWangLandauEnsemble`` driven by the WL coordinator.

        Combines ``mchammer_moves.CustomWangLandauEnsemble`` (custom
        move list via ``_do_trial_step``) with
        ``CoordinatedWangLandauEnsemble`` (halving suppressed; owned
        by the coordinator). The two parents override disjoint
        methods, so the MRO has no conflict.
        """

    __all__.append("CoordinatedCustomWangLandauEnsemble")
except ImportError:  # mchammer-moves not installed; class unavailable
    pass
