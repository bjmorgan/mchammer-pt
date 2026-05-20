"""Per-halving merge diagnostics for the WL coordinator.

When the collective halving gate fires for a multi-walker slot, the
coordinator computes a merged entropy dict (see
:func:`wl_coordinator.merge_entropies`) and writes it into every
walker as the new baseline. The merged dict is otherwise overwritten
by walker updates before the next halving, so the orchestrator
records it here for post-hoc compression diagnostics.

The corresponding per-walker pre-merge entropy is already preserved
by mchammer's ``WangLandauEnsemble._entropy_history``, keyed by the
same MC step as :class:`MergeEvent.step`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MergeEvent:
    """One merged-entropy snapshot for one slot at one halving.

    Attributes:
        slot_index: index of the slot (window) within the pool.
        step: master MC step at the halving; matches the key in each
            walker's ``_fill_factor_history`` and ``_entropy_history``.
        merged_entropy: maps energy-bin index to merged log-density of
            states, normalised so the minimum value is zero (the icet
            ``WangLandauEnsemble`` convention).
    """

    slot_index: int
    step: int
    merged_entropy: dict[int, float]
