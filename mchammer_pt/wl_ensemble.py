"""Wang-Landau ensemble subclass with halving delegated to a coordinator."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from mchammer.ensembles import WangLandauEnsemble


def _validate_recency_visits_per_bin(value: int) -> int:
    """Return ``value`` as an int, or raise if it is not a positive integer.

    Rejects non-integer values (e.g. ``2.5``) rather than silently
    truncating them, so the error message's promise of an integer holds.
    Integer-valued floats such as ``1e3`` are accepted.
    """
    if int(value) != value or int(value) <= 0:
        raise ValueError(
            f"recency_visits_per_bin must be a positive integer; "
            f"got {value!r}"
        )
    return int(value)


class CoordinatedWangLandauEnsemble(WangLandauEnsemble):  # type: ignore[misc]
    """`WangLandauEnsemble` with internal halving suppressed.

    Bin counters and periodic entropy reshift behave identically to
    upstream. The flatness check, halving, ``_fill_factor_history``
    recording, histogram reset, BP-phase transition,
    ``_entropy_history`` snapshots, and ``_converged`` writes in the
    1/t branch are all suppressed; ``WangLandauWindowGroup`` owns
    those decisions and applies them via ``WangLandauReplica``.
    """

    def __init__(
        self, *args: Any, recency_visits_per_bin: int = 1000, **kwargs: Any
    ) -> None:
        recency = _validate_recency_visits_per_bin(recency_visits_per_bin)
        super().__init__(*args, **kwargs)
        # Bins the walker has reached via `_update_entropy` since
        # window entry. Populated only by that method (guarded on
        # `_reached_energy_window`).
        self._visited_bins: set[int] = set()
        # EWMA recency state: per-bin weight and the step it was last
        # updated. Decayed lazily (only the visited bin is touched per
        # step; all known bins are decayed at read time).
        self._recency_visits_per_bin: int = recency
        self._recent_weight: dict[int, float] = {}
        self._recent_last_step: dict[int, int] = {}

    def _update_entropy(self, bin_cur: int) -> None:
        # ``_window_entry_step`` is inherited from upstream's
        # ``WangLandauEnsemble`` (typed Any to mypy because mchammer
        # has no stubs); narrow it explicitly.
        entry: int | None = self._window_entry_step  # type: ignore[has-type]
        if (
            self._schedule == "1_over_t"
            and self._reached_energy_window
            and entry is None
        ):
            self._window_entry_step = self.step
            entry = self.step

        if self._phase == "1_over_t":
            # By construction, ``_phase == '1_over_t'`` only after
            # ``_window_entry_step`` has been set (the coordinator
            # flips the phase post-entry). Narrow for the type
            # checker.
            t = self.step - cast(int, entry) + 1
            self._fill_factor = 1.0 / t

        self._entropy[bin_cur] = (
            self._entropy.get(bin_cur, 0) + self._fill_factor
        )
        self._histogram[bin_cur] = self._histogram.get(bin_cur, 0) + 1

        if self._reached_energy_window:
            self._visited_bins.add(bin_cur)
            self._record_recency_visit(bin_cur, int(self.step))

        if (
            self.step > 0
            and self.step % self._flatness_check_interval == 0
            and self._reached_energy_window
        ):
            ref = np.min(list(self._entropy.values()))
            for k in self._entropy:
                self._entropy[k] -= ref

    def _recency_alpha(self) -> float:
        """EWMA rate ``1 / tau`` with ``tau = recency_visits_per_bin * N``.

        ``N`` is the current known-bin count (``len(self._histogram)``,
        at least 1), so early in a run the effective averaging window is
        shorter than its final value because ``N`` is still growing.
        """
        n_bins = max(1, len(self._histogram))
        return 1.0 / (self._recency_visits_per_bin * n_bins)

    def _record_recency_visit(self, bin_cur: int, step: int) -> None:
        """Decay this bin's weight to ``step``, then add one visit."""
        alpha = self._recency_alpha()
        last = self._recent_last_step.get(bin_cur, step)
        decayed = self._recent_weight.get(bin_cur, 0.0) * (
            (1.0 - alpha) ** (step - last)
        )
        self._recent_weight[bin_cur] = decayed + 1.0
        self._recent_last_step[bin_cur] = step

    def recency_effective_weights(
        self, step: int | None = None
    ) -> dict[int, float]:
        """Per-known-bin EWMA weights decayed to ``step`` (default now).

        Keys are the current known bins (``self._histogram`` keys); a
        bin with no recorded visit reads 0.0.
        """
        if step is None:
            step = int(self.step)
        alpha = self._recency_alpha()
        weights: dict[int, float] = {}
        for b in self._histogram:
            w = self._recent_weight.get(b, 0.0)
            if w:
                w *= (1.0 - alpha) ** (step - self._recent_last_step[b])
            weights[b] = w
        return weights
