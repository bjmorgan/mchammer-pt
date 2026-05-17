"""Wang-Landau ensemble subclass with halving delegated to a coordinator."""

from __future__ import annotations

import numpy as np
from mchammer.ensembles import WangLandauEnsemble


class CoordinatedWangLandauEnsemble(WangLandauEnsemble):  # type: ignore[misc]
    """`WangLandauEnsemble` with internal halving suppressed.

    Bin counters and periodic entropy reshift behave identically to
    upstream. The flatness check, halving, ``_fill_factor_history``
    recording, histogram reset, BP-phase transition,
    ``_entropy_history`` snapshots, and ``_converged`` writes in the
    1/t branch are all suppressed; ``WangLandauWindowGroup`` owns
    those decisions and applies them via ``WangLandauReplica``.
    """

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
            assert entry is not None, (
                "_phase == '1_over_t' implies _window_entry_step is set"
            )
            t = self.step - entry + 1
            self._fill_factor = 1.0 / t

        self._entropy[bin_cur] = (
            self._entropy.get(bin_cur, 0) + self._fill_factor
        )
        self._histogram[bin_cur] = self._histogram.get(bin_cur, 0) + 1

        if (
            self.step > 0
            and self.step % self._flatness_check_interval == 0
            and self._reached_energy_window
        ):
            ref = np.min(list(self._entropy.values()))
            for k in self._entropy:
                self._entropy[k] -= ref
