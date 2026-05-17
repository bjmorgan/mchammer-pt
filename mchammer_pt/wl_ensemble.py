"""Wang-Landau ensemble subclass with halving delegated to a coordinator."""

from __future__ import annotations

import numpy as np
from mchammer.ensembles import WangLandauEnsemble


class CoordinatedWangLandauEnsemble(WangLandauEnsemble):
    """`WangLandauEnsemble` with internal halving suppressed.

    Bin counters and periodic entropy reshift behave identically to
    upstream. The flatness check, halving, ``_fill_factor_history``
    recording, histogram reset, BP-phase transition,
    ``_entropy_history`` snapshots, and ``_converged`` writes in the
    1/t branch are all suppressed; ``WangLandauWindowGroup`` owns
    those decisions and applies them via ``WangLandauReplica``.
    """

    def _update_entropy(self, bin_cur: int) -> None:
        # Mainline icet's WangLandauEnsemble lacks _schedule and _phase
        # (the patched fork added them for the 1/t schedule). Treat
        # missing attributes as ``schedule="halving"`` /
        # ``phase="halving"`` and skip the 1/t prologue.
        schedule = getattr(self, "_schedule", "halving")
        phase = getattr(self, "_phase", "halving")
        if (
            schedule == "1_over_t"
            and self._reached_energy_window
            and self._window_entry_step is None
        ):
            self._window_entry_step = self.step

        if phase == "1_over_t":
            t = self.step - self._window_entry_step + 1
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
