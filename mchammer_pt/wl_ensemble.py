"""Wang-Landau ensemble subclass with halving delegated to a coordinator."""

from __future__ import annotations

import numpy as np
from mchammer.ensembles import WangLandauEnsemble


class CoordinatedWangLandauEnsemble(WangLandauEnsemble):
    """`WangLandauEnsemble` with internal halving suppressed.

    Bin counters and periodic entropy reshift behave identically to
    upstream. The following behaviours are suppressed relative to
    upstream: the flatness check, halving, `_fill_factor_history`
    recording, histogram reset, BP-phase transition, `_entropy_history`
    snapshots, and `_converged` writes in the 1/t branch.
    A `WangLandauWindowGroup` coordinator runs the flatness check
    across all walkers in the window and triggers collective halving
    via `WangLandauReplica.force_halve()`.

    Direct construction is supported; in normal use this class is
    constructed by `WangLandauReplica`.
    """

    def _update_entropy(self, bin_cur: int) -> None:
        if (
            self._schedule == "1_over_t"
            and self._reached_energy_window
            and self._window_entry_step is None
        ):
            self._window_entry_step = self.step

        if self._phase == "1_over_t":
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
