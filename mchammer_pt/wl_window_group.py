"""Multi-walker Wang-Landau window group."""

from __future__ import annotations

import numpy as np

from .wl_replica import WangLandauReplica


class WangLandauWindowGroup:
    """W WangLandauReplica instances sharing one energy window.

    Each cycle: all W replicas advance independently, fill factors are
    synchronised (detect-and-resync), and entropies are averaged bin-wise
    so every replica starts the next cycle from the best shared ln g estimate.

    Args:
        replicas: W pre-constructed WangLandauReplica instances, all
            with the same energy window and energy spacing.
        random_seed: seed for the exchange-walker selection RNG.
    """

    def __init__(
        self,
        replicas: list[WangLandauReplica],
        *,
        random_seed: int,
    ) -> None:
        if len(replicas) < 1:
            raise ValueError(
                "WangLandauWindowGroup requires at least one replica"
            )
        self._replicas = list(replicas)
        self._rng = np.random.default_rng(int(random_seed))
        self._exchange_idx: int = 0

    def _sync_fill_factors(self) -> None:
        """Force-halve lagging replicas to match the most-halved one.

        Detect-and-resync: after mchammer's per-walker auto-halving,
        replicas may be at different fill factors. We count halvings via
        len(_fill_factor_history) and force-halve any replica that is
        behind the leader, mirroring mchammer's own halving logic
        (append current fill factor to history, halve, clear histogram).
        """
        halvings = [len(r.ensemble._fill_factor_history) for r in self._replicas]
        target = max(halvings)
        for r, h in zip(self._replicas, halvings, strict=True):
            for _ in range(target - h):
                r.ensemble._fill_factor_history.append(r.ensemble._fill_factor)
                r.ensemble._fill_factor /= 2.0
                r.ensemble._histogram.clear()

    def _merge_entropies(self) -> None:
        """Average ln g bin-wise across all replicas and write back to each.

        Bins not yet visited by a replica contribute 0.0 (mchammer's
        default for unvisited bins via _entropy.get(bin, 0.0)).
        """
        all_bins: set[int] = set()
        for r in self._replicas:
            all_bins.update(r.ensemble._entropy.keys())
        n = len(self._replicas)
        merged = {
            b: sum(r.ensemble._entropy.get(b, 0.0) for r in self._replicas) / n
            for b in all_bins
        }
        for r in self._replicas:
            r.ensemble._entropy = dict(merged)
