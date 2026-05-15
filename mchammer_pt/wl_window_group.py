"""Multi-walker Wang-Landau window group."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from mchammer.observers.base_observer import BaseObserver

from .wl_replica import WangLandauReplica

_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED = (
    "checkpointing is not yet supported for n_walkers_per_window > 1; "
    "pass data_container_file=None and avoid save_checkpoint() / "
    "attach_checkpoint_writer() when using multiple walkers per window."
)


def merge_entropies(entropies: list[dict[int, float]]) -> dict[int, float]:
    """Average bin-wise entropy estimates across multiple walkers.

    Args:
        entropies: list of {bin_index: entropy_value} dicts from each walker.

    Returns:
        Merged entropy dict with bin-wise averages; missing bins contribute 0.0.
        Unvisited bins are deliberately suppressed: frontier regions entered by
        only a subset of walkers contribute a reduced entropy estimate until all
        walkers reach them.
    """
    if not entropies:
        return {}
    all_bins: set[int] = set()
    for e in entropies:
        all_bins.update(e.keys())
    n = len(entropies)
    return {b: sum(e.get(b, 0.0) for e in entropies) / n for b in all_bins}


if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )


class WangLandauWindowGroup:
    """A group of independent Wang-Landau walkers sharing one energy window.

    Each cycle: all walkers advance independently, fill factors are
    synchronised (detect-and-resync), and entropies are averaged bin-wise
    so every walker starts the next cycle from the best shared ln g estimate.

    Args:
        replicas: pre-constructed WangLandauReplica instances, all
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
        if len(replicas) > 1:
            w0 = replicas[0].energy_window
            s0 = replicas[0].energy_spacing
            for r in replicas[1:]:
                if r.energy_window != w0 or r.energy_spacing != s0:
                    raise ValueError(
                        "all replicas in a WangLandauWindowGroup must share "
                        "the same energy window and spacing"
                    )
        self._replicas = list(replicas)
        self._rng = np.random.default_rng(int(random_seed))
        self._exchange_idx: int = 0

    def _sync_fill_factors(self) -> None:
        """Force-halve lagging replicas to match the most-halved one.

        Detect-and-resync: after mchammer's per-walker auto-halving,
        replicas may be at different fill factors. We count halvings via
        len(_fill_factor_history) and force-halve any replica that is
        behind the leader.
        """
        halvings = [len(r.ensemble._fill_factor_history) for r in self._replicas]
        target = max(halvings)
        for r, h in zip(self._replicas, halvings, strict=True):
            for _ in range(target - h):
                r.force_halve()

    def _merge_entropies(self) -> None:
        """Average ln g bin-wise across all replicas and write back to each."""
        merged = merge_entropies(
            [dict(r.ensemble._entropy) for r in self._replicas]
        )
        for r in self._replicas:
            r.ensemble._entropy = dict(merged)

    def advance(self, n_steps: int) -> None:
        """Advance all W replicas, sync fill factors, merge entropies.

        Re-selects the exchange walker after each cycle.
        """
        for r in self._replicas:
            r.advance(int(n_steps))
        self._sync_fill_factors()
        self._merge_entropies()
        self._exchange_idx = int(self._rng.integers(0, len(self._replicas)))

    @property
    def ensemble(self) -> Any:
        """The mchammer ensemble for the first walker.

        Used by pool-level metadata queries (ensemble class, observer
        snapshot). All walkers share the same ensemble class and kwargs.
        For read-only metadata queries only; use `attach_mchammer_observer`,
        `attach_observer_class`, or `attach_observer_factory` to add
        observers to all walkers.
        """
        return self._replicas[0].ensemble

    @property
    def energy_window(self) -> tuple[float | None, float | None]:
        return self._replicas[0].energy_window

    @property
    def energy_spacing(self) -> float:
        return self._replicas[0].energy_spacing

    @property
    def cluster_expansion_path(self) -> str | None:
        return self._replicas[0].cluster_expansion_path

    def current_energy(self) -> float:
        return self._replicas[self._exchange_idx].current_energy()

    def current_occupations(self) -> np.ndarray:
        return self._replicas[self._exchange_idx].current_occupations()

    def set_occupations(self, occupations: np.ndarray) -> None:
        self._replicas[self._exchange_idx].set_occupations(occupations)

    def log_g(self, energy: float) -> float:
        return self._replicas[0].log_g(energy)

    @property
    def converged(self) -> bool:
        return all(r.converged for r in self._replicas)

    def data_container(self) -> WangLandauDataContainer:
        """Data container for the first walker in the group."""
        return self._replicas[0].data_container()

    def all_data_containers(self) -> list[WangLandauDataContainer]:
        """All per-walker data containers."""
        return [r.data_container() for r in self._replicas]

    def window_stats(self) -> dict[str, Any]:
        """Per-window convergence metrics: fill_factor, halvings, histogram, converged.

        fill_factor and halvings are taken from replica 0 (all in sync after
        advance); histogram is the sum across all walkers.
        """
        e0 = self._replicas[0].ensemble
        combined_hist: dict[int, int] = {}
        for r in self._replicas:
            for k, v in r.ensemble._histogram.items():
                combined_hist[k] = combined_hist.get(k, 0) + v
        return {
            "fill_factor": float(e0._fill_factor),
            "halvings": max(0, len(e0._fill_factor_history) - 1),
            "histogram": combined_hist,
            "converged": self.converged,
        }

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError(_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED)

    def attach_mchammer_observer(self, observer: BaseObserver) -> None:
        """Attach observer to all W replicas; each receives its own copy."""
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        for r in self._replicas:
            r.attach_mchammer_observer(pickle.loads(blob))

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to every walker in the group.

        Each of the W replicas receives its own ``cls(*args, **kwargs)``
        instance, ensuring independent per-walker state.
        """
        for r in self._replicas:
            r.attach_mchammer_observer(cls(*args, **kwargs))

    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
    ) -> None:
        """Attach an observer constructed per walker via ``factory``.

        ``factory(replica)`` is called once per walker and must return a
        fresh ``BaseObserver``.
        """
        for r in self._replicas:
            observer = factory(r)
            if not isinstance(observer, BaseObserver):
                raise TypeError(
                    f"attach_observer_factory: factory returned "
                    f"{type(observer).__name__}, not a BaseObserver"
                )
            r.attach_mchammer_observer(observer)
