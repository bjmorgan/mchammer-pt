"""Abstract parallel-tempering orchestrator.

`BaseParallelTempering` drives the cycle loop, records per-cycle
observations, and coordinates exchange proposals. All replica state
lives in the pool (`ReplicaPool` or `ObservablePool`); the orchestrator
routes queries through it and never holds replica state directly.

Ensemble-specific subclasses override exactly one method:
`_log_prob_ratio(i, j)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from .callbacks import ExchangeCallback
from .exchange import metropolis_accept, pair_set_for_cycle
from .history import ExchangeHistory
from .parallel.backend import ObservablePool, ReplicaPool


class BaseParallelTempering(ABC):
    """Abstract PT orchestrator.

    Args:
        pool: a `ReplicaPool` owning one replica per temperature, in
            ascending-T order. If the pool is an `ObservablePool`,
            `attach_observer` will forward to it; otherwise calling
            `attach_observer` raises `TypeError`.
        block_size: MC trial steps per replica per cycle.
        random_seed: master seed for the exchange-proposal RNG.
    """

    def __init__(
        self,
        pool: ReplicaPool,
        block_size: int,
        random_seed: int,
    ) -> None:
        if len(pool) < 2:
            raise ValueError("parallel tempering requires at least 2 replicas")
        self._pool = pool
        self._block_size = int(block_size)
        self._rng = np.random.default_rng(int(random_seed))
        self._callbacks: list[ExchangeCallback] = []
        self._replica_labels = np.arange(len(pool), dtype=np.int64)
        self._history: ExchangeHistory | None = None

    # --- public API ----

    def __enter__(self) -> BaseParallelTempering:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Shutdown the underlying pool on context exit.

        Context-manager use is the recommended pattern — `with
        CanonicalParallelTempering.process_pool(...) as pt: pt.run(...)`
        ensures worker processes are joined and any tempdirs owned by
        the factory are cleaned up on exit, including the exceptional
        path.
        """
        self._pool.shutdown()

    @property
    def pool(self) -> ReplicaPool:
        """The underlying replica pool."""
        return self._pool

    @property
    def history(self) -> ExchangeHistory | None:
        return self._history

    @property
    def replica_labels(self) -> np.ndarray:
        return self._replica_labels.copy()

    def attach_callback(self, callback: ExchangeCallback) -> None:
        self._callbacks.append(callback)

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to one or more replicas.

        Requires the pool to satisfy `ObservablePool`. Any pool that
        implements only `ReplicaPool` does not satisfy it; calling this
        method with such a pool raises `TypeError`.

        For ``attach_observer_class`` and ``attach_observer_factory``,
        reach the pool directly via ``self.pool``.

        Args:
            observer: an mchammer `BaseObserver` instance.
            replicas: either the string ``"all"``, or an explicit sequence
                of replica indices to attach to.
        """
        if not isinstance(self._pool, ObservablePool):
            raise TypeError(
                f"attach_observer requires an ObservablePool; "
                f"{type(self._pool).__name__} does not satisfy it."
            )
        self._pool.attach_observer(observer, replicas)

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Advance all replicas for ``n_cycles`` MC+exchange cycles.

        Overwrites any history from a previous call. If an exception is
        raised at any point during the call, `self.history` reflects the
        partial history (rows past the failure point are zeros).
        """
        n_replicas = len(self._pool)
        history = ExchangeHistory.empty(n_cycles=n_cycles, n_replicas=n_replicas)
        try:
            history.energies_per_cycle[0] = self._pool.current_energies()
            history.replica_labels_per_cycle[0] = self._replica_labels
            for c in range(n_cycles):
                self._pool.advance_all(self._block_size)
                for pair in pair_set_for_cycle(n_replicas, c):
                    self._try_exchange(int(pair), int(pair) + 1, c, history)
                history.energies_per_cycle[c + 1] = self._pool.current_energies()
                history.replica_labels_per_cycle[c + 1] = self._replica_labels
        finally:
            self._history = history
        return history

    # --- abstract hook ----

    @abstractmethod
    def _log_prob_ratio(self, i: int, j: int) -> float:
        """Log of the exchange acceptance ratio for adjacent replicas i, j."""
        ...

    # --- internals ----

    def _try_exchange(
        self,
        i: int,
        j: int,
        cycle: int,
        history: ExchangeHistory,
    ) -> None:
        log_r = self._log_prob_ratio(i, j)
        if not np.isfinite(log_r):
            E_i = self._pool.current_energy(i)
            E_j = self._pool.current_energy(j)
            raise RuntimeError(
                f"Non-finite log-probability ratio on cycle {cycle}, "
                f"pair ({i}, {j}): log_r = {log_r}, "
                f"E_i = {E_i}, E_j = {E_j}. "
                f"Check for NaN/inf replica energies (diverged MC, "
                f"bad cluster expansion, etc.)."
            )
        accepted = metropolis_accept(log_r, self._rng)
        pair_index = min(i, j)
        history.swap_attempted[pair_index] += 1
        for callback in self._callbacks:
            callback.on_exchange(
                cycle=cycle,
                pair_index=pair_index,
                accepted=accepted,
                log_prob_ratio=log_r,
            )
        if accepted:
            self._pool.swap_configurations(i, j)
            self._replica_labels[[i, j]] = self._replica_labels[[j, i]]
            history.swap_accepted[pair_index] += 1
