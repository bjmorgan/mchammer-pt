"""Abstract parallel-tempering orchestrator.

`BaseParallelTempering` holds the replicas, the parallel backend, the
callbacks, and the history, and drives the cycle loop. Ensemble-specific
subclasses override exactly one method: `_log_prob_ratio(i, j)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from .callbacks import ExchangeCallback
from .exchange import metropolis_accept, pair_set_for_cycle
from .history import ExchangeHistory
from .parallel.backend import Backend
from .parallel.serial import SerialBackend
from .replica import Replica


class BaseParallelTempering(ABC):
    """Abstract PT orchestrator.

    Args:
        replicas: one replica per temperature, in ascending-T order.
        block_size: MC trial steps per replica per cycle.
        random_seed: master seed; the exchange-proposal RNG is seeded
            deterministically from this.
        backend: parallel backend for the replica-advance phase.
            Defaults to serial execution.
    """

    def __init__(
        self,
        replicas: list[Replica],
        block_size: int,
        random_seed: int,
        backend: Backend | None = None,
    ) -> None:
        if len(replicas) < 2:
            raise ValueError("parallel tempering requires at least 2 replicas")
        self._replicas = list(replicas)
        self._block_size = int(block_size)
        self._rng = np.random.default_rng(int(random_seed))
        self._backend: Backend = backend if backend is not None else SerialBackend()
        self._callbacks: list[ExchangeCallback] = []
        self._replica_labels = np.arange(len(replicas), dtype=np.int64)
        self._history: ExchangeHistory | None = None

    # --- public API ----

    @property
    def replicas(self) -> list[Replica]:
        return list(self._replicas)

    @property
    def history(self) -> ExchangeHistory | None:
        return self._history

    @property
    def replica_labels(self) -> np.ndarray:
        return self._replica_labels.copy()

    def attach_callback(self, callback: ExchangeCallback) -> None:
        self._callbacks.append(callback)

    def attach_observer(
        self, observer: BaseObserver, replicas: list[int] | str = "all"
    ) -> None:
        """Attach an mchammer observer to one or more replicas.

        Args:
            observer: an mchammer BaseObserver instance.
            replicas: either the string ``"all"``, or an explicit list
                of replica indices to attach to.
        """
        indices = (
            range(len(self._replicas))
            if replicas == "all"
            else [int(i) for i in replicas]
        )
        for i in indices:
            self._replicas[i].attach_mchammer_observer(observer)

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Advance all replicas for ``n_cycles`` MC+exchange cycles."""
        n_replicas = len(self._replicas)
        history = ExchangeHistory.empty(n_cycles=n_cycles, n_replicas=n_replicas)
        history.energies_per_cycle[0] = self._current_energies()
        history.replica_labels_per_cycle[0] = self._replica_labels

        for c in range(n_cycles):
            self._backend.advance_all(self._replicas, self._block_size)
            for pair in pair_set_for_cycle(n_replicas, c):
                self._try_exchange(int(pair), int(pair) + 1, c, history)
            history.energies_per_cycle[c + 1] = self._current_energies()
            history.replica_labels_per_cycle[c + 1] = self._replica_labels

        self._history = history
        return history

    # --- abstract hook ----

    @abstractmethod
    def _log_prob_ratio(self, i: int, j: int) -> float:
        """Log of the exchange acceptance ratio for adjacent replicas i, j."""
        ...

    # --- internals ----

    def _current_energies(self) -> np.ndarray:
        return np.array([r.current_energy() for r in self._replicas])

    def _try_exchange(
        self,
        i: int,
        j: int,
        cycle: int,
        history: ExchangeHistory,
    ) -> None:
        log_r = self._log_prob_ratio(i, j)
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
            self._exchange_configurations(i, j)
            self._replica_labels[[i, j]] = self._replica_labels[[j, i]]
            history.swap_accepted[pair_index] += 1

    def _exchange_configurations(self, i: int, j: int) -> None:
        occ_i = self._replicas[i].current_occupations()
        occ_j = self._replicas[j].current_occupations()
        self._replicas[i].set_occupations(occ_j)
        self._replicas[j].set_occupations(occ_i)
