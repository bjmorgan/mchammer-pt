"""Abstract parallel-tempering orchestrator.

`BaseParallelTempering` drives the cycle loop, records per-cycle
observations, and coordinates exchange proposals. All replica state
lives in the pool (`ReplicaPool`, `CanonicalPool`, `WangLandauPool`,
or their observable variants); the orchestrator routes queries
through it and never holds replica state directly.

Ensemble-specific subclasses override exactly one method:
`_log_prob_ratio(i, j)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import Atoms
from mchammer.observers.base_observer import (
    BaseObserver,
)

from .callbacks import CycleCallback, ExchangeCallback
from .checkpoint import CheckpointWriter
from .exchange import metropolis_accept, pair_set_for_cycle
from .history import ExchangeHistory
from .parallel.backend import ReplicaPool, _ObserverAttach

if TYPE_CHECKING:
    from .history import MetaValue


class BaseParallelTempering(ABC):
    """Abstract PT orchestrator.

    Args:
        pool: a `ReplicaPool` owning one replica per ladder rung. If
            the pool satisfies `_ObserverAttach`
            (i.e. it is an `ObservablePool` or
            `WangLandauObservablePool`), `attach_observer` will
            forward to it; otherwise calling `attach_observer` raises
            `TypeError`.
        block_size: MC trial steps per replica per cycle.
        random_seed: master seed for the exchange-proposal RNG.
        template_atoms: reference structure whose cell, positions, and
            pbc are used by `final_configurations` to reconstruct
            full ``Atoms`` objects from occupation vectors.
    """

    def __init__(
        self,
        pool: ReplicaPool,
        block_size: int,
        random_seed: int,
        *,
        template_atoms: Atoms,
    ) -> None:
        if len(pool) < 2:
            raise ValueError("parallel tempering requires at least 2 replicas")
        self._pool = pool
        self._block_size = int(block_size)
        self._rng = np.random.default_rng(int(random_seed))
        self._callbacks: list[ExchangeCallback] = []
        self._cycle_callbacks: list[CycleCallback] = []
        self._replica_labels = np.arange(len(pool), dtype=np.int64)
        self._history: ExchangeHistory | None = None
        self._template_atoms: Atoms = template_atoms.copy()  # type: ignore[no-untyped-call]

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

    def final_configurations(self) -> list[Atoms]:
        """The current Atoms at each temperature position.

        Returns one ``Atoms`` per temperature, in ladder order. Each
        ``Atoms`` has the same cell, positions, and pbc as the starting
        structure; only the site occupations (atomic numbers) reflect
        the current state.
        """
        configs: list[Atoms] = []
        for i in range(len(self._pool)):
            a: Atoms = self._template_atoms.copy()  # type: ignore[no-untyped-call]
            a.numbers = self._pool.current_occupations(i)
            configs.append(a)
        return configs

    def attach_callback(self, callback: ExchangeCallback) -> None:
        """Register an exchange-event callback.

        Multiple exchange callbacks compose; they are invoked in
        registration order on each proposed exchange.
        """
        self._callbacks.append(callback)

    def attach_cycle_callback(self, callback: CycleCallback) -> None:
        """Register a per-cycle callback.

        Multiple cycle callbacks compose; they are invoked in
        registration order at the end of each cycle, after history
        rows for that cycle have been written.
        """
        self._cycle_callbacks.append(callback)

    def attach_checkpoint_writer(
        self,
        path: Path | str,
        *,
        interval: int = 1000,
    ) -> None:
        """Attach a periodic checkpoint writer to this orchestrator.

        Convenience wrapper around `CheckpointWriter` that binds the
        orchestrator at attach time. Equivalent to::

            from mchammer_pt import CheckpointWriter
            pt.attach_cycle_callback(CheckpointWriter(path, interval=interval, pt=pt))

        See `CheckpointWriter` for output shape and atomicity
        semantics.
        """
        self.attach_cycle_callback(
            CheckpointWriter(path, interval=interval, pt=self)
        )

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to one or more replicas.

        Requires the pool to satisfy the `_ObserverAttach` mixin
        Protocol, which `ObservablePool` (canonical) and
        `WangLandauObservablePool` (REWL) both extend. Pools that
        don't expose observer attach raise `TypeError`.

        For ``attach_observer_class`` and ``attach_observer_factory``,
        reach the pool directly via ``self.pool``.

        Args:
            observer: an mchammer `BaseObserver` instance.
            replicas: either the string ``"all"``, or an explicit sequence
                of replica indices to attach to.
        """
        if not isinstance(self._pool, _ObserverAttach):
            raise TypeError(
                f"attach_observer requires a pool that supports observer "
                f"attach; {type(self._pool).__name__} does not."
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
        # Publish the in-progress history on `self._history` before the loop
        # so cycle callbacks (e.g. `CheckpointWriter`) can read live
        # orchestrator state via `self`, and so an exception inside the loop
        # leaves `self.history` pointing at the partially filled history.
        self._history = history
        history.energies_per_cycle[0] = self._pool.current_energies()
        history.replica_labels_per_cycle[0] = self._replica_labels
        for c in range(n_cycles):
            self._pool.advance_all(self._block_size)
            for pair in pair_set_for_cycle(n_replicas, c):
                self._try_exchange(int(pair), int(pair) + 1, c, history)
            history.energies_per_cycle[c + 1] = self._pool.current_energies()
            history.replica_labels_per_cycle[c + 1] = self._replica_labels
            for cb in self._cycle_callbacks:
                cb.on_cycle_end(c, n_cycles, history)
        return history

    # --- abstract hook ----

    @abstractmethod
    def _log_prob_ratio(self, i: int, j: int) -> float:
        """Log of the exchange acceptance ratio for adjacent replicas i, j."""
        ...

    def _checkpoint_meta(self) -> dict[str, MetaValue]:
        """Subclass-specific keys for the checkpoint meta dict.

        Default: empty dict (subclass contributes nothing beyond the
        shared keys). Canonical PT returns ``{"temperatures": ...}``;
        Wang-Landau PT returns ``{"windows": ..., "energy_spacing": ...}``.
        Shared keys (block_size, random_seed, identity hashes,
        schema_version) live in `checkpoint._write_checkpoint`.
        """
        return {}

    # --- internals ----

    def _try_exchange(
        self,
        i: int,
        j: int,
        cycle: int,
        history: ExchangeHistory,
    ) -> None:
        log_r = self._log_prob_ratio(i, j)
        if np.isnan(log_r) or log_r == np.inf:
            E_i = self._pool.current_energy(i)
            E_j = self._pool.current_energy(j)
            raise RuntimeError(
                f"Non-finite log-probability ratio on cycle {cycle}, "
                f"pair ({i}, {j}): log_r = {log_r}, "
                f"E_i = {E_i}, E_j = {E_j}. "
                f"Check for NaN/+inf replica energies (diverged MC, "
                f"bad cluster expansion, or a REWL replica that has "
                f"drifted outside its own window). Negative-infinity ratios "
                f"are legal (e.g. out-of-window partner energy in REWL)."
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
