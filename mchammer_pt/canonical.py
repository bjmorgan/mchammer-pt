"""Canonical-ensemble parallel tempering."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from icet import ClusterExpansion

from .base import BaseParallelTempering
from .parallel.backend import Backend
from .replica import Replica

# Boltzmann constant in eV / K. Energies returned by `Replica.current_energy`
# are in eV (total energy for the supercell), so beta has units 1/eV.
_KB = 8.617333262145e-5


class CanonicalParallelTempering(BaseParallelTempering):
    """Parallel tempering over a temperature ladder of canonical MC replicas.

    Each temperature gets its own persistent `mchammer.CanonicalEnsemble`;
    the orchestrator proposes configuration exchanges between adjacent
    temperatures on a regular cadence.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: starting structure; every replica begins from a copy.
        temperatures: list of temperatures in kelvin (ascending order
            by convention but not required).
        block_size: MC trial steps per replica per cycle.
        random_seed: master seed; each replica's MC RNG and the
            orchestrator's exchange-proposal RNG are deterministically
            spawned from it.
        backend: parallel backend; default is `SerialBackend`.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        temperatures: Sequence[float],
        block_size: int,
        random_seed: int,
        backend: Backend | None = None,
    ) -> None:
        temperatures = list(temperatures)
        if len(temperatures) < 2:
            raise ValueError("parallel tempering requires at least 2 temperatures")
        seed_sequence = np.random.SeedSequence(int(random_seed))
        # One child seed per replica plus one for the master exchange RNG.
        child_seeds = seed_sequence.spawn(len(temperatures) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]
        master_seed = int(child_seeds[-1].generate_state(1)[0])

        replicas = [
            Replica(
                cluster_expansion=cluster_expansion,
                atoms=atoms,
                temperature=float(T),
                random_seed=seed,
            )
            for T, seed in zip(temperatures, replica_seeds, strict=True)
        ]
        super().__init__(
            replicas=replicas,
            block_size=block_size,
            random_seed=master_seed,
            backend=backend,
        )
        self._temperatures = np.asarray(temperatures, dtype=np.float64)
        self._beta = 1.0 / (_KB * self._temperatures)

    @property
    def temperatures(self) -> np.ndarray:
        """Copy of the per-replica temperature array (kelvin)."""
        return self._temperatures.copy()

    def _log_prob_ratio(self, i: int, j: int) -> float:
        E_i = self._replicas[i].current_energy()
        E_j = self._replicas[j].current_energy()
        return float((self._beta[i] - self._beta[j]) * (E_i - E_j))
