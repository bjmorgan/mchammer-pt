"""Per-temperature ensemble handle.

`Replica` wraps a single `mchammer.CanonicalEnsemble` and is the
only place in `mchammer-pt` that directly calls `mchammer`'s MC
machinery. The orchestrator and the parallel backends interact with
ensembles exclusively through `Replica`.

This confinement keeps the rest of the library free of `mchammer`
internals, so changes to `mchammer`'s API affect only this file.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.data_containers.base_data_container import BaseDataContainer
from mchammer.ensembles import CanonicalEnsemble
from mchammer.observers.base_observer import BaseObserver


class Replica:
    """One `CanonicalEnsemble` at one temperature, wrapped for PT use.

    The orchestrator holds a list of these. Each Replica knows its
    temperature, its current configuration, and how to advance itself;
    the orchestrator composes them into a parallel-tempering run.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: starting structure (copied, not mutated).
        temperature: simulation temperature in kelvin.
        random_seed: seed for this replica's MC random generator.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        temperature: float,
        random_seed: int,
    ) -> None:
        self._temperature = float(temperature)
        # Copy atoms so the caller's object is not mutated by mchammer.
        atoms_copy = atoms.copy()
        calculator = ClusterExpansionCalculator(atoms_copy, cluster_expansion)
        self._ensemble = CanonicalEnsemble(
            structure=atoms_copy,
            calculator=calculator,
            temperature=self._temperature,
            random_seed=int(random_seed),
        )

    @property
    def temperature(self) -> float:
        """Replica temperature in kelvin."""
        return self._temperature

    def advance(self, n_steps: int) -> None:
        """Run `n_steps` canonical MC trial steps."""
        self._ensemble.run(int(n_steps))

    def current_energy(self) -> float:
        """Total CE energy (eV) of the current configuration."""
        return float(
            self._ensemble.calculator.calculate_total(
                occupations=self._ensemble.configuration.occupations
            )
        )

    def current_occupations(self) -> np.ndarray:
        """Copy of the current occupation vector (atomic numbers)."""
        return self._ensemble.configuration.occupations.copy()

    def set_occupations(self, occupations: np.ndarray) -> None:
        """Overwrite the replica's configuration.

        Calls `CanonicalEnsemble.update_occupations`, which keeps the
        configuration manager and the calculator's cached state
        consistent. After this returns, `current_energy` and
        `current_occupations` reflect the new state.
        """
        occ = np.asarray(occupations, dtype=int)
        self._ensemble.update_occupations(
            sites=list(range(len(occ))), species=list(occ)
        )

    def attach_mchammer_observer(self, observer: BaseObserver) -> None:
        """Attach an mchammer observer to this replica's ensemble.

        The observer fires inside `advance(...)` at its configured
        interval, exactly as it would in a standalone single-ensemble
        run.
        """
        self._ensemble.attach_observer(observer)

    def data_container(self) -> BaseDataContainer:
        """The replica's `mchammer.BaseDataContainer` (live view).

        Trajectories written by attached observers land here. This is
        the native `mchammer` type, so downstream analysis tools work
        unchanged.
        """
        return self._ensemble.data_container
