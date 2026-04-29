"""Per-temperature ensemble handle.

`Replica` wraps a single `mchammer.CanonicalEnsemble` and is the
only place in `mchammer-pt` that directly calls `mchammer`'s MC
machinery. The orchestrator and the parallel backends interact with
ensembles exclusively through `Replica`.

Each Replica owns its own logical RNG stream. `mchammer` drives its
Monte Carlo from Python's global `random` module, which means two
replicas built in the same process would otherwise clobber each
other's seeds. `advance` therefore save/restores the global state
around every call so each Replica evolves as if it had the process
to itself.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.calculators import (  # type: ignore[import-untyped]
    ClusterExpansionCalculator,
)
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

_RESERVED_ENSEMBLE_KWARGS: frozenset[str] = frozenset(
    {"structure", "calculator", "temperature", "random_seed"}
)


class Replica:
    """One canonical ensemble at one temperature, wrapped for PT use.

    The orchestrator holds a list of these. Each Replica knows its
    temperature, its current configuration, and how to advance itself;
    the orchestrator composes them into a parallel-tempering run.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: starting structure (copied, not mutated).
        temperature: simulation temperature in kelvin.
        random_seed: seed for this replica's MC random generator.
        ensemble_cls: `CanonicalEnsemble` or a subclass thereof. Defaults
            to `CanonicalEnsemble`. Pinned to canonical because the
            orchestrator's exchange acceptance is canonical-only;
            non-canonical subclasses would silently produce wrong
            physics.
        ensemble_kwargs: extra keyword arguments forwarded to
            ``ensemble_cls(...)`` on top of the four standard ones
            (``structure``, ``calculator``, ``temperature``,
            ``random_seed``). Reserved names cannot appear here; see
            `__init__`.
        cluster_expansion_path: path the cluster expansion was loaded
            from, if known. Accepts ``str`` or any
            ``os.PathLike[str]``; coerced to ``str`` for storage.
            Auto-populated on workers spawned by ``ProcessPool``;
            optional elsewhere.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        temperature: float,
        random_seed: int,
        *,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
        cluster_expansion_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._temperature = float(temperature)
        self._cluster_expansion_path = (
            None
            if cluster_expansion_path is None
            else os.fspath(cluster_expansion_path)
        )
        extra = dict(ensemble_kwargs) if ensemble_kwargs else {}
        clash = _RESERVED_ENSEMBLE_KWARGS & extra.keys()
        if clash:
            raise ValueError(
                f"ensemble_kwargs must not contain {sorted(clash)}; "
                f"these are set by Replica from its own arguments "
                f"(structure/calculator from cluster_expansion+atoms; "
                f"temperature and random_seed from their dedicated parameters)."
            )
        # Copy atoms so the caller's object is not mutated by mchammer.
        # `ase.Atoms.copy` is untyped upstream, so annotate the target here.
        atoms_copy: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
        calculator = ClusterExpansionCalculator(atoms_copy, cluster_expansion)
        # `CanonicalEnsemble.__init__` calls `random.seed(random_seed)` on
        # Python's global RNG. Save the caller's state first, snapshot the
        # seeded state for this replica, then restore the caller's state —
        # so constructing a Replica has no observable side effect on
        # external `random.*` consumers, and every replica still owns an
        # independent stream.
        caller_state = random.getstate()
        try:
            self._ensemble = ensemble_cls(
                structure=atoms_copy,
                calculator=calculator,
                temperature=self._temperature,
                random_seed=int(random_seed),
                **extra,
            )
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)

    @property
    def temperature(self) -> float:
        """Replica temperature in kelvin."""
        return self._temperature

    @property
    def ensemble(self) -> CanonicalEnsemble:
        """The underlying mchammer ensemble."""
        return self._ensemble

    @property
    def cluster_expansion_path(self) -> str | None:
        """Path the cluster expansion was loaded from, if known.

        Auto-populated on workers spawned by ``ProcessPool`` (each
        worker reads its CE from a path supplied at pool
        construction). Optional on ``SerialPool`` — pass
        ``cluster_expansion_path=`` to ``Replica`` if you want
        factory-path observers to reload the CE fresh.

        Returns ``None`` if no path was supplied. Factories whose
        constructors take a ``ClusterSpace`` or ``ClusterExpansion``
        should reload via
        ``ClusterExpansion.read(replica.cluster_expansion_path)``;
        reading from disk yields a fresh ``ClusterSpace`` independent
        of the calculator's mutated copy.
        """
        return self._cluster_expansion_path

    def advance(self, n_steps: int) -> None:
        """Run `n_steps` canonical MC trial steps.

        Restores this replica's private RNG snapshot before calling
        `mchammer`, then captures the updated state so repeated
        advances form a single continuous stream.
        """
        previous_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble.run(int(n_steps))
            self._rng_state = random.getstate()
        finally:
            random.setstate(previous_state)

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
