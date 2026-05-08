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

    @classmethod
    def restart_from(
        cls,
        container: BaseDataContainer,
        *,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        temperature: float,
        random_seed: int,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
        cluster_expansion_path: str | os.PathLike[str] | None = None,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> Replica:
        """Construct a Replica whose ensemble has been restored from `container`.

        Drives mchammer's restoration path: builds the replica
        normally, swaps its empty `BaseDataContainer` for the
        supplied one, then explicitly calls
        `replica.ensemble._restart_ensemble()` to fire the
        upstream `_step` / `occupations` / `_accepted_trials` /
        stdlib-`random` restoration. The replica's private RNG
        snapshot is updated to match.

        `temperature`, `random_seed`, `ensemble_cls`, and
        `ensemble_kwargs` are passed through to the standard
        `Replica.__init__`. The seed is overwritten by
        `_restart_ensemble`'s `random.setstate` call, so its
        value is not load-bearing for bit-identical resume —
        supplied for completeness only.

        Bit-identical resume additionally requires restoring
        `ConfigurationManager._sites_by_species` — a
        path-dependent cache of per-sublattice ordered site lists
        keyed by species. mchammer's `_restart_ensemble` rebuilds
        this cache from current occupations in natural sublattice
        order, but the canonical-ensemble trial-step proposal
        draws `random.choice` from a list built in the cache's
        live order, so a freshly-rebuilt cache produces a
        different proposal stream than the original run. Passing
        the saved cache via ``sites_by_species`` overwrites the
        rebuilt cache to match the original order.

        Args:
            container: an mchammer `BaseDataContainer` whose
                ``_last_state`` carries the saved per-replica state.
            cluster_expansion: same CE used at the original run.
            atoms: structure with the right cell/positions/pbc.
                Occupations are immediately overwritten by
                `_restart_ensemble` from `container._last_state`,
                so the atoms argument is load-bearing only for the
                geometry template.
            temperature: simulation temperature (must match the
                replica's slot on the original ladder).
            random_seed: forwarded to `Replica.__init__`; overwritten
                by the saved `random_state`.
            ensemble_cls: same class used at the original run.
            ensemble_kwargs: same kwargs used at the original run.
            cluster_expansion_path: optional path the CE was loaded
                from; same semantics as `Replica.__init__`.
            sites_by_species: saved
                ``ConfigurationManager._sites_by_species`` cache
                (one dict per sublattice, mapping atomic number to
                ordered list of site indices). When supplied,
                overwrites the rebuilt cache so the next
                `random.choice`-driven proposal lands on the same
                site as the original run.
        """
        replica = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms,
            temperature=temperature,
            random_seed=random_seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            cluster_expansion_path=cluster_expansion_path,
        )
        replica.restore_state(container, sites_by_species=sites_by_species)
        return replica

    def restore_state(
        self,
        container: BaseDataContainer,
        *,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> None:
        """Mutate this replica to match a saved checkpoint.

        Swaps the ensemble's `BaseDataContainer` for ``container``,
        calls mchammer's ``_restart_ensemble`` to restore step count,
        configuration, accepted-trial count, and stdlib-``random``
        state from ``container._last_state``, and (if supplied)
        overwrites the path-dependent
        ``ConfigurationManager._sites_by_species`` cache with the
        saved order so the next trial-step proposal matches the
        original run.

        Used by `Replica.restart_from` (which constructs a fresh
        replica before calling this) and by `ProcessPool` workers
        (which call this on their existing in-process replica when
        the parent broadcasts a `RESTORE_STATE` opcode).

        Args:
            container: an mchammer `BaseDataContainer` whose
                ``_last_state`` carries the saved per-replica state.
            sites_by_species: saved
                ``ConfigurationManager._sites_by_species`` cache.
                When supplied, overwrites the rebuilt cache so the
                next ``random.choice``-driven proposal lands on the
                same site as the original run.
        """
        # Swap in the saved container so `_restart_ensemble` reads
        # from it. mchammer reads `self.data_container._last_state`
        # to drive every field of the restoration.
        self._ensemble._data_container = container
        # Save the caller's stdlib-`random` state and restore the
        # replica's private snapshot first — the same caller-isolation
        # discipline `Replica.advance` follows. `_restart_ensemble`
        # ends with a `random.setstate(saved_state)`, which we then
        # capture into `_rng_state`.
        caller_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble._restart_ensemble()
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)
        if sites_by_species is not None:
            # mchammer-internal: restoring the path-dependent
            # `_sites_by_species` cache so the next trial-step
            # proposal matches the original run. Same direct-access
            # pattern as the `_data_container` swap above.
            self._ensemble.configuration._sites_by_species = sites_by_species

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

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        """Refresh restart-state on the live container and return extras.

        Populates ``data_container._last_state`` with the four fields
        `_restart_ensemble` reads on resume — ``last_step``,
        ``occupations``, ``accepted_trials``, and ``random_state`` —
        and returns the additional per-replica state the orchestrator
        needs to checkpoint alongside the container. mchammer's own
        ``BaseEnsemble.write_data_container`` performs the same
        refresh inline; serialising the container directly (e.g. via
        the checkpoint writer) requires us to replicate it.

        ``random_state`` is taken from the replica's saved snapshot
        (`Replica.advance` pins this), not stdlib
        ``random.getstate()`` — the latter is the *caller*'s state at
        snapshot time, not the replica's.

        Returns:
            Dict with key ``"sites_by_species"`` carrying a deep copy
            of `ConfigurationManager._sites_by_species` (the
            path-dependent per-sublattice species → site-list cache
            consulted by canonical trial-step proposals). The
            orchestrator-side checkpoint code embeds this alongside
            the container.
        """
        self._ensemble._data_container._update_last_state(
            last_step=self._ensemble.step,
            occupations=self._ensemble.configuration.occupations.tolist(),
            accepted_trials=self._ensemble._accepted_trials,
            random_state=self._rng_state,
        )
        # Deep-copy to insulate the snapshot from later
        # `update_occupations` calls that mutate the live cache.
        # Cast numpy ints to Python ints so the JSON serialiser
        # downstream can handle them.
        sites_by_species: list[dict[int, list[int]]] = [
            {
                int(species): [int(s) for s in sites]
                for species, sites in sublattice.items()
            }
            for sublattice in self._ensemble.configuration._sites_by_species
        ]
        return {"sites_by_species": sites_by_species}
