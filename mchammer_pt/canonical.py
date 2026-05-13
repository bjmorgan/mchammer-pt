"""Canonical-ensemble parallel tempering."""

from __future__ import annotations

import tempfile
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from .base import BaseParallelTempering
from .checkpoint import (
    _compute_ce_identity,
    _compute_ensemble_kwargs_hash,
    _validate_kwargs_hash,
    _write_checkpoint,
)
from .history import ExchangeHistory, MetaValue
from .parallel.backend import CanonicalPool
from .parallel.processes import ProcessPool
from .parallel.serial import SerialPool
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
        atoms: starting structure, either a single ``Atoms`` (broadcast
            to every replica) or a sequence of ``Atoms`` (one per
            temperature, length-validated). In canonical MC only site
            occupations vary; all entries should share the same cell,
            positions, and pbc.
        temperatures: non-decreasing temperatures in kelvin. At least
            two values required. Equal adjacent temperatures are
            allowed (produces a same-T null case where exchange is a
            pure relabelling); strictly decreasing values are rejected.
        block_size: MC trial steps per replica per cycle. Must be >= 1.
        random_seed: master seed; each replica's MC RNG and the
            orchestrator's exchange-proposal RNG are deterministically
            spawned from it.
        pool: optional `CanonicalPool` to use as the execution backend.
            If None (the default), a `SerialPool` is constructed from
            ``cluster_expansion``, ``atoms``, ``temperatures``, and the
            spawned per-replica seeds.
        data_container_file: optional path; if given, `run` writes a
            full checkpoint to this path on completion (the
            `ExchangeHistory`, each replica's
            `mchammer.BaseDataContainer`, run metadata, identity hashes,
            and orchestrator-level state). Files written this way are
            valid resume sources for
            `CanonicalParallelTempering.resume`.
        ensemble_cls: `CanonicalEnsemble` or a subclass thereof, used by
            every replica when this orchestrator constructs the default
            pool. Rejected when ``pool`` is supplied directly. Pinned to
            canonical because the exchange acceptance is canonical-only.
        ensemble_kwargs: extra keyword arguments forwarded to
            ``ensemble_cls(...)`` for every replica. Cannot include
            ``structure``, ``calculator``, ``temperature``, or
            ``random_seed`` (these are set by `Replica`). Rejected when
            ``pool`` is supplied directly.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms | Sequence[Atoms],
        temperatures: Sequence[float],
        block_size: int,
        random_seed: int,
        pool: CanonicalPool | None = None,
        data_container_file: Path | str | None = None,
        *,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        temperatures = [float(T) for T in temperatures]
        if len(temperatures) < 2:
            raise ValueError("parallel tempering requires at least 2 temperatures")
        pairs = zip(temperatures[:-1], temperatures[1:], strict=True)
        if any(b < a for a, b in pairs):
            raise ValueError(f"temperatures must be non-decreasing; got {temperatures}")
        if int(block_size) < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        if isinstance(atoms, Atoms):
            atoms_list: list[Atoms] = [atoms] * len(temperatures)
        else:
            atoms_list = list(atoms)
            if len(atoms_list) != len(temperatures):
                raise ValueError(
                    f"atoms has {len(atoms_list)} entries but temperatures "
                    f"has {len(temperatures)}; supply one Atoms per "
                    f"temperature or a single Atoms to broadcast"
                )
            ref = atoms_list[0]
            for i, a in enumerate(atoms_list[1:], 1):
                if not (
                    np.array_equal(a.cell.array, ref.cell.array)
                    and np.array_equal(a.positions, ref.positions)
                    and np.array_equal(a.pbc, ref.pbc)
                ):
                    raise ValueError(
                        f"atoms[{i}] has different cell/positions/pbc "
                        f"than atoms[0]; canonical MC requires identical "
                        f"lattice geometry across replicas"
                    )
        seed_sequence = np.random.SeedSequence(int(random_seed))
        # One child seed per replica plus one for the master exchange RNG.
        child_seeds = seed_sequence.spawn(len(temperatures) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]
        master_seed = int(child_seeds[-1].generate_state(1)[0])

        # Pool/ensemble exclusion runs first: combining ``pool=`` with
        # custom ensemble args reflects a more fundamental misuse of the
        # API than a length/temperature mismatch, and the latter is
        # often a downstream consequence of the former (the user built
        # a pool with the wrong ladder *because* they thought the
        # orchestrator would re-derive the ladder from ensemble_cls).
        if pool is not None and (
            ensemble_cls is not CanonicalEnsemble or ensemble_kwargs
        ):
            raise ValueError(
                "ensemble_cls / ensemble_kwargs cannot be combined with an "
                "explicit pool=; the pool already owns its replicas. Pass "
                "these kwargs only when letting CanonicalParallelTempering "
                "build the default SerialPool, or use process_pool(...) "
                "which forwards them."
            )
        if pool is None:
            replicas = [
                Replica(
                    cluster_expansion=cluster_expansion,
                    atoms=a,
                    temperature=T,
                    random_seed=seed,
                    ensemble_cls=ensemble_cls,
                    ensemble_kwargs=ensemble_kwargs,
                )
                for a, T, seed in zip(
                    atoms_list, temperatures, replica_seeds, strict=True
                )
            ]
            pool = SerialPool(replicas)
        else:
            # When the caller supplies a pool directly, its replica
            # count and per-replica temperatures must match the
            # orchestrator's temperatures kwarg. If they disagree, the
            # orchestrator would compute Boltzmann factors from one
            # ladder while the pool's replicas run on another, silently
            # biasing every exchange acceptance. Catch it here rather
            # than letting it through.
            if len(pool) != len(temperatures):
                raise ValueError(
                    f"pool has {len(pool)} replicas but temperatures "
                    f"has {len(temperatures)} entries; construct the pool "
                    f"with the same ladder, or use "
                    f"CanonicalParallelTempering.process_pool(...) which "
                    f"owns pool construction."
                )
            pool_temps = [float(T) for T in pool.temperatures]
            if pool_temps != temperatures:
                raise ValueError(
                    f"pool.temperatures ({pool_temps}) does not match "
                    f"temperatures ({temperatures}); the orchestrator's "
                    f"beta values and the pool's per-replica temperatures "
                    f"must agree exactly, or exchange acceptance is "
                    f"silently biased. Use "
                    f"CanonicalParallelTempering.process_pool(...) to "
                    f"avoid constructing the ladder twice."
                )
        super().__init__(
            pool=pool,
            block_size=block_size,
            random_seed=master_seed,
            template_atoms=atoms_list[0],
        )
        self._temperatures = np.asarray(temperatures, dtype=np.float64)
        self._beta = 1.0 / (_KB * self._temperatures)
        self._data_container_file = data_container_file
        # Stored for the checkpoint payload — computed once at
        # construction so checkpoint writes don't repeat the
        # CE-write-and-hash cost on every emission.
        self._random_seed = int(random_seed)
        self._ce_identity = _compute_ce_identity(cluster_expansion)
        self._ensemble_cls_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        self._ensemble_kwargs_hash = _compute_ensemble_kwargs_hash(ensemble_kwargs)

    @property
    def temperatures(self) -> np.ndarray:
        """Copy of the per-replica temperature array (kelvin)."""
        return self._temperatures.copy()

    def save_checkpoint(self, path: Path | str) -> None:
        """Write a full checkpoint of this orchestrator atomically.

        Captures everything `CanonicalParallelTempering.resume` needs
        to reconstruct an equivalent orchestrator: per-replica state
        (each `mchammer.BaseDataContainer`'s ``_last_state`` plus the
        path-dependent ``ConfigurationManager._sites_by_species``
        cache that bit-identical resume requires), the orchestrator's
        replica-label permutation and exchange-proposal RNG state,
        and identity hashes for the CE, ensemble class, and ensemble
        kwargs.

        Requires `run()` to have been called at least once. Raises
        `RuntimeError` otherwise — the per-replica data containers
        do not populate `_last_state` until a run completes.
        """
        _write_checkpoint(self, path)

    def _log_prob_ratio(self, i: int, j: int) -> float:
        E_i = self._pool.current_energy(i)
        E_j = self._pool.current_energy(j)
        return float((self._beta[i] - self._beta[j]) * (E_i - E_j))

    def _checkpoint_meta(self) -> dict[str, MetaValue]:
        return {"temperatures": self._temperatures}

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Run `n_cycles` PT cycles, optionally writing an HDF5 bundle.

        When `data_container_file` was provided at construction, the
        full checkpoint payload (`ExchangeHistory`, each replica's
        `mchammer.BaseDataContainer`, run metadata, identity hashes,
        and orchestrator-level state) is written to that path on
        completion as a single atomic HDF5 file. Files written this
        way are valid resume sources for
        `CanonicalParallelTempering.resume`.
        """
        history = super().run(n_cycles=n_cycles)
        if self._data_container_file is not None:
            _write_checkpoint(self, Path(self._data_container_file))
        return history

    @classmethod
    def resume(
        cls,
        path: Path | str,
        *,
        cluster_expansion: ClusterExpansion,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> CanonicalParallelTempering:
        """Resume a previously-checkpointed canonical PT run.

        Reconstructs the orchestrator at the state the checkpoint
        captured: per-replica state (configuration, step count,
        accepted-trial count, stdlib `random` state, plus
        ``ConfigurationManager._sites_by_species`` cache) via
        mchammer's `_restart_ensemble` augmented by
        `Replica.restart_from`, plus orchestrator-level state
        (replica-label permutation, exchange-proposal RNG state).

        The next `run(M)` call returns an `ExchangeHistory`
        covering only those `M` cycles. Combine it with the prior
        history via
        `ExchangeHistory.concatenate(prior_history, run_b_history)`
        to get the bit-identical-to-single-run view.

        To start a new run from a previous run's equilibrated
        configurations under a *different* CE, use
        `pt.final_configurations()` plus a fresh orchestrator
        instead. `resume` enforces CE identity to make the two
        workflows distinguishable.

        Args:
            path: checkpoint file written by `CheckpointWriter` or
                `pt.save_checkpoint`.
            cluster_expansion: must match the CE used when the
                checkpoint was written. Hard error on mismatch.
            ensemble_cls: same class used at original construction.
                Hard error if its fully-qualified name does not
                match the checkpoint's ``ensemble_cls_fqn``.
            ensemble_kwargs: same kwargs used at original
                construction. Where representable, the kwargs hash
                is validated; non-representable kwargs (e.g. those
                containing icet `ClusterSpace`) skip the check.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            KeyError: file is missing required schema groups.
            ValueError: ``schema_version`` is unknown, CE identity
                mismatches, or ``ensemble_cls_fqn`` mismatches.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
        )
        from .history import read_hdf5

        history, containers, meta = read_hdf5(path)
        schema_version = meta.get("schema_version")
        if schema_version != "3":
            raise ValueError(
                f"{path}: unknown schema_version {schema_version!r}; "
                f"this version of mchammer-pt understands '3' only."
            )
        expected_ce_identity = _compute_ce_identity(cluster_expansion)
        if meta["ce_identity"] != expected_ce_identity:
            raise ValueError(
                f"{path}: CE identity mismatch. The checkpoint was "
                f"written against a different cluster_expansion than "
                f"the one supplied. Use `pt.final_configurations()` "
                f"plus a fresh orchestrator if you intend to continue "
                f"under a different CE."
            )
        expected_ensemble_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        if meta["ensemble_cls_fqn"] != expected_ensemble_fqn:
            raise ValueError(
                f"{path}: ensemble_cls FQN mismatch. Checkpoint has "
                f"{meta['ensemble_cls_fqn']!r}; resume was called "
                f"with {expected_ensemble_fqn!r}."
            )
        _validate_kwargs_hash(path, meta, ensemble_kwargs, "resume")

        orchestrator_state = _read_orchestrator_state(path)
        replica_extras = _read_replica_extra(path)
        temperatures = list(np.asarray(meta["temperatures"]))
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])

        # Reconstruct per-replica atoms from each container. The
        # atoms argument is a per-T list because each container's
        # structure already carries the saved occupations (which
        # `_restart_ensemble` will restore again from `_last_state`).
        atoms_list = [container.structure.copy() for container in containers]

        # Build replicas via `Replica.restart_from`. Per-replica seeds
        # are reproduced from the master seed using the same
        # `SeedSequence` spawn the constructor uses; values are
        # immediately overwritten by `_restart_ensemble`.
        seed_sequence = np.random.SeedSequence(random_seed)
        child_seeds = seed_sequence.spawn(len(temperatures) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]

        replicas = [
            Replica.restart_from(
                container,
                cluster_expansion=cluster_expansion,
                atoms=atoms,
                temperature=T,
                random_seed=seed,
                ensemble_cls=ensemble_cls,
                ensemble_kwargs=ensemble_kwargs,
                sites_by_species=extra["sites_by_species"],
            )
            for container, atoms, T, seed, extra in zip(
                containers,
                atoms_list,
                temperatures,
                replica_seeds,
                replica_extras,
                strict=True,
            )
        ]
        pool = SerialPool(replicas)
        pt = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms_list,
            temperatures=temperatures,
            block_size=block_size,
            random_seed=random_seed,
            pool=pool,
        )
        # Preserve fidelity for users who custom-supplied non-default
        # ensemble kwargs: re-stamp the identity hashes from the
        # loaded checkpoint (the constructor recomputes them from the
        # supplied kwargs, which already passed validation above).
        # ``meta`` values are typed as the ``MetaValue`` union; cast
        # to the concrete types we know each field carries.
        pt._ensemble_cls_fqn = str(meta["ensemble_cls_fqn"])
        pt._ensemble_kwargs_hash = str(meta["ensemble_kwargs_hash"])

        # Restore orchestrator-level state.
        pt._replica_labels = np.asarray(
            orchestrator_state["replica_labels"], dtype=np.int64
        )
        rng_state_raw = orchestrator_state["rng_state"]
        assert isinstance(rng_state_raw, str)
        pt._rng.bit_generator.state = json.loads(rng_state_raw)
        return pt

    @classmethod
    def resume_process_pool(
        cls,
        path: Path | str,
        *,
        cluster_expansion: ClusterExpansion,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> CanonicalParallelTempering:
        """Resume a checkpointed canonical PT run into a `ProcessPool`.

        Same identity validation and per-replica restoration as
        `resume`, but reconstructs the pool as a `ProcessPool`
        instead of a `SerialPool`. Worker scheduling
        non-determinism means the bit-identical contract does NOT
        hold across the serial-to-process or process-to-serial
        boundary; resume into the same pool kind that wrote the
        checkpoint for bit-identical continuation, or accept that
        cross-pool resume gives a statistically-valid continuation
        only.

        See `resume` for argument and error semantics.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
        )
        from .history import read_hdf5

        history, containers, meta = read_hdf5(path)
        schema_version = meta.get("schema_version")
        if schema_version != "3":
            raise ValueError(
                f"{path}: unknown schema_version {schema_version!r}; "
                f"this version of mchammer-pt understands '3' only."
            )
        expected_ce_identity = _compute_ce_identity(cluster_expansion)
        if meta["ce_identity"] != expected_ce_identity:
            raise ValueError(
                f"{path}: CE identity mismatch. The checkpoint was "
                f"written against a different cluster_expansion than "
                f"the one supplied. Use `pt.final_configurations()` "
                f"plus a fresh orchestrator if you intend to continue "
                f"under a different CE."
            )
        expected_ensemble_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        if meta["ensemble_cls_fqn"] != expected_ensemble_fqn:
            raise ValueError(
                f"{path}: ensemble_cls FQN mismatch. Checkpoint has "
                f"{meta['ensemble_cls_fqn']!r}; resume_process_pool "
                f"was called with {expected_ensemble_fqn!r}."
            )
        _validate_kwargs_hash(
            path, meta, ensemble_kwargs, "resume_process_pool"
        )

        orchestrator_state = _read_orchestrator_state(path)
        replica_extras = _read_replica_extra(path)
        temperatures = list(np.asarray(meta["temperatures"]))
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])

        # Reconstruct per-replica atoms from each container — mirrors
        # the SerialPool resume path.
        atoms_list = [container.structure.copy() for container in containers]

        # Use the existing `process_pool` factory to build a ProcessPool
        # at the saved ladder. The factory writes the CE to a managed
        # tempdir and spawns workers; each worker's data container is
        # initially empty until `restore_replica_state` overwrites it.
        pt = cls.process_pool(
            cluster_expansion=cluster_expansion,
            atoms=atoms_list,
            temperatures=temperatures,
            block_size=block_size,
            random_seed=random_seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
        )
        # Once `process_pool` has spawned workers, any failure on the
        # rest of the resume path leaks them — the orchestrator is
        # never returned, the caller has no handle to call
        # `pt.shutdown()`, and a notebook-cell retry would accumulate
        # workers. Shut the pool down on any failure between here
        # and the return statement, mirroring the cleanup discipline
        # `process_pool(...)` uses for its CE tempdir.
        try:
            # Restore per-replica state on each worker via the new
            # RESTORE_STATE opcode. After this returns, each worker's
            # replica matches the saved checkpoint.
            pt._pool.restore_replica_state(  # type: ignore[attr-defined]
                containers, replica_extras
            )

            # Re-stamp identity hashes from the loaded checkpoint so
            # users custom-supplying non-default kwargs see preserved
            # identity rather than the constructor's recompute.
            pt._ensemble_cls_fqn = str(meta["ensemble_cls_fqn"])
            pt._ensemble_kwargs_hash = str(meta["ensemble_kwargs_hash"])

            # Restore orchestrator-level state.
            pt._replica_labels = np.asarray(
                orchestrator_state["replica_labels"], dtype=np.int64
            )
            rng_state_raw = orchestrator_state["rng_state"]
            assert isinstance(rng_state_raw, str)
            pt._rng.bit_generator.state = json.loads(rng_state_raw)
        except BaseException:
            pt._pool.shutdown()
            raise
        return pt

    @classmethod
    def process_pool(
        cls,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms | Sequence[Atoms],
        temperatures: Sequence[float],
        block_size: int,
        random_seed: int,
        data_container_file: Path | str | None = None,
        *,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> CanonicalParallelTempering:
        """Construct a process-parallel PT run in one call.

        The factory owns:

        - per-replica seed spawning from ``random_seed`` (same scheme
          as the serial default path);
        - writing ``cluster_expansion`` to a managed temporary
          directory, so each worker process can read it at startup;
        - constructing a `ProcessPool` at the same temperature ladder
          as the orchestrator, closing the alignment trap that
          separate pool + orchestrator construction opens up.

        The tempdir is released when the returned orchestrator is
        garbage-collected; call sites that want deterministic cleanup
        should use the orchestrator as a context manager::

            with CanonicalParallelTempering.process_pool(
                cluster_expansion=ce,
                atoms=atoms,
                temperatures=[200, 400, 800, 1600],
                block_size=1000,
                random_seed=0,
            ) as pt:
                pt.run(n_cycles=200)

        On exit the pool's workers are joined; the CE tempdir is
        cleaned shortly after by Python's garbage collector.

        Args:
            cluster_expansion: icet ClusterExpansion defining the energy.
            atoms: starting structure, either a single ``Atoms`` (broadcast
                to every replica) or a sequence of ``Atoms`` (one per
                temperature, length-validated). In canonical MC only site
                occupations vary; all entries should share the same cell,
                positions, and pbc.
            temperatures: non-decreasing temperatures in kelvin. At least
                two values required. Equal adjacent temperatures are
                allowed (produces a same-T null case where exchange is a
                pure relabelling); strictly decreasing values are rejected.
            block_size: MC trial steps per replica per cycle. Must be >= 1.
            random_seed: master seed; each replica's MC RNG and the
                orchestrator's exchange-proposal RNG are deterministically
                spawned from it.
            data_container_file: optional path; if given, `run` writes a
                full checkpoint to this path on completion (the
                `ExchangeHistory`, each replica's
                `mchammer.BaseDataContainer`, run metadata, identity hashes,
                and orchestrator-level state). Files written this way are
                valid resume sources for
                `CanonicalParallelTempering.resume`.
            ensemble_cls: `CanonicalEnsemble` or a subclass thereof, used by
                every worker's Replica. Spawn workers re-import the class by
                fully qualified name. Top-level classes in a ``python
                script.py`` invocation work (the worker re-runs the script as
                ``__main__``); classes defined in a Jupyter cell or REPL do
                not — `ProcessPool` rejects the interactive-``__main__``
                case up-front rather than letting it surface as an opaque
                multiprocessing error.
            ensemble_kwargs: extra keyword arguments forwarded to
                ``ensemble_cls(...)`` for every worker's Replica. Cannot
                include ``structure``, ``calculator``, ``temperature``, or
                ``random_seed`` (these are set by `Replica`). All values must
                be picklable for the spawn boundary.
        """
        temperatures_list = [float(T) for T in temperatures]
        seed_sequence = np.random.SeedSequence(int(random_seed))
        child_seeds = seed_sequence.spawn(len(temperatures_list) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]

        tmpdir = tempfile.TemporaryDirectory()
        try:
            ce_path = Path(tmpdir.name) / "cluster_expansion.ce"
            cluster_expansion.write(str(ce_path))
            pool = ProcessPool(
                ce_path=ce_path,
                initial_atoms=atoms,
                temperatures=temperatures_list,
                seeds=replica_seeds,
                ensemble_cls=ensemble_cls,
                ensemble_kwargs=ensemble_kwargs,
            )
        except BaseException:
            tmpdir.cleanup()
            raise

        pt = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms,
            temperatures=temperatures_list,
            block_size=block_size,
            random_seed=random_seed,
            pool=pool,
            data_container_file=data_container_file,
        )
        # The constructor's pool-plus-ensemble-kwargs guard prevents
        # forwarding `ensemble_cls`/`ensemble_kwargs` past `pool=`, so
        # the call above leaves the orchestrator's identity hashes
        # computed from the defaults — not from the kwargs the workers
        # actually received. Re-stamp from the real values now, so a
        # checkpoint written by this run records what ran rather than
        # what the constructor's defaults would have been.
        pt._ensemble_cls_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        pt._ensemble_kwargs_hash = _compute_ensemble_kwargs_hash(ensemble_kwargs)
        # Tie tempdir lifetime to the orchestrator: cleaned when `pt`
        # is garbage-collected (or when its finalizer runs explicitly).
        # The CE file is only read by workers during their own
        # startup, so a modest GC delay does not affect correctness.
        weakref.finalize(pt, tmpdir.cleanup)
        return pt
