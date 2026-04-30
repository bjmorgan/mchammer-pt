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
from .history import ExchangeHistory
from .parallel.backend import ReplicaPool
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
        pool: optional `ReplicaPool` to use as the execution backend.
            If None (the default), a `SerialPool` is constructed from
            ``cluster_expansion``, ``atoms``, ``temperatures``, and the
            spawned per-replica seeds.
        data_container_file: optional path; if given, `run` writes an
            HDF5 bundle of the `ExchangeHistory`, each replica's
            `mchammer.BaseDataContainer`, and run metadata to this path
            on completion.
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
        pool: ReplicaPool | None = None,
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

    @property
    def temperatures(self) -> np.ndarray:
        """Copy of the per-replica temperature array (kelvin)."""
        return self._temperatures.copy()

    def _log_prob_ratio(self, i: int, j: int) -> float:
        E_i = self._pool.current_energy(i)
        E_j = self._pool.current_energy(j)
        return float((self._beta[i] - self._beta[j]) * (E_i - E_j))

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Run `n_cycles` PT cycles, optionally writing an HDF5 bundle.

        When `data_container_file` was provided at construction, the
        `ExchangeHistory`, each replica's `mchammer.BaseDataContainer`,
        and a metadata dict (temperatures and block size) are written
        to that path as a single HDF5 file on completion.
        """
        history = super().run(n_cycles=n_cycles)
        if self._data_container_file is not None:
            from .history import write_hdf5

            write_hdf5(
                Path(self._data_container_file),
                history=history,
                replica_containers=self._pool.data_containers(),
                meta={
                    "temperatures": self._temperatures,
                    "block_size": int(self._block_size),
                },
            )
        return history

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
            data_container_file: optional path; if given, `run` writes an
                HDF5 bundle of the `ExchangeHistory`, each replica's
                `mchammer.BaseDataContainer`, and run metadata to this path
                on completion.
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
        # Tie tempdir lifetime to the orchestrator: cleaned when `pt`
        # is garbage-collected (or when its finalizer runs explicitly).
        # The CE file is only read by workers during their own
        # startup, so a modest GC delay does not affect correctness.
        weakref.finalize(pt, tmpdir.cleanup)
        return pt
