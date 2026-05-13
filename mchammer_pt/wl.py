"""Wang-Landau parallel tempering (REWL).

Sibling of `mchammer_pt.canonical.CanonicalParallelTempering`. Each
replica owns a fixed energy window; adjacent windows attempt
configuration swaps between cycles using a within-window
log-density-of-states ratio for acceptance.
"""

from __future__ import annotations

import tempfile
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.ensembles import WangLandauEnsemble

from .base import BaseParallelTempering
from .checkpoint import (
    _compute_ce_identity,
    _compute_ensemble_kwargs_hash,
    _write_checkpoint,
)
from .exchange import pair_set_for_cycle
from .history import ExchangeHistory, MetaValue
from .parallel.backend import WangLandauPool
from .parallel.processes import ProcessWangLandauPool
from .parallel.serial import SerialWangLandauPool
from .wl_replica import WangLandauReplica


def _validate_windows(
    windows: Sequence[tuple[float | None, float | None]],
) -> None:
    if len(windows) < 2:
        raise ValueError(
            f"parallel tempering requires at least 2 windows, got {len(windows)}"
        )
    for i, (lo, hi) in enumerate(windows):
        if lo is not None and hi is not None and not (lo < hi):
            raise ValueError(
                f"window {i}: left edge {lo} must be strictly less than "
                f"right edge {hi}"
            )


def _windows_to_array(
    windows: Sequence[tuple[float | None, float | None]],
) -> np.ndarray:
    """Encode windows as a (N, 2) float64 array, using NaN for None edges."""
    out = np.full((len(windows), 2), np.nan, dtype=np.float64)
    for k, (lo, hi) in enumerate(windows):
        if lo is not None:
            out[k, 0] = float(lo)
        if hi is not None:
            out[k, 1] = float(hi)
    return out


def _array_to_windows(
    arr: np.ndarray,
) -> list[tuple[float | None, float | None]]:
    """Inverse of `_windows_to_array`."""
    result: list[tuple[float | None, float | None]] = []
    for row in arr:
        lo = None if np.isnan(row[0]) else float(row[0])
        hi = None if np.isnan(row[1]) else float(row[1])
        result.append((lo, hi))
    return result


class WangLandauParallelTempering(BaseParallelTempering):
    """Single-walker REWL across a sequence of energy windows.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: one starting structure per window. Each structure's
            energy must lie inside its window. Single-`Atoms`
            broadcast is not supported (every window needs an
            initial configuration that lands in that window).
        windows: per-replica energy windows as (left, right) tuples;
            `None` on either side means unbounded.
        energy_spacing: bin size of the WL energy grid (shared
            across replicas).
        block_size: WL trial steps per replica per cycle.
        random_seed: master seed; per-replica seeds and the
            exchange-proposal RNG are deterministically spawned.
        pool: optional `WangLandauPool` to use as backend.
        data_container_file: optional path; if given, `run` writes a
            schema-3 checkpoint to it on completion.
        ensemble_cls: WL ensemble class. Defaults to
            `WangLandauEnsemble`. To use the 1/t schedule, pass
            ``ensemble_kwargs={'schedule': '1_over_t'}``.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.

    Raises:
        TypeError: if `atoms` is a single `Atoms` rather than a sequence.
        ValueError: on window validation or length-mismatch failures.
    """

    _pool: WangLandauPool  # narrow from ReplicaPool

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Sequence[Atoms],
        windows: Sequence[tuple[float | None, float | None]],
        energy_spacing: float,
        block_size: int,
        random_seed: int,
        pool: WangLandauPool | None = None,
        data_container_file: Path | str | None = None,
        *,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(atoms, Atoms):
            raise TypeError(
                "WangLandauParallelTempering requires a sequence of Atoms "
                "(one per window). Each window needs an initial "
                "configuration whose energy lies in that window; there "
                "is no general way to produce one from a single "
                "starting structure."
            )
        atoms_list = list(atoms)
        _validate_windows(windows)
        if len(atoms_list) != len(windows):
            raise ValueError(
                f"atoms has {len(atoms_list)} entries but windows has "
                f"{len(windows)}; supply one Atoms per window."
            )
        if int(block_size) < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        seed_sequence = np.random.SeedSequence(int(random_seed))
        child_seeds = seed_sequence.spawn(len(windows) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]
        master_seed = int(child_seeds[-1].generate_state(1)[0])

        if pool is not None and (
            ensemble_cls is not WangLandauEnsemble or ensemble_kwargs
        ):
            raise ValueError(
                "ensemble_cls / ensemble_kwargs cannot be combined with an "
                "explicit pool=; the pool already owns its replicas. Pass "
                "these kwargs only when letting WangLandauParallelTempering "
                "build the default SerialWangLandauPool, or use "
                "process_pool(...) which forwards them."
            )
        if pool is None:
            replicas = [
                WangLandauReplica(
                    cluster_expansion=cluster_expansion,
                    atoms=a,
                    energy_spacing=energy_spacing,
                    energy_limit_left=lo,
                    energy_limit_right=hi,
                    random_seed=seed,
                    ensemble_cls=ensemble_cls,
                    ensemble_kwargs=ensemble_kwargs,
                )
                for a, (lo, hi), seed in zip(
                    atoms_list, windows, replica_seeds, strict=True
                )
            ]
            pool = SerialWangLandauPool(replicas, energy_spacing=energy_spacing)
        else:
            if len(pool) != len(windows):
                raise ValueError(
                    f"pool has {len(pool)} replicas but windows has "
                    f"{len(windows)} entries."
                )
            if list(pool.windows) != [tuple(w) for w in windows]:
                raise ValueError(
                    f"pool.windows ({list(pool.windows)}) does not match "
                    f"windows ({list(windows)})."
                )
            if pool.energy_spacing != float(energy_spacing):
                raise ValueError(
                    f"pool.energy_spacing ({pool.energy_spacing}) does "
                    f"not match energy_spacing ({energy_spacing})."
                )
        super().__init__(
            pool=pool,
            block_size=block_size,
            random_seed=master_seed,
            template_atoms=atoms_list[0],
        )
        self._windows: list[tuple[float | None, float | None]] = [
            (lo, hi) for lo, hi in windows
        ]
        self._energy_spacing = float(energy_spacing)
        self._data_container_file = data_container_file
        self._random_seed = int(random_seed)
        self._ce_identity = _compute_ce_identity(cluster_expansion)
        # Read identity from the pool when available (process pools
        # carry it); otherwise compute from the constructor args (which
        # are correct for the serial-build path where the guard
        # ensures ensemble_cls/ensemble_kwargs are defaults).
        pool_fqn = getattr(pool, "ensemble_cls_fqn", None)
        pool_hash = getattr(pool, "ensemble_kwargs_hash", None)
        self._ensemble_cls_fqn = pool_fqn if pool_fqn is not None else (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        self._ensemble_kwargs_hash = pool_hash if pool_hash is not None else (
            _compute_ensemble_kwargs_hash(ensemble_kwargs)
        )
        self.cycles_in_segment = 0

    @property
    def windows(self) -> list[tuple[float | None, float | None]]:
        return list(self._windows)

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    def _log_prob_ratio(self, i: int, j: int) -> float:
        """Log of the REWL exchange acceptance ratio.

        Standard REWL detailed balance (Vogel/Li/Wuest 2013):

            log A = ln g_i(E_i) - ln g_i(E_j) + ln g_j(E_j) - ln g_j(E_i)

        When the swap would move a replica to an energy outside its own
        window, the configuration is forbidden in that window's restricted
        state space, so the swap is rejected with probability 1
        (log_r = -inf). `WangLandauPool.log_g_pair` returns -inf for
        out-of-window energies; we detect that and short-circuit rather
        than letting the formula yield +inf (which `_try_exchange` would
        treat as a diagnostic failure).
        """
        E_i = self._pool.current_energy(i)
        E_j = self._pool.current_energy(j)
        g_i_Ei, g_i_Ej, g_j_Ei, g_j_Ej = self._pool.log_g_pair(i, j, E_i, E_j)
        # Per the WangLandauReplica always-in-window invariant, the
        # "same-bin" terms log g_i(E_i) and log g_j(E_j) are never
        # -inf. Only the "cross-bin" terms can be -inf, which we
        # short-circuit below.
        if g_i_Ej == -np.inf or g_j_Ei == -np.inf:
            # Swap would land at least one replica outside its window.
            return -float(np.inf)
        return float((g_i_Ei - g_i_Ej) + (g_j_Ej - g_j_Ei))

    def _checkpoint_meta(self) -> dict[str, MetaValue]:
        return {
            "windows": _windows_to_array(self._windows),
            "energy_spacing": float(self._energy_spacing),
        }

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Advance until `n_cycles` reached or every replica converged.

        Differs from `BaseParallelTempering.run` in only one place: at
        the end of each cycle we query `pool.converged_flags()` and exit
        early if every replica reports True. The returned history's
        rows past the stopping cycle remain at their zero-initialised
        values; `cycles_in_segment` records how far the run got.
        """
        n_replicas = len(self._pool)
        history = ExchangeHistory.empty(n_cycles=n_cycles, n_replicas=n_replicas)
        self._history = history
        history.energies_per_cycle[0] = self._pool.current_energies()
        history.replica_labels_per_cycle[0] = self._replica_labels
        self.cycles_in_segment = 0
        for c in range(n_cycles):
            self._pool.advance_all(self._block_size)
            for pair in pair_set_for_cycle(n_replicas, c):
                self._try_exchange(int(pair), int(pair) + 1, c, history)
            history.energies_per_cycle[c + 1] = self._pool.current_energies()
            history.replica_labels_per_cycle[c + 1] = self._replica_labels
            self.cycles_in_segment = c + 1
            converged = self._pool.converged_flags().all()
            effective_n = c + 1 if converged else n_cycles
            for cb in self._cycle_callbacks:
                cb.on_cycle_end(c, effective_n, history)
            if converged:
                break
        if self._data_container_file is not None:
            _write_checkpoint(self, Path(self._data_container_file))
        return history

    def save_checkpoint(self, path: Path | str) -> None:
        """Write a full checkpoint of this orchestrator atomically."""
        _write_checkpoint(self, path)

    @classmethod
    def resume(
        cls,
        path: Path | str,
        *,
        cluster_expansion: ClusterExpansion,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Resume a previously-checkpointed REWL run.

        Schema-3 only. CE identity, ensemble_cls FQN, and
        ensemble_kwargs hash validate against the checkpoint;
        mismatches raise. Bit-identical resume requires the original
        `_sites_by_species` cache, which is persisted in the
        checkpoint.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
            _validate_kwargs_hash,
        )
        from .history import read_hdf5

        _, containers, meta = read_hdf5(path)
        schema_version = meta.get("schema_version")
        if schema_version != "3":
            raise ValueError(
                f"{path}: unknown schema_version {schema_version!r}; "
                f"this version of mchammer-pt understands '3' only."
            )
        expected_ce_identity = _compute_ce_identity(cluster_expansion)
        if meta["ce_identity"] != expected_ce_identity:
            raise ValueError(f"{path}: CE identity mismatch.")
        expected_ensemble_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        if meta["ensemble_cls_fqn"] != expected_ensemble_fqn:
            raise ValueError(f"{path}: ensemble_cls FQN mismatch.")
        _validate_kwargs_hash(path, meta, ensemble_kwargs, "resume")

        orchestrator_state = _read_orchestrator_state(path)
        replica_extras = _read_replica_extra(path)
        windows = _array_to_windows(np.asarray(meta["windows"]))
        energy_spacing = float(meta["energy_spacing"])
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])

        atoms_list = [container.structure.copy() for container in containers]
        seed_sequence = np.random.SeedSequence(random_seed)
        child_seeds = seed_sequence.spawn(len(windows) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]

        replicas = [
            WangLandauReplica.restart_from(
                container,
                cluster_expansion=cluster_expansion,
                atoms=atoms,
                energy_spacing=energy_spacing,
                energy_limit_left=lo,
                energy_limit_right=hi,
                random_seed=seed,
                ensemble_cls=ensemble_cls,
                ensemble_kwargs=ensemble_kwargs,
                sites_by_species=extra["sites_by_species"],
            )
            for container, atoms, (lo, hi), seed, extra in zip(
                containers,
                atoms_list,
                windows,
                replica_seeds,
                replica_extras,
                strict=True,
            )
        ]
        pool = SerialWangLandauPool(replicas, energy_spacing=energy_spacing)
        pt = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms_list,
            windows=windows,
            energy_spacing=energy_spacing,
            block_size=block_size,
            random_seed=random_seed,
            pool=pool,
        )
        pt._ensemble_cls_fqn = str(meta["ensemble_cls_fqn"])
        pt._ensemble_kwargs_hash = str(meta["ensemble_kwargs_hash"])
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
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Resume a checkpointed REWL run into a `ProcessWangLandauPool`.

        Same identity validation and per-replica restoration as `resume`,
        but reconstructs the pool as a `ProcessWangLandauPool` instead of
        a `SerialWangLandauPool`. Worker scheduling non-determinism means
        the bit-identical contract does NOT hold across the serial-to-
        process or process-to-serial boundary; resume into the same pool
        kind that wrote the checkpoint for bit-identical continuation,
        or accept that cross-pool resume gives a statistically-valid
        continuation only.

        See `resume` for argument and error semantics.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
            _validate_kwargs_hash,
        )
        from .history import read_hdf5

        _, containers, meta = read_hdf5(path)
        schema_version = meta.get("schema_version")
        if schema_version != "3":
            raise ValueError(
                f"{path}: unknown schema_version {schema_version!r}; "
                f"this version of mchammer-pt understands '3' only."
            )
        expected_ce_identity = _compute_ce_identity(cluster_expansion)
        if meta["ce_identity"] != expected_ce_identity:
            raise ValueError(f"{path}: CE identity mismatch.")
        expected_ensemble_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        if meta["ensemble_cls_fqn"] != expected_ensemble_fqn:
            raise ValueError(f"{path}: ensemble_cls FQN mismatch.")
        _validate_kwargs_hash(path, meta, ensemble_kwargs, "resume_process_pool")

        orchestrator_state = _read_orchestrator_state(path)
        replica_extras = _read_replica_extra(path)
        windows = _array_to_windows(np.asarray(meta["windows"]))
        energy_spacing = float(meta["energy_spacing"])
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])

        atoms_list = [container.structure.copy() for container in containers]

        pt = cls.process_pool(
            cluster_expansion=cluster_expansion,
            atoms=atoms_list,
            windows=windows,
            energy_spacing=energy_spacing,
            block_size=block_size,
            random_seed=random_seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
        )
        try:
            pt._pool.restore_replica_state(  # type: ignore[attr-defined]
                containers, replica_extras
            )
            pt._ensemble_cls_fqn = str(meta["ensemble_cls_fqn"])
            pt._ensemble_kwargs_hash = str(meta["ensemble_kwargs_hash"])
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
    def from_bin_count(
        cls,
        cluster_expansion: ClusterExpansion,
        atoms: Sequence[Atoms],
        n_bins: int,
        energy_spacing: float,
        minimum_energy: float,
        maximum_energy: float,
        block_size: int,
        random_seed: int,
        *,
        overlap: int = 4,
        bin_size_exponent: float = 1.0,
        pool: WangLandauPool | None = None,
        data_container_file: Path | str | None = None,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Construct an REWL run from a uniform bin specification.

        Wraps icet's `get_bins_for_parallel_simulations` for the
        common case of an even split. Power users construct
        `windows` by hand.
        """
        from mchammer.ensembles.wang_landau_ensemble import (
            get_bins_for_parallel_simulations,
        )

        raw_windows = get_bins_for_parallel_simulations(
            n_bins=n_bins,
            energy_spacing=energy_spacing,
            minimum_energy=minimum_energy,
            maximum_energy=maximum_energy,
            overlap=overlap,
            bin_size_exponent=bin_size_exponent,
        )
        # icet returns NaN for the unbounded edges of the first and
        # last windows; the orchestrator's window convention uses
        # `None` for unbounded edges. Translate.
        windows: list[tuple[float | None, float | None]] = [
            (
                None if np.isnan(lo) else float(lo),
                None if np.isnan(hi) else float(hi),
            )
            for lo, hi in raw_windows
        ]
        return cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms,
            windows=windows,
            energy_spacing=energy_spacing,
            block_size=block_size,
            random_seed=random_seed,
            pool=pool,
            data_container_file=data_container_file,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
        )

    @classmethod
    def process_pool(
        cls,
        cluster_expansion: ClusterExpansion,
        atoms: Sequence[Atoms],
        windows: Sequence[tuple[float | None, float | None]],
        energy_spacing: float,
        block_size: int,
        random_seed: int,
        data_container_file: Path | str | None = None,
        *,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Construct a process-parallel REWL run in one call.

        Owns CE-write to tempdir and worker spawn; the tempdir is
        cleaned when the returned orchestrator is garbage-collected.
        """
        seed_sequence = np.random.SeedSequence(int(random_seed))
        child_seeds = seed_sequence.spawn(len(windows) + 1)
        replica_seeds = [int(s.generate_state(1)[0]) for s in child_seeds[:-1]]

        tmpdir = tempfile.TemporaryDirectory()
        try:
            ce_path = Path(tmpdir.name) / "cluster_expansion.ce"
            cluster_expansion.write(str(ce_path))
            pool = ProcessWangLandauPool(
                ce_path=ce_path,
                initial_atoms=atoms,
                windows=windows,
                energy_spacing=energy_spacing,
                seeds=replica_seeds,
                ensemble_cls=ensemble_cls,
                ensemble_kwargs=ensemble_kwargs,
            )
        except BaseException:
            tmpdir.cleanup()
            raise
        try:
            pt = cls(
                cluster_expansion=cluster_expansion,
                atoms=atoms,
                windows=windows,
                energy_spacing=energy_spacing,
                block_size=block_size,
                random_seed=random_seed,
                pool=pool,
                data_container_file=data_container_file,
            )
        except BaseException:
            pool.shutdown()
            tmpdir.cleanup()
            raise
        weakref.finalize(pt, tmpdir.cleanup)
        return pt
