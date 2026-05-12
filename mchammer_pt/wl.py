"""Wang-Landau parallel tempering (REWL).

Sibling of `mchammer_pt.canonical.CanonicalParallelTempering`. Each
replica owns a fixed energy window; adjacent windows attempt
configuration swaps between cycles using a within-window
log-density-of-states ratio for acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.ensembles import WangLandauEnsemble  # type: ignore[import-untyped]
from mchammer.ensembles.one_over_t_wang_landau_ensemble import (  # type: ignore[import-untyped]
    OneOverTWangLandauEnsemble,
)

from .base import BaseParallelTempering
from .checkpoint import (
    _compute_ce_identity,
    _compute_ensemble_kwargs_hash,
    _write_checkpoint,
)
from .exchange import pair_set_for_cycle
from .history import ExchangeHistory, MetaValue
from .parallel.backend import WangLandauPool
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
            `OneOverTWangLandauEnsemble`.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.

    Raises:
        TypeError: if `atoms` is a single `Atoms` rather than a sequence.
        ValueError: on window validation or length-mismatch failures.
    """

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
        ensemble_cls: type[WangLandauEnsemble] = OneOverTWangLandauEnsemble,
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
        self._windows = [tuple(w) for w in windows]
        self._energy_spacing = float(energy_spacing)
        self._data_container_file = data_container_file
        self._random_seed = int(random_seed)
        self._ce_identity = _compute_ce_identity(cluster_expansion)
        self._ensemble_cls_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        self._ensemble_kwargs_hash = _compute_ensemble_kwargs_hash(
            ensemble_kwargs
        )
        self.cycles_completed = 0

    @property
    def windows(self) -> list[tuple[float | None, float | None]]:
        return list(self._windows)

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    def _log_prob_ratio(self, i: int, j: int) -> float:
        E_i = self._pool.current_energy(i)
        E_j = self._pool.current_energy(j)
        g_i_Ei, g_i_Ej, g_j_Ei, g_j_Ej = self._pool.log_g_pair(i, j, E_i, E_j)
        return float((g_i_Ej - g_i_Ei) + (g_j_Ei - g_j_Ej))

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
        values; `cycles_completed` records how far the run got.
        """
        n_replicas = len(self._pool)
        history = ExchangeHistory.empty(n_cycles=n_cycles, n_replicas=n_replicas)
        self._history = history
        history.energies_per_cycle[0] = self._pool.current_energies()
        history.replica_labels_per_cycle[0] = self._replica_labels
        self.cycles_completed = 0
        for c in range(n_cycles):
            self._pool.advance_all(self._block_size)
            for pair in pair_set_for_cycle(n_replicas, c):
                self._try_exchange(int(pair), int(pair) + 1, c, history)
            history.energies_per_cycle[c + 1] = self._pool.current_energies()
            history.replica_labels_per_cycle[c + 1] = self._replica_labels
            for cb in self._cycle_callbacks:
                cb.on_cycle_end(c, n_cycles, history)
            self.cycles_completed = c + 1
            if self._pool.converged_flags().all():
                break
        if self._data_container_file is not None:
            _write_checkpoint(self, Path(self._data_container_file))
        return history
