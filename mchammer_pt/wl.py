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
from .wl_coordinator import (
    FlatnessMode,
    MergeCadence,
    _validate_flatness_mode,
    _validate_merge_cadence,
)
from .wl_ensemble import CoordinatedWangLandauEnsemble
from .wl_merge_diagnostics import MergeEvent
from .wl_replica import WangLandauReplica, WangLandauSlot
from .wl_result import WindowResult


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


def _spawn_wl_seeds(
    random_seed: int,
    walkers_per_window: Sequence[int],
) -> tuple[list[list[int]], list[int], int]:
    """Spawn deterministic per-walker / per-group / master seeds.

    Reproduces the constructor's SeedSequence walk as a standalone
    helper so both the constructor and the resume path derive seeds
    identically, without duplicating the spawn logic.

    Args:
        random_seed: top-level seed.
        walkers_per_window: walker count per window.

    Returns:
        Tuple ``(walker_seeds, group_seeds, master_seed)`` where
        ``walker_seeds[g][w]`` is walker w's seed in window g,
        ``group_seeds[g]`` is the per-window-group exchange-RNG seed,
        and ``master_seed`` seeds the orchestrator's swap-pair RNG.
    """
    n_windows = len(walkers_per_window)
    seed_sequence = np.random.SeedSequence(int(random_seed))
    total_walker_seeds = sum(walkers_per_window)
    child_seeds = seed_sequence.spawn(total_walker_seeds + n_windows + 1)
    offsets: list[int] = []
    off = 0
    for ww in walkers_per_window:
        offsets.append(off)
        off += ww
    walker_seeds = [
        [
            int(child_seeds[offsets[w] + j].generate_state(1)[0])
            for j in range(walkers_per_window[w])
        ]
        for w in range(n_windows)
    ]
    group_seeds = [
        int(child_seeds[total_walker_seeds + g].generate_state(1)[0])
        for g in range(n_windows)
    ]
    master_seed = int(
        child_seeds[total_walker_seeds + n_windows].generate_state(1)[0]
    )
    return walker_seeds, group_seeds, master_seed


class WangLandauParallelTempering(BaseParallelTempering):
    """REWL orchestrator across a sequence of energy windows.

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
            schema-4 checkpoint to it on completion.
        ensemble_cls: WL ensemble class. Defaults to
            ``CoordinatedWangLandauEnsemble``; must be a subclass of
            it. To use the 1/t schedule, pass
            ``ensemble_kwargs={'schedule': '1_over_t'}``.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.
        n_walkers_per_window: number of independent WL walkers per
            window. Accepts either a single ``int`` applied uniformly
            to all windows, or a ``Sequence[int]`` with one value per
            window. Every window is wrapped in a
            `WangLandauWindowGroup`; the coordinator runs a collective
            flatness gate and halves all walkers in lockstep. With
            count > 1 the group also merges entropies across walkers
            (cadence controlled by ``merge_cadence``). Same-pool resume
            for windows with count > 1 is structurally correct but not
            bit-identical: ``run()``'s end-of-run merge destroys the
            pre-merge per-walker entropy state that bit-identity would
            require.
        flatness_mode: ``"per_walker"`` (every walker independently
            flat; published Vogel et al. 2013) or ``"pooled"`` (default;
            summed histogram flat -- a single combined bin sees ``W x``
            as many samples as any individual walker's bin under the
            same wall-clock budget). Applies to the collective halve
            gate in the halving phase.
        merge_cadence: ``"at_halve"`` (default; Vogel cadence: merge
            entropies at each collective halve) or ``"never"`` (no
            mid-run merge).

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
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
        n_walkers_per_window: int | Sequence[int] = 1,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
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
        n_windows = len(windows)
        if len(atoms_list) != n_windows:
            raise ValueError(
                f"atoms has {len(atoms_list)} entries but windows has "
                f"{n_windows}; supply one Atoms per window."
            )
        if int(block_size) < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        _validate_flatness_mode(flatness_mode)
        _validate_merge_cadence(merge_cadence)

        if isinstance(n_walkers_per_window, int):
            walkers_per_window = [int(n_walkers_per_window)] * n_windows
        else:
            walkers_per_window = [int(w) for w in n_walkers_per_window]
            if len(walkers_per_window) != n_windows:
                raise ValueError(
                    f"n_walkers_per_window has {len(walkers_per_window)} entries "
                    f"but windows has {n_windows}; supply one count per window "
                    f"or a single int."
                )
        if any(w < 1 for w in walkers_per_window):
            raise ValueError(
                f"all n_walkers_per_window values must be >= 1; "
                f"got {walkers_per_window}"
            )
        walker_seeds, group_seeds, master_seed = _spawn_wl_seeds(
            random_seed, walkers_per_window
        )

        if pool is not None and (
            ensemble_cls is not CoordinatedWangLandauEnsemble or ensemble_kwargs
        ):
            raise ValueError(
                "ensemble_cls / ensemble_kwargs cannot be combined with an "
                "explicit pool=; the pool already owns its replicas. Pass "
                "these kwargs only when letting WangLandauParallelTempering "
                "build the default SerialWangLandauPool, or use "
                "process_pool(...) which forwards them."
            )
        if pool is None:
            from .wl_window_group import WangLandauWindowGroup

            slots: list[WangLandauSlot] = []
            for w in range(n_windows):
                lo, hi = windows[w]
                W_w = walkers_per_window[w]
                walker_replicas = [
                    WangLandauReplica(
                        cluster_expansion=cluster_expansion,
                        atoms=atoms_list[w],
                        energy_spacing=energy_spacing,
                        energy_limit_left=lo,
                        energy_limit_right=hi,
                        random_seed=walker_seeds[w][j],
                        ensemble_cls=ensemble_cls,
                        ensemble_kwargs=ensemble_kwargs,
                    )
                    for j in range(W_w)
                ]
                if W_w == 1:
                    slots.append(walker_replicas[0])
                else:
                    slots.append(
                        WangLandauWindowGroup(
                            walker_replicas,
                            random_seed=group_seeds[w],
                        )
                    )
            pool = SerialWangLandauPool(
                slots,
                energy_spacing=energy_spacing,
                flatness_mode=flatness_mode,
                merge_cadence=merge_cadence,
            )
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
        self._flatness_mode: FlatnessMode = flatness_mode
        self._merge_cadence: MergeCadence = merge_cadence
        self._walkers_per_window: list[int] = walkers_per_window
        self._data_container_file = data_container_file
        self._random_seed = int(random_seed)
        self._ce_identity = _compute_ce_identity(cluster_expansion)
        # All built-in pools carry ensemble identity. FQN is always
        # correct (computed from the first replica's ensemble class).
        # kwargs hash is a sentinel on serial pools (the kwargs are
        # consumed during construction and not stored on replicas);
        # fall back to computing from the constructor args.
        self._ensemble_cls_fqn = pool.ensemble_cls_fqn
        self._ensemble_kwargs_hash = (
            pool.ensemble_kwargs_hash
            or _compute_ensemble_kwargs_hash(ensemble_kwargs)
        )
        self.cycles_in_segment = 0

    @property
    def windows(self) -> list[tuple[float | None, float | None]]:
        return list(self._windows)

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    @property
    def merge_events(self) -> tuple[MergeEvent, ...]:
        """Merged-entropy events recorded by the underlying pool.

        See :class:`mchammer_pt.wl_merge_diagnostics.MergeEvent`.
        """
        return self._pool.merge_events

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
        """Return the WL-specific checkpoint metadata.

        Contains the window edges, energy spacing, flatness mode,
        merge cadence, and walkers-per-window boundary array.
        """
        return {
            "windows": _windows_to_array(self._windows),
            "energy_spacing": float(self._energy_spacing),
            "flatness_mode": self._flatness_mode,
            "merge_cadence": self._merge_cadence,
            "walkers_per_window": np.asarray(
                self._walkers_per_window, dtype=np.int32
            ),
        }

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Advance until `n_cycles` reached or every replica converged.

        At the end of each cycle, queries ``pool.converged_flags()``
        and exits early if every replica reports True. The returned
        history's rows past the stopping cycle remain at their
        zero-initialised values; ``cycles_in_segment`` records how
        far the run got.
        """
        n_replicas = len(self._pool)
        history = ExchangeHistory.empty(n_cycles=n_cycles, n_replicas=n_replicas)
        self._history = history
        history.energies_per_cycle[0] = self._pool.current_energies()
        history.replica_labels_per_cycle[0] = self._replica_labels
        self.cycles_in_segment = 0
        try:
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
        finally:
            # End-of-run merge MUST fire on every successful exit path.
            # On an exception path the pool may already have been shut
            # down (e.g. ProcessWangLandauPool.advance_all shuts down on
            # worker errors before re-raising); calling finalise on a
            # closed pool would raise ``RuntimeError("pool is shut
            # down")`` and mask the original failure. Skip the merge
            # when the pool is closed — pt.results() will then surface
            # whatever per-walker state was last collected.
            if self._pool.is_open:
                self._pool.finalise_for_reporting()
        if self._data_container_file is not None:
            # Checkpoint write is deliberately outside the try/finally:
            # on a mid-run exception the on-disk file reflects the last
            # successful ``save_checkpoint()`` call. Callers who want
            # the post-exception in-memory state can read
            # ``pt.results()``, which is consistent because
            # ``finalise_for_reporting`` ran in the finally above.
            _write_checkpoint(self, Path(self._data_container_file))
        return history

    def results(self) -> list[WindowResult]:
        """Per-window analysis output.

        Returns one ``WindowResult`` per energy window. Each result
        wraps the per-walker data containers and provides merged
        ``get_entropy()`` and ``get_histogram()`` methods.
        """
        grouped = self._pool.per_window_data_containers()
        return [
            WindowResult(
                energy_limit_left=float(lo) if lo is not None else float("-inf"),
                energy_limit_right=float(hi) if hi is not None else float("inf"),
                energy_spacing=self._energy_spacing,
                containers=tuple(containers),
            )
            for (lo, hi), containers in zip(self._windows, grouped, strict=True)
        ]

    def save_checkpoint(self, path: Path | str) -> None:
        """Write a full checkpoint of this orchestrator atomically."""
        _write_checkpoint(self, path)

    @classmethod
    def resume(
        cls,
        path: Path | str,
        *,
        cluster_expansion: ClusterExpansion,
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Resume a previously-checkpointed REWL run.

        Schema-4 only. CE identity, ensemble_cls FQN, and
        ensemble_kwargs hash validate against the checkpoint;
        mismatches raise. Bit-identical resume requires the original
        `_sites_by_species` cache, which is persisted in the
        checkpoint. Supports any mix of W=1 and W>1 windows.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
            _read_window_groups,
            _validate_kwargs_hash,
        )
        from .history import read_hdf5
        from .wl_window_group import WangLandauWindowGroup

        _, containers, meta = read_hdf5(path)
        schema_version = meta.get("schema_version")
        if schema_version != "4":
            raise ValueError(
                f"{path}: unsupported schema_version {schema_version!r}; this "
                f"mchammer-pt understands '4' only. For v3 checkpoints, resume "
                f"with mchammer-pt 0.9.0 or earlier."
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
        window_groups = _read_window_groups(path)
        windows = _array_to_windows(np.asarray(meta["windows"]))
        energy_spacing = float(meta["energy_spacing"])
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])
        walkers_per_window = [int(w) for w in np.asarray(meta["walkers_per_window"])]
        # Checkpoints written before ``flatness_mode`` and
        # ``merge_cadence`` were persisted in meta resume with the
        # values that were the defaults at the time of writing
        # ("pooled" and "at_halve").
        flatness_mode: FlatnessMode = str(
            meta.get("flatness_mode", "pooled")
        )  # type: ignore[assignment]
        merge_cadence: MergeCadence = str(
            meta.get("merge_cadence", "at_halve")
        )  # type: ignore[assignment]

        walker_seeds, group_seeds, master_seed = _spawn_wl_seeds(
            random_seed, walkers_per_window
        )

        atoms_list = [container.structure.copy() for container in containers]

        # Build the flat list of replicas (one per walker across all windows).
        # Per-replica RNG state is restored inside restart_from; the seed
        # passed here is only used to initialise the ensemble before restore.
        flat_replicas: list[WangLandauReplica] = []
        flat_idx = 0
        for g, (lo, hi) in enumerate(windows):
            for w in range(walkers_per_window[g]):
                flat_replicas.append(
                    WangLandauReplica.restart_from(
                        containers[flat_idx],
                        cluster_expansion=cluster_expansion,
                        atoms=atoms_list[flat_idx],
                        energy_spacing=energy_spacing,
                        energy_limit_left=lo,
                        energy_limit_right=hi,
                        random_seed=walker_seeds[g][w],
                        ensemble_cls=ensemble_cls,
                        ensemble_kwargs=ensemble_kwargs,
                        sites_by_species=replica_extras[flat_idx]["sites_by_species"],
                    )
                )
                flat_idx += 1

        # Build slots: bare replica for W=1 windows, WindowGroup for W>1.
        slots: list[WangLandauSlot] = []
        offset = 0
        for g in range(len(windows)):
            nw = walkers_per_window[g]
            if nw == 1:
                slots.append(flat_replicas[offset])
            else:
                gs = window_groups[g]
                if gs is None:
                    raise ValueError(
                        f"{path}: /orchestrator/window_groups/{g} missing despite "
                        f"walkers_per_window[{g}] = {nw}; corrupted checkpoint."
                    )
                group = WangLandauWindowGroup(
                    flat_replicas[offset : offset + nw],
                    random_seed=group_seeds[g],
                )
                group._rng.bit_generator.state = json.loads(gs["rng_state"])
                group._exchange_idx = int(gs["exchange_idx"])
                slots.append(group)
            offset += nw

        pool = SerialWangLandauPool(
            slots,
            energy_spacing=energy_spacing,
            flatness_mode=flatness_mode,
            merge_cadence=merge_cadence,
        )

        # One Atoms per window (not per walker) for the constructor.
        atoms_per_window: list[Atoms] = []
        offset = 0
        for g in range(len(windows)):
            atoms_per_window.append(atoms_list[offset])
            offset += walkers_per_window[g]

        pt = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms_per_window,
            windows=windows,
            energy_spacing=energy_spacing,
            block_size=block_size,
            random_seed=random_seed,
            pool=pool,
            flatness_mode=flatness_mode,
            merge_cadence=merge_cadence,
            n_walkers_per_window=walkers_per_window,
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
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
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
            _read_window_groups,
            _validate_kwargs_hash,
        )
        from .history import read_hdf5

        _, containers, meta = read_hdf5(path)
        schema_version = meta.get("schema_version")
        if schema_version != "4":
            raise ValueError(
                f"{path}: unsupported schema_version {schema_version!r}; this "
                f"mchammer-pt understands '4' only. For v3 checkpoints, resume "
                f"with mchammer-pt 0.9.0 or earlier."
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
        window_groups = _read_window_groups(path)
        windows = _array_to_windows(np.asarray(meta["windows"]))
        energy_spacing = float(meta["energy_spacing"])
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])
        walkers_per_window = [int(w) for w in np.asarray(meta["walkers_per_window"])]
        # Checkpoints written before ``flatness_mode`` and
        # ``merge_cadence`` were persisted in meta resume with the
        # values that were the defaults at the time of writing
        # ("pooled" and "at_halve").
        flatness_mode: FlatnessMode = str(
            meta.get("flatness_mode", "pooled")
        )  # type: ignore[assignment]
        merge_cadence: MergeCadence = str(
            meta.get("merge_cadence", "at_halve")
        )  # type: ignore[assignment]

        # One Atoms per window (not per walker) for the constructor path.
        atoms_per_window: list[Atoms] = []
        offset = 0
        for g in range(len(windows)):
            atoms_per_window.append(containers[offset].structure.copy())
            offset += walkers_per_window[g]

        pt = cls.process_pool(
            cluster_expansion=cluster_expansion,
            atoms=atoms_per_window,
            windows=windows,
            energy_spacing=energy_spacing,
            block_size=block_size,
            random_seed=random_seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            n_walkers_per_window=walkers_per_window,
            flatness_mode=flatness_mode,
            merge_cadence=merge_cadence,
        )
        try:
            pt._pool.restore_replica_state(  # type: ignore[attr-defined]
                containers=containers,
                per_walker_extras=replica_extras,
                group_state=window_groups,
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
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
        n_walkers_per_window: int | Sequence[int] = 1,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
    ) -> WangLandauParallelTempering:
        """Construct an REWL run from a uniform bin specification.

        Wraps icet's `get_bins_for_parallel_simulations` for the
        common case of an even split. Power users construct
        `windows` by hand. ``flatness_mode`` and ``merge_cadence``
        have the same meaning as on
        :class:`WangLandauParallelTempering`.
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
            n_walkers_per_window=n_walkers_per_window,
            flatness_mode=flatness_mode,
            merge_cadence=merge_cadence,
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
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
        n_walkers_per_window: int | Sequence[int] = 1,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
    ) -> WangLandauParallelTempering:
        """Construct a process-parallel REWL run in one call.

        Owns CE-write to tempdir and worker spawn; the tempdir is
        cleaned when the returned orchestrator is garbage-collected.
        ``flatness_mode`` and ``merge_cadence`` have the same meaning
        as on :class:`WangLandauParallelTempering`.
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
                n_walkers_per_window=n_walkers_per_window,
                ensemble_cls=ensemble_cls,
                ensemble_kwargs=ensemble_kwargs,
                flatness_mode=flatness_mode,
                merge_cadence=merge_cadence,
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
                n_walkers_per_window=n_walkers_per_window,
                flatness_mode=flatness_mode,
                merge_cadence=merge_cadence,
            )
        except BaseException:
            pool.shutdown()
            tmpdir.cleanup()
            raise
        weakref.finalize(pt, tmpdir.cleanup)
        return pt
