"""Wang-Landau parallel tempering (REWL).

Sibling of `mchammer_pt.canonical.CanonicalParallelTempering`. Each
replica owns a fixed energy window; adjacent windows attempt
configuration swaps between cycles using a within-window
log-density-of-states ratio for acceptance.
"""

from __future__ import annotations

import math
import tempfile
import warnings
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.observers.base_observer import BaseObserver

from .base import BaseParallelTempering
from .checkpoint import (
    _compute_ce_identity,
    _compute_ensemble_kwargs_hash,
    _write_checkpoint,
)
from .history import ExchangeHistory, MetaValue
from .parallel.backend import WangLandauPool, _RecorderAttach
from .parallel.processes import ProcessWangLandauPool
from .parallel.serial import SerialWangLandauPool
from .wl_coordinator import (
    FlatnessMode,
    MergeCadence,
    OneOverTEntry,
    OneOverTGate,
    _validate_bp_stall_multiple,
    _validate_flatness_mode,
    _validate_merge_cadence,
    _validate_one_over_t_entry,
    _validate_one_over_t_gate,
    reconstruct_stall_state,
)
from .wl_ensemble import (
    CoordinatedWangLandauEnsemble,
    _validate_dos_snapshot_ratio,
    _validate_recency_visits_per_bin,
)
from .wl_initial_structures import expand_initial_structures
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


def _warn_post_merge_resume_if_multi_walker(
    walkers_per_window: Sequence[int],
    caller: str,
) -> None:
    """Emit a `UserWarning` on resume when any window has W > 1.

    The end-of-run `pool.finalise_for_reporting()` destructively merges
    per-walker entropies, so any checkpoint written after `run()`
    returned captures the post-merge state. A resumed run continues
    from merged entropy rather than from per-walker entropy, which
    means the trajectory will not be bit-identical to an
    uninterrupted run. The warning fires unconditionally for any
    W > 1 window because there is no on-disk marker distinguishing
    pre-merge (mid-run save) from post-merge (post-run save)
    checkpoints — the safe assumption is that the resumed trajectory
    diverges. The warning does not fire for all-W=1 checkpoints,
    which retain the bit-identical contract.
    """
    multi = [g for g, w in enumerate(walkers_per_window) if w > 1]
    if not multi:
        return
    warnings.warn(
        f"{caller}: windows {multi} have walkers_per_window > 1. "
        f"The end-of-run entropy merge in pool.finalise_for_reporting() "
        f"is destructive, so the resumed trajectory is not bit-identical "
        f"to an uninterrupted run (only structurally correct). See "
        f"WangLandauParallelTempering.resume docstring for the full "
        f"contract.",
        UserWarning,
        stacklevel=3,
    )


def _spawn_wl_seeds(
    random_seed: int,
    walkers_per_window: Sequence[int],
) -> tuple[list[list[int]], list[int], int]:
    """Spawn deterministic per-walker / per-group / master seeds.

    Used by the in-process constructor and by `resume` / `resume_process_pool`
    on the serial-pool path so both derive seeds identically without
    duplicating the SeedSequence walk. `process_pool` and the
    `resume_process_pool` worker-spawn path use mchammer-pt's worker
    builder, which independently constructs its own SeedSequence; the
    contract there is that both paths produce identical seed plans for
    the same `random_seed` + `walkers_per_window`.

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


def _restored_replica_labels(
    raw: object,
    pt: WangLandauParallelTempering,
    path: Path | str,
) -> np.ndarray:
    """Validate and coerce the restored orchestrator replica-label array.

    The labels are position-indexed over ``N_w = sum(walkers_per_window)``.
    A length mismatch against the freshly reconstructed pool signals a
    corrupted or mismatched checkpoint; this mirrors the adjacent
    ``walkers_per_window`` and container-length corruption guards.
    """
    labels = np.asarray(raw, dtype=np.int64)
    expected = pt._pool.n_carriers()
    if labels.shape != (expected,):
        raise ValueError(
            f"{path}: orchestrator replica_labels has shape {tuple(labels.shape)} "
            f"but the reconstructed pool has {expected} carriers; "
            f"corrupted or mismatched checkpoint."
        )
    return labels


def _decode_dos_snapshot_ratio(meta: dict[str, MetaValue]) -> float | None:
    """Read ``dos_snapshot_ratio`` from checkpoint meta, decoding NaN as None.

    ``MetaValue`` has no ``None``, so the disabled state is stored as NaN
    (see ``WangLandauParallelTempering._checkpoint_meta``). Older
    checkpoints lacking the key default to 2.0 (the factor-2 ladder),
    matching the constructor default.
    """
    raw = meta.get("dos_snapshot_ratio", 2.0)
    if isinstance(raw, float) and math.isnan(raw):
        return None
    return _validate_dos_snapshot_ratio(raw)


class WangLandauParallelTempering(BaseParallelTempering):
    """REWL orchestrator across a sequence of energy windows.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: one entry per window. Each entry is either a single
            ``Atoms`` (broadcast: every walker in that window starts
            from a copy of it) or a ``Sequence[Atoms]`` of length
            ``n_walkers_per_window`` for that window (one structure per
            walker, in walker order). Windows may mix the two forms.
            Every structure's energy must lie inside its window. A bare
            ``Atoms`` for the whole argument is rejected (every window
            needs an initial configuration that lands in that window).
        windows: per-replica energy windows as (left, right) tuples;
            `None` on either side means unbounded.
        energy_spacing: bin size of the WL energy grid (shared
            across replicas).
        block_size: WL trial steps per replica per cycle.
        random_seed: master seed; per-replica seeds and the
            exchange-proposal RNG are deterministically spawned.
        pool: optional `WangLandauPool` to use as backend.
        data_container_file: optional path; if given, `run` writes a
            schema-5 checkpoint to it on completion.
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
            (cadence controlled by ``merge_cadence``). At each active
            window boundary a random matching pairs the two windows'
            walkers and attempts one swap per pair, so per-walker
            exchange rate does not dilute as walkers are added.
            Same-pool resume for windows with count > 1 is structurally
            correct but not bit-identical: ``run()``'s end-of-run merge
            destroys the pre-merge per-walker entropy state that
            bit-identity would require.
        flatness_mode: ``"per_walker"`` (every walker independently
            flat; published Vogel et al. 2013) or ``"pooled"`` (default;
            summed histogram flat -- a single combined bin sees ``W x``
            as many samples as any individual walker's bin under the
            same wall-clock budget). Applies to the collective halve
            gate in the halving phase.
        merge_cadence: ``"at_halve"`` (default; Vogel cadence: merge
            entropies at each collective halve) or ``"never"`` (no
            mid-run merge).
        recency_visits_per_bin: EWMA timescale for the recency-flatness
            diagnostic, in expected visits per bin (the decay constant
            is roughly ``recency_visits_per_bin * N_bins`` MC steps).
            Default 1000; larger gives a longer, smoother averaging
            window. Must be a positive integer. Recorded in the
            checkpoint and adopted from there on resume.
        dos_snapshot_ratio: ratio of the log fill-factor ladder on which
            ln g(E) snapshots are recorded during the 1/t regime, for
            convergence-vs-length diagnostics. Default 2.0 (a snapshot
            each time f halves); None disables snapshotting. Read back
            via WindowResult.get_entropy(fill_factor_limit=...).
        one_over_t_gate: halving-phase gate under the 1/t schedule.
            ``"visit_once"`` (default) halves once every bin has been
            visited, with the coupled Belardinelli-Pereyra switch.
            ``"flatness"`` halves on the WL flatness criterion
            (``min(H) >= flatness_limit * mean(H)``, reusing the
            ``flatness_limit`` ensemble kwarg) bundled with a decoupled,
            stall-safe switch evaluated every block. Only meaningful under
            ``ensemble_kwargs={"schedule": "1_over_t"}``; selecting
            ``"flatness"`` without that schedule raises. Recorded in the
            checkpoint and adopted from there on resume.
        bp_stall_multiple: only consulted under ``one_over_t_gate="flatness"``.
            A window that has halved at least once but then stalls (cannot
            meet the flatness gate) adopts the 1/t schedule once it has run
            ``bp_stall_multiple`` times its first-stage duration since its
            last halve. Default 4.0; larger is more patient. Recorded in
            the checkpoint and adopted from there on resume.
        one_over_t_entry: how a window's fill factor enters the 1/t
            phase at the BP switch. ``"window_clock"`` (default):
            f jumps to ``1/(step - window_entry + 1)`` and the 1/t
            clock runs from window entry. ``"f_continuous"`` starts the 1/t clock
            from the f that halving actually reached, so f is
            continuous across the switch; this applies at every switch
            path (canonical and stall, coupled and decoupled) and is
            orthogonal to ``one_over_t_gate``. Requires
            ``ensemble_kwargs={"schedule": "1_over_t"}``; selecting it
            without that schedule raises. Recorded in the checkpoint
            and adopted from there on resume.

    Raises:
        TypeError: if `atoms` is a single `Atoms` rather than a sequence.
        ValueError: on window validation or length-mismatch failures.
    """

    _pool: WangLandauPool  # narrow from ReplicaPool

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Sequence[Atoms | Sequence[Atoms]],
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
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        one_over_t_gate: OneOverTGate = "visit_once",
        bp_stall_multiple: float = 4.0,
        one_over_t_entry: OneOverTEntry = "window_clock",
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
        recency_visits_per_bin = _validate_recency_visits_per_bin(
            recency_visits_per_bin
        )
        dos_snapshot_ratio = _validate_dos_snapshot_ratio(dos_snapshot_ratio)
        _validate_one_over_t_gate(one_over_t_gate)
        bp_stall_multiple = _validate_bp_stall_multiple(bp_stall_multiple)
        _validate_one_over_t_entry(one_over_t_entry)

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
        walker_atoms = expand_initial_structures(atoms_list, walkers_per_window)
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
                        atoms=walker_atoms[w][j],
                        energy_spacing=energy_spacing,
                        energy_limit_left=lo,
                        energy_limit_right=hi,
                        random_seed=walker_seeds[w][j],
                        ensemble_cls=ensemble_cls,
                        ensemble_kwargs=ensemble_kwargs,
                        recency_visits_per_bin=recency_visits_per_bin,
                        dos_snapshot_ratio=dos_snapshot_ratio,
                        one_over_t_entry=one_over_t_entry,
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
                one_over_t_gate=one_over_t_gate,
                bp_stall_multiple=bp_stall_multiple,
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
            template_atoms=walker_atoms[0][0],
        )
        self._windows: list[tuple[float | None, float | None]] = [
            (lo, hi) for lo, hi in windows
        ]
        self._energy_spacing = float(energy_spacing)
        # The pool is the single source of truth for halving/switch policy.
        # The constructor's policy kwargs only build the default pool (when
        # pool is None); read the effective values back from the pool so an
        # explicit pool= cannot diverge from the persisted checkpoint meta.
        self._flatness_mode: FlatnessMode = pool.flatness_mode
        self._merge_cadence: MergeCadence = pool.merge_cadence
        self._one_over_t_gate: OneOverTGate = pool.one_over_t_gate
        self._bp_stall_multiple: float = pool.bp_stall_multiple
        self._one_over_t_entry: OneOverTEntry = pool.one_over_t_entry
        # Walker-side diagnostics config lives on the replicas, which
        # pools do not expose; store it from the constructor arguments.
        self._recency_visits_per_bin: int = recency_visits_per_bin
        self._dos_snapshot_ratio: float | None = dos_snapshot_ratio
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

    def _log_prob_ratio(self, i: int, a: int, j: int, b: int) -> float:
        """Log of the REWL exchange acceptance ratio.

        Standard REWL detailed balance (Vogel/Li/Wuest 2013) for walker
        ``a`` of window ``i`` against walker ``b`` of window ``j``:

            log A = ln g_i(E_i) - ln g_i(E_j) + ln g_j(E_j) - ln g_j(E_i)

        When the swap would move a walker to an energy outside its own
        window, the configuration is forbidden in that window's restricted
        state space, so the swap is rejected with probability 1
        (log_r = -inf). `WangLandauPool.walker_log_g` returns -inf for
        out-of-window energies; we detect that and short-circuit rather
        than letting the formula yield +inf (which `_propose_boundary`
        would treat as a diagnostic failure).
        """
        E_i = self._pool.walker_energy(i, a)
        E_j = self._pool.walker_energy(j, b)
        g_i_Ei = self._pool.walker_log_g(i, a, E_i)
        g_i_Ej = self._pool.walker_log_g(i, a, E_j)
        g_j_Ei = self._pool.walker_log_g(j, b, E_i)
        g_j_Ej = self._pool.walker_log_g(j, b, E_j)
        # Per the WangLandauReplica always-in-window invariant, the
        # "same-bin" terms log g_i(E_i) and log g_j(E_j) are never
        # -inf. Only the "cross-bin" terms can be -inf, which we
        # short-circuit below.
        if g_i_Ej == -np.inf or g_j_Ei == -np.inf:
            # Swap would land at least one walker outside its window.
            return -float(np.inf)
        return float((g_i_Ei - g_i_Ej) + (g_j_Ej - g_j_Ei))

    def _checkpoint_meta(self) -> dict[str, MetaValue]:
        """Return the WL-specific checkpoint metadata.

        Contains the window edges, energy spacing, flatness mode,
        merge cadence, recency visits per bin, DOS snapshot ratio,
        and walkers-per-window boundary array.
        """
        return {
            "windows": _windows_to_array(self._windows),
            "energy_spacing": float(self._energy_spacing),
            "flatness_mode": self._flatness_mode,
            "merge_cadence": self._merge_cadence,
            "recency_visits_per_bin": int(self._recency_visits_per_bin),
            "one_over_t_gate": self._one_over_t_gate,
            "bp_stall_multiple": float(self._bp_stall_multiple),
            "one_over_t_entry": self._one_over_t_entry,
            # None is not in MetaValue; encode "disabled" as NaN.
            # The resume path reverses this: a NaN reads back as None.
            "dos_snapshot_ratio": (
                float(self._dos_snapshot_ratio)
                if self._dos_snapshot_ratio is not None
                else float("nan")
            ),
            "walkers_per_window": np.asarray(
                self._walkers_per_window, dtype=np.int32
            ),
        }

    def record_observable(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer for per-bin microcanonical moment accumulation.

        Requires the pool to satisfy `_RecorderAttach`, which
        `SerialWangLandauPool` and `ProcessWangLandauPool` both implement.
        Pools that do not expose recorder attach raise `TypeError`.

        Each selected window's replica(s) receive their own deserialised
        copy of ``observer`` via a pickle round-trip. Recorders are
        restore-aware: if a prior checkpoint stored a state for
        ``observer.tag``, the recorder seeds from it on resume.

        Args:
            observer: any ``mchammer.BaseObserver`` whose
                ``get_observable`` returns a scalar, sequence, or Mapping.
            replicas: ``"all"`` or an explicit sequence of window indices.

        Raises:
            TypeError: if the pool does not support recorder attach.
            TypeError: if ``observer`` is not picklable.
            ValueError: if a recorder for this tag is already attached.
        """
        if not isinstance(self._pool, _RecorderAttach):
            raise TypeError(
                f"record_observable requires a pool that supports recorder "
                f"attach; {type(self._pool).__name__} does not."
            )
        self._pool.record_observable(observer, replicas)

    def run(self, n_cycles: int) -> ExchangeHistory:
        """Advance until `n_cycles` reached or every replica converged.

        At the end of each cycle, queries ``pool.converged_flags()``
        and exits early if every replica reports True. The returned
        history's rows past the stopping cycle remain at their
        zero-initialised values; ``cycles_in_segment`` records how
        far the run got.
        """
        n_replicas = len(self._pool)
        history = ExchangeHistory.empty(
            n_cycles=n_cycles,
            n_replicas=n_replicas,
            n_carriers=self._pool.n_carriers(),
            window_of_position=self._pool.window_of_position(),
        )
        self._history = history
        # Reset the counter atomically with `_history` (before the row-0
        # snapshot writes) so it is never stale relative to the history a
        # checkpoint would serialise.
        self.cycles_in_segment = 0
        history.energies_per_cycle[0] = self._pool.current_energies()
        history.replica_labels_per_cycle[0] = self._replica_labels
        try:
            for c in range(n_cycles):
                self._pool.advance_all(self._block_size)
                self._exchange_phase(c, history)
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
        """Write a full checkpoint of this orchestrator atomically.

        For windows with `n_walkers_per_window > 1`, the checkpoint
        captures whatever entropy state the walkers hold at the time
        of the call. If save_checkpoint is invoked **after** ``run()``
        has returned, the saved per-walker entropies have already been
        merged by ``finalise_for_reporting`` (run-end side-effect), so
        the on-disk state cannot reconstruct the pre-merge trajectory
        — resuming from such a file produces a structurally correct
        but not bit-identical continuation. See `resume` and the
        class docstring for the full W>1 contract.
        """
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
        allow_kwargs_mismatch: bool = False,
        _frozen: bool = False,
    ) -> WangLandauParallelTempering:
        """Resume a previously-checkpointed REWL run.

        Resumes schema-5 checkpoints, and schema-4 checkpoints from
        single-walker runs; a schema-4 multi-walker checkpoint is
        rejected (its window-indexed labels are incompatible with the
        current walker-indexed layout). CE identity, ensemble_cls FQN,
        and ensemble_kwargs hash validate against the checkpoint;
        mismatches raise. Bit-identical resume requires the original
        `_sites_by_species` cache, which is persisted in the
        checkpoint. Supports any mix of W=1 and W>1 windows.

        For windows with ``n_walkers_per_window > 1`` the resumed
        continuation is structurally correct but not bit-identical
        to an uninterrupted run: ``save_checkpoint`` captures the
        merged-entropy state left behind by ``finalise_for_reporting``
        at the end of each ``run()``, and that merge is destructive.
        A `UserWarning` is emitted at resume time naming the affected
        windows. The relaxation does not apply to all-W=1
        checkpoints, which retain the bit-identical contract.

        Set ``allow_kwargs_mismatch=True`` to downgrade an
        ``ensemble_kwargs`` hash mismatch from a hard error to a
        `UserWarning`. Only the kwargs-identity check is relaxed; CE
        identity and ``ensemble_cls`` are still enforced. Use it to
        resume across software environments (differing Python, numpy, or
        platform) where the pickle of identical move objects differs;
        bit-identical continuation is not guaranteed.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
            _read_window_groups,
            _validate_kwargs_hash,
            _validate_wl_schema_version,
        )
        from .history import read_hdf5
        from .wl_window_group import WangLandauWindowGroup

        _, containers, meta = read_hdf5(path)
        _validate_wl_schema_version(path, meta.get("schema_version"))
        expected_ce_identity = _compute_ce_identity(cluster_expansion)
        if meta["ce_identity"] != expected_ce_identity:
            raise ValueError(f"{path}: CE identity mismatch.")
        expected_ensemble_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        if meta["ensemble_cls_fqn"] != expected_ensemble_fqn:
            raise ValueError(f"{path}: ensemble_cls FQN mismatch.")
        _validate_kwargs_hash(
            path,
            meta,
            ensemble_kwargs,
            "resume",
            allow_mismatch=allow_kwargs_mismatch,
        )

        windows = _array_to_windows(np.asarray(meta["windows"]))
        energy_spacing = float(meta["energy_spacing"])
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])
        walkers_per_window = [int(w) for w in np.asarray(meta["walkers_per_window"])]
        if len(walkers_per_window) != len(windows):
            raise ValueError(
                f"{path}: walkers_per_window has "
                f"{len(walkers_per_window)} entries but windows has "
                f"{len(windows)}; checkpoint is corrupted."
            )
        expected_m = sum(walkers_per_window)
        if len(containers) != expected_m:
            raise ValueError(
                f"{path}: walker-count mismatch — "
                f"sum(walkers_per_window) = {expected_m} but "
                f"file contains {len(containers)} replica containers; "
                f"checkpoint is corrupted or truncated."
            )
        orchestrator_state = _read_orchestrator_state(path)
        replica_extras = _read_replica_extra(path)
        window_groups = _read_window_groups(path, containers)
        if not _frozen:
            _warn_post_merge_resume_if_multi_walker(walkers_per_window, "resume")
        flatness_mode: FlatnessMode = str(
            meta["flatness_mode"]
        )  # type: ignore[assignment]
        merge_cadence: MergeCadence = str(
            meta["merge_cadence"]
        )  # type: ignore[assignment]
        recency_visits_per_bin = _validate_recency_visits_per_bin(
            meta.get("recency_visits_per_bin", 1000)
        )
        dos_snapshot_ratio = _decode_dos_snapshot_ratio(meta)
        one_over_t_gate: OneOverTGate = str(
            meta.get("one_over_t_gate", "visit_once")
        )  # type: ignore[assignment]
        _validate_one_over_t_gate(one_over_t_gate)
        bp_stall_multiple = _validate_bp_stall_multiple(
            meta.get("bp_stall_multiple", 4.0)
        )
        one_over_t_entry: OneOverTEntry = str(
            meta.get("one_over_t_entry", "window_clock")
        )  # type: ignore[assignment]
        _validate_one_over_t_entry(one_over_t_entry)

        # _master_seed unused: the orchestrator RNG is restored from
        # orchestrator_state["rng_state"] further down.
        walker_seeds, group_seeds, _master_seed = _spawn_wl_seeds(
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
                        recency_visits_per_bin=recency_visits_per_bin,
                        dos_snapshot_ratio=dos_snapshot_ratio,
                        one_over_t_entry=one_over_t_entry,
                        frozen_g=_frozen,
                    )
                )
                flat_idx += 1

        # Build slots: bare replica for W=1 windows, WindowGroup for W>1.
        slots: list[WangLandauSlot] = []
        offset = 0
        for g in range(len(windows)):
            nw = walkers_per_window[g]
            if nw == 1:
                if window_groups[g] is not None:
                    raise ValueError(
                        f"{path}: /orchestrator/window_groups/{g} present despite "
                        f"walkers_per_window[{g}] = 1; corrupted checkpoint."
                    )
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
                # Per-walker MC state was restored inside restart_from
                # above; restore_state would redo that work. Apply only
                # the group-level RNG state here.
                group._rng.bit_generator.state = json.loads(gs["rng_state"])
                slots.append(group)
            offset += nw

        pool = SerialWangLandauPool(
            slots,
            energy_spacing=energy_spacing,
            flatness_mode=flatness_mode,
            merge_cadence=merge_cadence,
            one_over_t_gate=one_over_t_gate,
            bp_stall_multiple=bp_stall_multiple,
            frozen_measurement=_frozen,
        )

        # Reconstruct per-slot stall state from the restored fill-factor
        # history (its halve boundaries) and each walker's window entry.
        per_slot_stall: list[tuple[int | None, int | None]] = []
        offset = 0
        for g in range(len(windows)):
            nw = walkers_per_window[g]
            slot_replicas = flat_replicas[offset : offset + nw]
            ffh_keys = list(slot_replicas[0].ensemble._fill_factor_history.keys())
            entries = [r.ensemble._window_entry_step for r in slot_replicas]
            per_slot_stall.append(reconstruct_stall_state(ffh_keys, entries))
            offset += nw
        pool.seed_stall_state(per_slot_stall)

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
            recency_visits_per_bin=recency_visits_per_bin,
            dos_snapshot_ratio=dos_snapshot_ratio,
            one_over_t_gate=one_over_t_gate,
            bp_stall_multiple=bp_stall_multiple,
            one_over_t_entry=one_over_t_entry,
            n_walkers_per_window=walkers_per_window,
        )
        pt._ensemble_cls_fqn = str(meta["ensemble_cls_fqn"])
        pt._ensemble_kwargs_hash = str(meta["ensemble_kwargs_hash"])
        pt._replica_labels = _restored_replica_labels(
            orchestrator_state["replica_labels"], pt, path
        )
        rng_state_raw = orchestrator_state["rng_state"]
        assert isinstance(rng_state_raw, str)
        pt._rng.bit_generator.state = json.loads(rng_state_raw)
        return pt

    @classmethod
    def measure_from_checkpoint(
        cls,
        path: Path | str,
        *,
        cluster_expansion: ClusterExpansion,
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Load a converged checkpoint in frozen measurement mode.

        Every replica's g(E) is held fixed (``frozen_g=True``) and the pool
        coordinator is disabled (``frozen_measurement=True``): MC steps proceed
        and exchanges are proposed, but halving, entropy merging, and phase
        switching are all skipped. The density of states is unchanged.

        Each ``run(n_cycles)`` call starts a fresh cycle segment independent of
        the DOS run's cycle accounting.

        Typical usage::

            pt = WangLandauParallelTempering.measure_from_checkpoint(
                "run.h5", cluster_expansion=ce
            )
            pt.record_observable(my_observer)
            pt.run(n_cycles=500)

        Args:
            path: path to a checkpoint produced by a completed REWL run
                (any schema version ``resume`` accepts).
            cluster_expansion: icet ``ClusterExpansion`` used in the
                original run.
            ensemble_cls: WL ensemble class (default
                ``CoordinatedWangLandauEnsemble``).
            ensemble_kwargs: extra kwargs forwarded to ensemble construction.
                Must match the checkpoint's hash.

        Returns:
            A `WangLandauParallelTempering` in frozen measurement mode.
        """
        return cls.resume(
            path,
            cluster_expansion=cluster_expansion,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            _frozen=True,
        )

    @classmethod
    def measure_from_checkpoint_process_pool(
        cls,
        path: Path | str,
        *,
        cluster_expansion: ClusterExpansion,
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> WangLandauParallelTempering:
        """Load a converged checkpoint in frozen-g measurement mode (process pool).

        Returns a process-backed orchestrator in frozen-g measurement mode.

        Every worker ensemble's g(E) is held fixed (``frozen_g=True``
        in the worker's :class:`WangLandauReplica` construction) and the
        pool coordinator is disabled (``frozen_measurement=True``): MC
        steps proceed and exchanges are proposed, but halving, entropy
        merging, and phase switching are all skipped.

        Mirrors :meth:`measure_from_checkpoint` but uses a
        :class:`ProcessWangLandauPool` instead of a
        :class:`SerialWangLandauPool`.

        Args:
            path: path to a checkpoint produced by a completed REWL run
                (any schema version ``resume_process_pool`` accepts).
            cluster_expansion: icet ``ClusterExpansion`` used in the
                original run.
            ensemble_cls: WL ensemble class (default
                ``CoordinatedWangLandauEnsemble``).
            ensemble_kwargs: extra kwargs forwarded to ensemble construction.
                Must match the checkpoint's hash.

        Returns:
            A `WangLandauParallelTempering` backed by a
            `ProcessWangLandauPool` in frozen measurement mode.
        """
        return cls.resume_process_pool(
            path,
            cluster_expansion=cluster_expansion,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            _frozen=True,
        )

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
        allow_kwargs_mismatch: bool = False,
        _frozen: bool = False,
    ) -> WangLandauParallelTempering:
        """Resume a checkpointed REWL run into a `ProcessWangLandauPool`.

        Same identity validation and per-replica restoration as `resume`,
        but reconstructs the pool as a `ProcessWangLandauPool` instead of
        a `SerialWangLandauPool`. Worker scheduling non-determinism means
        the bit-identical contract does NOT hold across the serial-to-
        process or process-to-serial boundary; resume into the same pool
        kind that wrote the checkpoint for bit-identical continuation,
        or accept that cross-pool resume gives a statistically-valid
        continuation only. The same W>1 entropy-merge relaxation
        documented on `resume` applies here too; a `UserWarning` is
        emitted for any window with ``walkers_per_window[g] > 1``.

        See `resume` for argument and error semantics.

        ``_frozen`` is an internal flag used by
        ``measure_from_checkpoint_process_pool`` to load the checkpoint
        with ``frozen_g=True`` on every worker ensemble and disable the
        master-side coordinator. Not part of the public API.

        Set ``allow_kwargs_mismatch=True`` to downgrade an
        ``ensemble_kwargs`` hash mismatch from a hard error to a
        `UserWarning`; only the kwargs-identity check is relaxed (CE
        identity and ``ensemble_cls`` stay enforced). For resuming across
        software environments where the pickle of identical move objects
        differs; bit-identical continuation is not guaranteed.
        """
        import json

        from .checkpoint import (
            _read_orchestrator_state,
            _read_replica_extra,
            _read_window_groups,
            _validate_kwargs_hash,
            _validate_wl_schema_version,
        )
        from .history import read_hdf5

        _, containers, meta = read_hdf5(path)
        _validate_wl_schema_version(path, meta.get("schema_version"))
        expected_ce_identity = _compute_ce_identity(cluster_expansion)
        if meta["ce_identity"] != expected_ce_identity:
            raise ValueError(f"{path}: CE identity mismatch.")
        expected_ensemble_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        if meta["ensemble_cls_fqn"] != expected_ensemble_fqn:
            raise ValueError(f"{path}: ensemble_cls FQN mismatch.")
        _validate_kwargs_hash(
            path,
            meta,
            ensemble_kwargs,
            "resume_process_pool",
            allow_mismatch=allow_kwargs_mismatch,
        )

        windows = _array_to_windows(np.asarray(meta["windows"]))
        energy_spacing = float(meta["energy_spacing"])
        block_size = int(meta["block_size"])
        random_seed = int(meta["random_seed"])
        walkers_per_window = [int(w) for w in np.asarray(meta["walkers_per_window"])]
        if len(walkers_per_window) != len(windows):
            raise ValueError(
                f"{path}: walkers_per_window has "
                f"{len(walkers_per_window)} entries but windows has "
                f"{len(windows)}; checkpoint is corrupted."
            )
        expected_m = sum(walkers_per_window)
        if len(containers) != expected_m:
            raise ValueError(
                f"{path}: walker-count mismatch — "
                f"sum(walkers_per_window) = {expected_m} but "
                f"file contains {len(containers)} replica containers; "
                f"checkpoint is corrupted or truncated."
            )
        orchestrator_state = _read_orchestrator_state(path)
        replica_extras = _read_replica_extra(path)
        window_groups = _read_window_groups(path, containers)
        if not _frozen:
            _warn_post_merge_resume_if_multi_walker(
                walkers_per_window,
                "resume_process_pool",
            )
        flatness_mode: FlatnessMode = str(
            meta["flatness_mode"]
        )  # type: ignore[assignment]
        merge_cadence: MergeCadence = str(
            meta["merge_cadence"]
        )  # type: ignore[assignment]
        recency_visits_per_bin = _validate_recency_visits_per_bin(
            meta.get("recency_visits_per_bin", 1000)
        )
        dos_snapshot_ratio = _decode_dos_snapshot_ratio(meta)
        one_over_t_gate: OneOverTGate = str(
            meta.get("one_over_t_gate", "visit_once")
        )  # type: ignore[assignment]
        _validate_one_over_t_gate(one_over_t_gate)
        bp_stall_multiple = _validate_bp_stall_multiple(
            meta.get("bp_stall_multiple", 4.0)
        )
        one_over_t_entry: OneOverTEntry = str(
            meta.get("one_over_t_entry", "window_clock")
        )  # type: ignore[assignment]
        _validate_one_over_t_entry(one_over_t_entry)

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
            recency_visits_per_bin=recency_visits_per_bin,
            dos_snapshot_ratio=dos_snapshot_ratio,
            one_over_t_gate=one_over_t_gate,
            bp_stall_multiple=bp_stall_multiple,
            one_over_t_entry=one_over_t_entry,
            _frozen=_frozen,
        )
        try:
            pt._pool.restore_replica_state(  # type: ignore[attr-defined]
                containers=containers,
                per_walker_extras=replica_extras,
                group_state=window_groups,
            )
            # Reconstruct per-slot stall state parent-side from the
            # restored container _last_state (fill-factor history halve
            # boundaries and each walker's window entry).
            offset = 0
            for g, slot in enumerate(pt._pool._slots):  # type: ignore[attr-defined]
                nw = walkers_per_window[g]
                slot_containers = containers[offset : offset + nw]
                ffh_keys = list(
                    slot_containers[0]._last_state.get(
                        "fill_factor_history", {}
                    ).keys()
                )
                entries = [
                    c._last_state.get("window_entry_step")
                    for c in slot_containers
                ]
                slot.last_halve_step, slot.first_halve_duration = (
                    reconstruct_stall_state(ffh_keys, entries)
                )
                offset += nw
            pt._ensemble_cls_fqn = str(meta["ensemble_cls_fqn"])
            pt._ensemble_kwargs_hash = str(meta["ensemble_kwargs_hash"])
            pt._replica_labels = _restored_replica_labels(
                orchestrator_state["replica_labels"], pt, path
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
        atoms: Sequence[Atoms | Sequence[Atoms]],
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
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        one_over_t_gate: OneOverTGate = "visit_once",
        bp_stall_multiple: float = 4.0,
        one_over_t_entry: OneOverTEntry = "window_clock",
    ) -> WangLandauParallelTempering:
        """Construct an REWL run from a uniform bin specification.

        Wraps icet's `get_bins_for_parallel_simulations` for the
        common case of an even split. Power users construct
        `windows` by hand. The ensemble and policy keywords (``flatness_mode``,
        ``merge_cadence``, ``recency_visits_per_bin``,
        ``dos_snapshot_ratio``, ``one_over_t_gate``, ``bp_stall_multiple``,
        ``one_over_t_entry``)
        have the same meaning as on
        :class:`WangLandauParallelTempering`.

        ``atoms`` accepts the same broadcast-or-per-walker shape as
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
            recency_visits_per_bin=recency_visits_per_bin,
            dos_snapshot_ratio=dos_snapshot_ratio,
            one_over_t_gate=one_over_t_gate,
            bp_stall_multiple=bp_stall_multiple,
            one_over_t_entry=one_over_t_entry,
        )

    @classmethod
    def process_pool(
        cls,
        cluster_expansion: ClusterExpansion,
        atoms: Sequence[Atoms | Sequence[Atoms]],
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
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        one_over_t_gate: OneOverTGate = "visit_once",
        bp_stall_multiple: float = 4.0,
        one_over_t_entry: OneOverTEntry = "window_clock",
        _frozen: bool = False,
    ) -> WangLandauParallelTempering:
        """Construct a process-parallel REWL run in one call.

        Owns CE-write to tempdir and worker spawn; the tempdir is
        cleaned when the returned orchestrator is garbage-collected.
        The ensemble and policy keywords (``flatness_mode``,
        ``merge_cadence``, ``recency_visits_per_bin``,
        ``dos_snapshot_ratio``, ``one_over_t_gate``, ``bp_stall_multiple``,
        ``one_over_t_entry``)
        have the same meaning as on
        :class:`WangLandauParallelTempering`.

        ``atoms`` accepts the same broadcast-or-per-walker shape as
        :class:`WangLandauParallelTempering`.

        ``_frozen`` is an internal flag used by ``resume_process_pool``
        and ``measure_from_checkpoint_process_pool`` to build worker
        ensembles with ``frozen_g=True`` and disable the master-side
        coordinator. Not part of the public API.
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
                recency_visits_per_bin=recency_visits_per_bin,
                dos_snapshot_ratio=dos_snapshot_ratio,
                one_over_t_gate=one_over_t_gate,
                bp_stall_multiple=bp_stall_multiple,
                one_over_t_entry=one_over_t_entry,
                frozen_measurement=_frozen,
                frozen_g=_frozen,
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
                recency_visits_per_bin=recency_visits_per_bin,
                dos_snapshot_ratio=dos_snapshot_ratio,
                one_over_t_gate=one_over_t_gate,
                bp_stall_multiple=bp_stall_multiple,
                one_over_t_entry=one_over_t_entry,
            )
        except BaseException:
            pool.shutdown()
            tmpdir.cleanup()
            raise
        weakref.finalize(pt, tmpdir.cleanup)
        return pt
