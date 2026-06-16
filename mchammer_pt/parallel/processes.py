"""Persistent-worker multiprocessing pool -- parent-side orchestration.

One OS process per walker. Workers live in spawn-mode subprocesses
implemented in ``_worker.py``; this file contains only the parent-side
``ProcessPool`` class and the per-worker ``Pipe``-based command/reply
plumbing.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from ase import Atoms
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.ensembles import (
    CanonicalEnsemble,
)
from mchammer.observers.base_observer import (
    BaseObserver,
)

from ..checkpoint import _compute_ensemble_kwargs_hash, _serialise_rng_state
from ..exchange import matching_for_boundary
from ..replica import Replica
from ..wl_coordinator import (
    FlatnessMode,
    MergeCadence,
    OneOverTEntry,
    OneOverTGate,
    Phase,
    Schedule,
    SlotView,
    WalkerPostBlockState,
    _compute_filled_bins,
    _compute_per_walker_breakdown,
    _compute_per_walker_flat_min,
    _compute_recency_flatness,
    _validate_bp_stall_multiple,
    _validate_entry_schedule,
    _validate_flatness_mode,
    _validate_gate_schedule,
    _validate_merge_cadence,
    _validate_one_over_t_entry,
    _validate_one_over_t_gate,
    decide_block_actions,
    merge_entropies,
)
from ..wl_ensemble import (
    CoordinatedWangLandauEnsemble,
    _validate_dos_snapshot_ratio,
    _validate_recency_visits_per_bin,
)
from ..wl_initial_structures import expand_initial_structures
from ..wl_merge_diagnostics import MergeEvent
from ..wl_replica import WangLandauReplica, log_g_at
from ._builder import AtomsSpec, CanonicalBuilder, WLBuilder
from ._comms import broadcast_gather, fanout_gather, recv_reply, request
from ._imports import _check_importable, _resolve_replicas
from ._worker import _wl_worker, _worker


class ProcessPool:
    """Persistent-worker multiprocessing pool.

    One OS process per replica. Satisfies `ObservablePool`: observers
    can be attached via three paths, each suited to a different kind
    of observer:

    - ``attach_observer(observer)`` — for observers that pickle as
      whole instances (most stock `mchammer` observers without icet
      construction inputs, and most user observers built from basic
      types). Each worker receives its own deserialised copy via a
      pickle round-trip.
    - ``attach_observer_class(cls, /, *args, **kwargs)`` — for
      observers whose constructor arguments are picklable but whose
      constructed instance is awkward to ship. Each worker constructs
      its own ``cls(*args, **kwargs)`` locally.
    - ``attach_observer_factory(factory)`` — for observers whose
      constructor takes icet objects (``ClusterSpace``,
      ``ClusterExpansion``) that do not pickle. The factory runs
      inside each worker with that worker's ``Replica``; reload the
      CE from disk via
      ``ClusterExpansion.read(replica.cluster_expansion_path)``
      (``ProcessPool`` auto-populates ``cluster_expansion_path`` on
      every worker).

    Args:
        ce_path: path to a CE file readable by ``ClusterExpansion.read``.
        initial_atoms: starting structure, either a single ``Atoms``
            (each worker receives a copy) or a sequence of ``Atoms``
            (one per temperature, length-validated against
            ``temperatures``).
        temperatures: one temperature per replica.
        seeds: one random seed per replica.
        ensemble_cls: `CanonicalEnsemble` or a subclass thereof, used
            by every worker's Replica. Spawn workers re-import the
            class by fully qualified name, so it must live in an
            importable module: top-level classes in a
            ``python script.py`` invocation work (the worker re-runs
            the script as ``__main__``); classes defined in a Jupyter
            cell or REPL do not. Move such classes to a ``.py``
            module file. The interactive-``__main__`` case is
            rejected up-front in ``__init__`` rather than producing a
            deep multiprocessing traceback. The same constraint
            applies to the class argument of ``attach_observer_class``
            and the callable argument of ``attach_observer_factory``.
        ensemble_kwargs: extra keyword arguments forwarded to
            ``ensemble_cls(...)``. All values must be picklable.
            Cannot include the four kwargs reserved by `Replica`
            (`structure`, `calculator`, `temperature`, `random_seed`);
            a clash raises in the worker and surfaces via the
            startup handshake.
    """

    def __init__(
        self,
        ce_path: Path | str,
        initial_atoms: Atoms | Sequence[Atoms],
        temperatures: Sequence[float],
        seeds: Sequence[int],
        *,
        ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        _check_importable(ensemble_cls, kind="ensemble_cls")
        temperatures_list = list(temperatures)
        seeds_list = list(seeds)
        if len(temperatures_list) != len(seeds_list):
            raise ValueError("temperatures and seeds must be the same length")
        self._temperatures: list[float] = [float(T) for T in temperatures_list]
        self._workers: list[tuple[mp.process.BaseProcess, Connection]] = []
        if isinstance(initial_atoms, Atoms):
            atoms_specs = [
                AtomsSpec.from_atoms(initial_atoms)
                for _ in range(len(temperatures_list))
            ]
        else:
            atoms_list = list(initial_atoms)
            if len(atoms_list) != len(temperatures_list):
                raise ValueError(
                    f"initial_atoms has {len(atoms_list)} entries but "
                    f"temperatures has {len(temperatures_list)}"
                )
            atoms_specs = [AtomsSpec.from_atoms(a) for a in atoms_list]
        extra_kwargs: dict[str, Any] = (
            dict(ensemble_kwargs) if ensemble_kwargs else {}
        )
        # Cover both spawn-time failures (e.g. ``process.start()``
        # raising ``PicklingError`` when ``extra_kwargs`` contains an
        # unpicklable value) and handshake-time failures with one
        # cleanup path. ``ctx.Process(...).start()`` pickles ``args=``
        # eagerly, so a failure on iteration N>1 leaves N-1 daemon
        # workers in ``self._workers`` that ``shutdown()`` then joins.
        ctx = mp.get_context("spawn")
        try:
            for T, seed, atoms_spec in zip(
                self._temperatures, seeds_list, atoms_specs, strict=True
            ):
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                builder = CanonicalBuilder(
                    ce_path=str(ce_path),
                    atoms=atoms_spec,
                    temperature=T,
                    seed=int(seed),
                    ensemble_cls=ensemble_cls,
                    ensemble_kwargs=dict(extra_kwargs),
                )
                process = ctx.Process(
                    target=_worker,
                    args=(child_conn, builder),
                    daemon=True,
                )
                process.start()
                child_conn.close()
                self._workers.append((process, parent_conn))

            # Synchronous ready-handshake. Each worker sends a single
            # OK after successful Replica construction, or ERR +
            # traceback if startup fails. Surfacing failures here means
            # the caller gets the actual traceback, rather than a
            # BrokenPipeError on the first ADVANCE with the original
            # cause lost.
            for i, (_, conn) in enumerate(self._workers):
                recv_reply(conn, "STARTUP", i)
        except BaseException:
            self.shutdown()
            raise
        self._ensemble_cls_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        self._ensemble_kwargs_hash = _compute_ensemble_kwargs_hash(
            ensemble_kwargs
        )

    def _check_open(self) -> None:
        if not self._workers:
            raise RuntimeError("pool is shut down")

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully-qualified name of the ensemble class used by workers."""
        return self._ensemble_cls_fqn

    @property
    def ensemble_kwargs_hash(self) -> str:
        """SHA-256 hash of the ensemble kwargs forwarded to workers."""
        return self._ensemble_kwargs_hash

    def _drain_remaining_replies(self, indices: list[int]) -> None:
        """Read pending replies on the given worker connections, ignoring contents."""
        for i in indices:
            _, conn = self._workers[i]
            try:
                conn.recv()
            except (EOFError, BrokenPipeError):
                pass

    def _recv_or_abort_attach(
        self,
        conn: Connection,
        op: str,
        i: int,
        remaining: list[int],
    ) -> None:
        """Receive an attach reply or abort the pool with full cleanup.

        Two failure paths produce the same outcome — drain pending
        replies on later workers, shut the pool down, raise:

        - Worker reports ERR (factory raised, isinstance check
          failed, etc.) — message includes the worker traceback.
        - Pipe closes (worker died via Ctrl-C, OOM, segfault) —
          message says the worker exited unexpectedly during attach.

        After this returns the pool is guaranteed shut down and
        further operations refuse via _check_open.
        """
        try:
            recv_reply(conn, op, i)
        except (RuntimeError, TypeError):
            self._drain_remaining_replies(remaining)
            self.shutdown()
            raise

    def __len__(self) -> int:
        self._check_open()
        return len(self._workers)

    @property
    def temperatures(self) -> list[float]:
        self._check_open()
        return list(self._temperatures)

    def advance_all(self, n_steps: int) -> None:
        self._check_open()
        targets = [(conn, i) for i, (_, conn) in enumerate(self._workers)]
        broadcast_gather(targets, ("ADVANCE", int(n_steps)))

    def current_energies(self) -> np.ndarray:
        self._check_open()
        targets = [(conn, i) for i, (_, conn) in enumerate(self._workers)]
        payloads = broadcast_gather(targets, ("ENERGY",))
        return np.array(payloads, dtype=np.float64)

    def current_energy(self, i: int) -> float:
        self._check_open()
        _, conn = self._workers[i]
        return float(request(conn, ("ENERGY",), i))

    def current_occupations(self, i: int) -> np.ndarray:
        self._check_open()
        _, conn = self._workers[i]
        return np.asarray(request(conn, ("GET_OCC",), i))

    def swap_configurations(self, i: int, j: int) -> None:
        self._check_open()
        _, conn_i = self._workers[i]
        _, conn_j = self._workers[j]
        occ_i, occ_j = broadcast_gather(
            [(conn_i, i), (conn_j, j)], ("GET_OCC",)
        )
        request(conn_i, ("SET_OCC", np.asarray(occ_j, dtype=np.int64)), i)
        try:
            request(conn_j, ("SET_OCC", np.asarray(occ_i, dtype=np.int64)), j)
        except BaseException:
            request(conn_i, ("SET_OCC", np.asarray(occ_i, dtype=np.int64)), i)
            raise

    def n_walkers(self, i: int) -> int:
        """Single-walker rungs: always 1."""
        return 1

    def walker_energy(self, i: int, walker: int) -> float:
        """Energy of rung ``i`` (single-walker; ``walker`` is always 0)."""
        return self.current_energy(i)

    def candidate_pairs(
        self, i: int, j: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Single-walker rungs exchange the one pair ``(0, 0)``.

        Returns the fixed pair without drawing from ``rng``, so the
        exchange RNG stream is left untouched.
        """
        return [(0, 0)]

    def window_of_position(self) -> np.ndarray:
        """One position per rung: the identity mapping."""
        return np.arange(len(self._workers), dtype=np.int64)

    def n_carriers(self) -> int:
        """One walker per rung."""
        return len(self._workers)

    def swap_walker_configurations(self, i: int, a: int, j: int, b: int) -> None:
        """Delegate to ``swap_configurations`` (rungs are single-walker)."""
        self.swap_configurations(i, j)

    def apply_swaps(self, swaps: list[tuple[int, int, int, int]]) -> None:
        """Apply accepted walker-config swaps, one at a time."""
        self._check_open()
        for i, a, j, b in swaps:
            self.swap_walker_configurations(i, a, j, b)

    def data_containers(self) -> list[BaseDataContainer]:
        self._check_open()
        targets = [(conn, i) for i, (_, conn) in enumerate(self._workers)]
        return broadcast_gather(targets, ("GET_DC",))

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]]:
        self._check_open()
        targets = [(conn, i) for i, (_, conn) in enumerate(self._workers)]
        return broadcast_gather(targets, ("SNAPSHOT_FOR_CHECKPOINT",))

    def restore_replica_state(
        self,
        containers: list[BaseDataContainer],
        replica_extras: list[dict[str, Any]],
    ) -> None:
        """Push saved per-replica state into each worker.

        Sends each worker its corresponding `BaseDataContainer` and the
        ``sites_by_species`` cache from ``replica_extras``; the worker
        applies them via `Replica.restore_state`. Used by
        `CanonicalParallelTempering.resume_process_pool` to bring a
        freshly-spawned process pool to the saved state.

        Args:
            containers: one container per replica, in replica order.
                Length must equal `len(self)`.
            replica_extras: one per-replica extras dict per worker.
                Each must carry a ``"sites_by_species"`` key.

        Raises:
            RuntimeError: pool is shut down, or any worker reports an
                error during restoration.
            ValueError: lengths of ``containers`` and ``replica_extras``
                do not match `len(self)`.
        """
        self._check_open()
        if len(containers) != len(self):
            raise ValueError(
                f"restore_replica_state expects {len(self)} containers, "
                f"got {len(containers)}"
            )
        if len(replica_extras) != len(self):
            raise ValueError(
                f"restore_replica_state expects {len(self)} extras dicts, "
                f"got {len(replica_extras)}"
            )
        for (_, conn), container, extra in zip(
            self._workers, containers, replica_extras, strict=True
        ):
            conn.send(
                ("RESTORE_STATE", container, extra["sites_by_species"])
            )
        for i, (_, conn) in enumerate(self._workers):
            recv_reply(conn, "RESTORE_STATE", i)

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to selected workers.

        Each selected worker receives its own deserialised copy via a
        pickle round-trip in the worker. The parent eagerly validates
        picklability before contacting any worker. Failure semantics:
        the parent's ``pickle.dumps`` raising leaves all workers
        untouched; a worker raising during ``pickle.loads`` shuts the
        pool down, ensuring subsequent operations raise via
        ``_check_open``.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._workers))
        if not target_indices:
            return
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        for i in target_indices:
            _, conn = self._workers[i]
            conn.send(("ATTACH_OBS", blob))
        for offset, i in enumerate(target_indices):
            _, conn = self._workers[i]
            self._recv_or_abort_attach(
                conn, "ATTACH_OBS", i, target_indices[offset + 1:]
            )

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        replicas: Sequence[int] | Literal["all"] = "all",
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to selected workers.

        Each selected worker constructs its own ``cls(*args, **kwargs)``
        locally. Multiprocessing pickles ``cls`` by fully qualified name
        — the same constraint as ``ensemble_cls``. Eager parent-side
        checks: importability of ``cls``, picklability of ``(args, kwargs)``,
        and a dry-run construction that asserts the result is a
        ``BaseObserver``.

        The constructor must be free of externally-visible side effects:
        the dry-run runs in the parent's address space (not in any worker),
        and is followed by one construction per selected worker.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._workers))
        if not target_indices:
            return
        _check_importable(cls, kind="observer class")
        try:
            pickle.dumps((args, kwargs))
        except Exception as exc:
            raise TypeError(
                f"attach_observer_class: args/kwargs for "
                f"{cls.__name__} are not picklable ({exc})"
            ) from exc
        probe = cls(*args, **kwargs)
        if not isinstance(probe, BaseObserver):
            raise TypeError(
                f"attach_observer_class: {cls.__name__}(...) returned "
                f"{type(probe).__name__}, not a BaseObserver"
            )
        del probe
        for i in target_indices:
            _, conn = self._workers[i]
            conn.send(("ATTACH_OBS_CLS", cls, args, kwargs))
        for offset, i in enumerate(target_indices):
            _, conn = self._workers[i]
            self._recv_or_abort_attach(
                conn, "ATTACH_OBS_CLS", i, target_indices[offset + 1:]
            )

    def get_observers(self, replica_index: int) -> dict[str, BaseObserver]:
        """Return a snapshot of the observers attached to one worker.

        The returned dict is keyed by observer tag. Values are
        independent copies — the worker pickles its observer dict
        on send and the parent unpickles, so mutations on the
        returned objects do not affect the worker's running state.

        Args:
            replica_index: zero-based index of the replica to query.

        Raises:
            IndexError: if ``replica_index`` is out of range.
            TypeError: if the observer dict cannot be round-tripped
                through pickle.
            RuntimeError: if the pool is shut down, the worker
                exited unexpectedly, or the worker reports any
                other ERR.
        """
        self._check_open()
        n = len(self._workers)
        if not 0 <= replica_index < n:
            raise IndexError(
                f"replica index {replica_index} out of range "
                f"for pool of size {n}"
            )
        _, conn = self._workers[replica_index]
        return request(conn, ("GET_OBSERVERS",), replica_index)

    def attach_observer_factory(
        self,
        factory: Callable[[Replica], BaseObserver],
        *,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer constructed inside each worker.

        Each selected worker calls ``factory(replica)`` locally and
        attaches the returned ``BaseObserver``. Use this for observers
        whose constructors take icet objects (``ClusterSpace``,
        ``ClusterExpansion``) that do not pickle. The factory should
        reload the CE from disk inside the worker::

            def make_obs(replica):
                ce = ClusterExpansion.read(replica.cluster_expansion_path)
                return ClusterCountObserver(
                    ce.get_cluster_space_copy(), ..., interval=...
                )

        ``ProcessPool`` auto-populates ``replica.cluster_expansion_path``
        on every worker from the ``ce_path`` supplied at pool
        construction.

        ``factory`` must be a top-level function or class method
        importable by fully qualified name; lambdas, locally-defined
        functions, and callables defined in interactive ``__main__``
        do not survive pickling and are rejected up-front.

        Eager parent-side checks: importability of ``factory`` and
        picklability of ``factory``. Unlike `attach_observer_class`,
        there is no parent-side dry-run because the parent has no
        `Replica` instances — construction failures surface from the
        worker instead. Construction errors inside the worker (the
        factory raising, or returning a non-``BaseObserver``) surface
        via the standard worker-error path as ``RuntimeError`` with
        the worker traceback. On a worker-side construction failure the
        pool shuts down, ensuring subsequent operations raise via
        ``_check_open``.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._workers))
        if not target_indices:
            return
        _check_importable(factory, kind="observer factory")
        try:
            pickle.dumps(factory)
        except Exception as exc:
            raise TypeError(
                f"attach_observer_factory: factory "
                f"{getattr(factory, '__name__', repr(factory))!r} "
                f"is not picklable ({exc})"
            ) from exc
        for i in target_indices:
            _, conn = self._workers[i]
            conn.send(("ATTACH_OBS_FACTORY", factory))
        for offset, i in enumerate(target_indices):
            _, conn = self._workers[i]
            self._recv_or_abort_attach(
                conn, "ATTACH_OBS_FACTORY", i, target_indices[offset + 1:]
            )

    # Idempotent: bypasses _check_open so __exit__ and the __init__ failure
    # path can call it unconditionally.
    def shutdown(self) -> None:
        for _, conn in self._workers:
            try:
                conn.send(("SHUTDOWN",))
                conn.recv()
            except (EOFError, BrokenPipeError):
                pass
            conn.close()
        for process, _ in self._workers:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
        self._workers = []

    def __enter__(self) -> ProcessPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.shutdown()


class ProcessWangLandauWindow:
    """Per-window remote walkers + coordinator-facing methods.

    Holds W worker subprocesses for a single energy window. Exposes
    the same coordinator-facing interface as
    ``WangLandauWindowGroup`` but implemented via pipe commands to
    the workers. Owns the local communication pattern; the policy
    decisions live in ``wl_coordinator`` and are invoked by
    ``ProcessWangLandauPool.advance_all``.

    Attributes:
        workers: list of (Process, Connection) pairs, one per walker.
        rng: per-window RNG whose state is checkpointed for
            reproducible resume.
        phase: collective WL phase, either ``"halving"`` or
            ``"1_over_t"``. Flipped by the coordinator after a
            collective BP switch.
    """

    def __init__(
        self,
        workers: list[tuple[mp.process.BaseProcess, Connection]],
        rng: np.random.Generator,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
        schedule: Schedule = "halving",
        flatness_limit: float = 0.8,
        one_over_t_gate: OneOverTGate = "visit_once",
        bp_stall_multiple: float = 4.0,
    ) -> None:
        _validate_flatness_mode(flatness_mode)
        _validate_merge_cadence(merge_cadence)
        _validate_one_over_t_gate(one_over_t_gate)
        _validate_gate_schedule(one_over_t_gate, schedule)
        self.workers = workers
        self.rng = rng
        self._flatness_mode: FlatnessMode = flatness_mode
        self._merge_cadence: MergeCadence = merge_cadence
        self._schedule: Schedule = schedule
        self._flatness_limit: float = float(flatness_limit)
        self._one_over_t_gate: OneOverTGate = one_over_t_gate
        self._bp_stall_multiple: float = _validate_bp_stall_multiple(
            bp_stall_multiple
        )
        self.last_halve_step: int | None = None
        self.first_halve_duration: int | None = None
        self.phase: Phase = "halving"
        self.walker_states: list[WalkerPostBlockState] = [
            WalkerPostBlockState(
                halving_criterion_met=False,
                fill_factor=1.0,
                entropy={},
                step=0,
                window_entry_step=None,
                histogram={},
                reached_energy_window=False,
                current_energy=0.0,
            )
            for _ in workers
        ]

    def collect_entropy_snapshots(self) -> list[dict[int, float]]:
        return [dict(s.entropy) for s in self.walker_states]

    def collect_ts(self) -> list[int]:
        """Per-walker ``t = step - window_entry_step + 1``.

        Requires every walker to have entered its window. Callers
        should gate on :meth:`has_unentered_walker` before calling.
        Raises ``RuntimeError`` if any walker has not yet entered.
        """
        ts: list[int] = []
        for s in self.walker_states:
            if s.window_entry_step is None:
                raise RuntimeError(
                    "collect_ts called on slot with unentered walker; "
                    "gate on has_unentered_walker() first"
                )
            ts.append(s.step - s.window_entry_step + 1)
        return ts

    def has_unentered_walker(self) -> bool:
        """True iff any walker has not yet reached its window."""
        return any(s.window_entry_step is None for s in self.walker_states)


def _merge_per_window_stats(
    slot_stats: list[dict[str, Any]],
    flatness_mode: FlatnessMode,
) -> dict[str, Any]:
    """Merge per-walker WL stats dicts for one window into a single dict.

    For multi-walker slots, builds a summed histogram and a union of
    MC-visited bins across walkers (from the ``visited_bins`` field
    that the worker attaches to every WL_STATS reply), reports
    ``bins_visited`` and ``bins_known`` from those unions, reports
    ``bins_filled`` (union of positive bins for pooled flatness,
    intersection for per-walker), and adds the per-walker flat-min,
    a ``per_walker_breakdown`` list, plus the slot's flatness mode.
    Resolves a window-level ``recency_flatness`` from the per-walker
    ``recency_weights`` (the EWMA effective weights each worker
    attaches to its WL_STATS reply) and carries the ``schedule``.
    Single-walker slots are returned unchanged apart from stripping
    the internal ``visited_bins`` and ``recency_weights`` fields.

    Pure function on the IPC payload.
    """
    if len(slot_stats) == 1:
        merged = dict(slot_stats[0])
        merged.pop("visited_bins", None)
        merged.pop("recency_weights", None)
        return merged
    combined_hist: dict[int, int] = {}
    visited_union: set[int] = set()
    histograms = [s["histogram"] for s in slot_stats]
    for s in slot_stats:
        for k, v in s["histogram"].items():
            combined_hist[k] = combined_hist.get(k, 0) + v
        visited_union.update(s.get("visited_bins", ()))
    per_walker_flat_min = _compute_per_walker_flat_min(histograms)
    recency_weights = [s["recency_weights"] for s in slot_stats]
    return {
        "fill_factor": slot_stats[0]["fill_factor"],
        "halvings": slot_stats[0]["halvings"],
        "histogram": combined_hist,
        "bins_visited": len(visited_union),
        "bins_filled": _compute_filled_bins(histograms, flatness_mode),
        "bins_known": len(combined_hist),
        "converged": all(s["converged"] for s in slot_stats),
        "flatness_mode": flatness_mode,
        "per_walker_flat_min": per_walker_flat_min,
        "per_walker_breakdown": _compute_per_walker_breakdown(histograms),
        "phase": slot_stats[0]["phase"],
        "recency_flatness": _compute_recency_flatness(
            recency_weights, flatness_mode
        ),
        "schedule": slot_stats[0]["schedule"],
    }


def _view_of(slot: ProcessWangLandauWindow) -> SlotView:
    """Build a SlotView from a process-backed slot's cached state."""
    return SlotView(
        walker_states=tuple(slot.walker_states),
        phase=slot.phase,
        flatness_mode=slot._flatness_mode,
        merge_cadence=slot._merge_cadence,
        schedule=slot._schedule,
        flatness_limit=slot._flatness_limit,
        one_over_t_gate=slot._one_over_t_gate,
        bp_stall_multiple=slot._bp_stall_multiple,
        last_halve_step=slot.last_halve_step,
        first_halve_duration=slot.first_halve_duration,
    )


class ProcessWangLandauPool:
    """Persistent-worker REWL pool.

    Satisfies `WangLandauObservablePool`:
    observers can be attached via the same three paths as
    `ProcessPool` — `attach_observer`, `attach_observer_class`,
    `attach_observer_factory` — and snapshotted back via
    `get_observers`. Observers fire inside each WL replica's
    `advance(...)` between exchange proposals.

    Args:
        ce_path: path to a CE file readable by `ClusterExpansion.read`.
        initial_atoms: one entry per window. Each entry is either a
            single ``Atoms`` (broadcast: every walker in that window
            starts from a copy) or a ``Sequence[Atoms]`` of length
            ``n_walkers_per_window`` for that window (one per walker).
            Every structure's energy must lie inside its window. A bare
            ``Atoms`` for the whole argument is rejected (every window
            needs its own initial configuration).
        windows: per-replica energy windows.
        energy_spacing: bin size shared across replicas.
        seeds: one random seed per window.
        n_walkers_per_window: walkers per energy window; > 1 enables
            collective halving across walkers; entropy merging cadence
            is controlled by ``merge_cadence``.
        ensemble_cls: WL ensemble class. Defaults to
            ``CoordinatedWangLandauEnsemble``; must be a subclass of
            it (the coordinator owns halving). To use the 1/t
            schedule, pass ``ensemble_kwargs={'schedule':
            '1_over_t'}``. Spawned workers re-import by FQN;
            interactive-``__main__`` classes are not supported.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.
            Must be picklable for the spawn boundary.
        flatness_mode: ``"per_walker"`` (published Vogel) or ``"pooled"``
            (default; halve when summed histogram is flat).
        merge_cadence: ``"at_halve"`` (default; Vogel cadence) or
            ``"never"`` (no mid-run merge, end-of-run finalisation only).
        recency_visits_per_bin: EWMA timescale (default 1000) forwarded
            to every walker's ensemble for the recency-flatness
            diagnostic.
        one_over_t_entry: 1/t entry policy forwarded to every walker's
            replica (see WangLandauParallelTempering). Selecting
            ``"f_continuous"`` without
            ``ensemble_kwargs={"schedule": "1_over_t"}`` raises.
        frozen_measurement: if ``True``, ``advance_all`` fans out
            ``ADVANCE`` to all workers as normal (so walkers accumulate
            MC steps and observables) but the master-side coordinator
            decisions (halving, entropy-merge, phase-switch) are skipped.
            Every walker's g(E) is left untouched. Intended for
            post-convergence measurement passes where the density of
            states must not change. Default ``False``.
    """

    def __init__(
        self,
        ce_path: Path | str,
        initial_atoms: Sequence[Atoms | Sequence[Atoms]],
        windows: Sequence[tuple[float | None, float | None]],
        energy_spacing: float,
        seeds: Sequence[int],
        *,
        n_walkers_per_window: int | Sequence[int] = 1,
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
        flatness_mode: FlatnessMode = "pooled",
        merge_cadence: MergeCadence = "at_halve",
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        one_over_t_gate: OneOverTGate = "visit_once",
        bp_stall_multiple: float = 4.0,
        one_over_t_entry: OneOverTEntry = "window_clock",
        frozen_measurement: bool = False,
    ) -> None:
        _check_importable(ensemble_cls, kind="ensemble_cls")
        self._frozen_measurement: bool = frozen_measurement
        _validate_flatness_mode(flatness_mode)
        _validate_merge_cadence(merge_cadence)
        _validate_one_over_t_gate(one_over_t_gate)
        _validate_gate_schedule(
            one_over_t_gate, (ensemble_kwargs or {}).get("schedule", "halving")
        )
        _validate_one_over_t_entry(one_over_t_entry)
        _validate_entry_schedule(
            one_over_t_entry, (ensemble_kwargs or {}).get("schedule", "halving")
        )
        self._one_over_t_entry: OneOverTEntry = one_over_t_entry
        self._one_over_t_gate: OneOverTGate = one_over_t_gate
        self._bp_stall_multiple: float = _validate_bp_stall_multiple(
            bp_stall_multiple
        )
        self._flatness_mode: FlatnessMode = flatness_mode
        self._merge_cadence: MergeCadence = merge_cadence
        self._recency_visits_per_bin: int = _validate_recency_visits_per_bin(
            recency_visits_per_bin
        )
        self._dos_snapshot_ratio: float | None = _validate_dos_snapshot_ratio(
            dos_snapshot_ratio
        )
        self._merge_events: list[MergeEvent] = []
        self._flatness_limit: float = float(
            (ensemble_kwargs or {}).get("flatness_limit", 0.8)
        )
        windows_list: list[tuple[float | None, float | None]] = [
            (lo, hi) for lo, hi in windows
        ]
        seeds_list = list(seeds)
        if isinstance(initial_atoms, Atoms):
            raise TypeError(
                "ProcessWangLandauPool requires a sequence of Atoms "
                "(one per window). Each window needs an initial "
                "configuration whose energy lies in that window; there "
                "is no general way to produce one from a single "
                "starting structure."
            )
        atoms_list = list(initial_atoms)
        if len(atoms_list) != len(windows_list):
            raise ValueError(
                f"initial_atoms has {len(atoms_list)} entries but "
                f"windows has {len(windows_list)}"
            )
        if len(seeds_list) != len(windows_list):
            raise ValueError("seeds and windows must be the same length")
        if isinstance(n_walkers_per_window, int):
            walkers_per_window = [int(n_walkers_per_window)] * len(windows_list)
        else:
            walkers_per_window = [int(w) for w in n_walkers_per_window]
            if len(walkers_per_window) != len(windows_list):
                raise ValueError(
                    f"n_walkers_per_window has {len(walkers_per_window)} entries "
                    f"but windows has {len(windows_list)}"
                )
        if any(w < 1 for w in walkers_per_window):
            raise ValueError(
                f"all n_walkers_per_window values must be >= 1; "
                f"got {walkers_per_window}"
            )
        self._windows: list[tuple[float | None, float | None]] = windows_list
        self._energy_spacing = float(energy_spacing)
        walker_atoms = expand_initial_structures(atoms_list, walkers_per_window)
        walker_specs = [
            [AtomsSpec.from_atoms(structure) for structure in window]
            for window in walker_atoms
        ]
        extra_kwargs: dict[str, Any] = (
            dict(ensemble_kwargs) if ensemble_kwargs else {}
        )
        self._slots: list[ProcessWangLandauWindow] = []

        # Cover both spawn-time and handshake-time failures with one
        # cleanup path. A failure on window N leaves N-1 fully started
        # slots in self._slots that shutdown() then joins.
        ctx = mp.get_context("spawn")
        try:
            for (lo, hi), window_seed, window_specs, W_w in zip(
                windows_list, seeds_list, walker_specs, walkers_per_window, strict=True,
            ):
                if W_w == 1:
                    walker_seeds = [int(window_seed)]
                    rng_seed = int(window_seed)
                else:
                    sub = np.random.SeedSequence(int(window_seed))
                    children = sub.spawn(W_w + 1)
                    walker_seeds = [int(c.generate_state(1)[0]) for c in children[:W_w]]
                    rng_seed = int(children[W_w].generate_state(1)[0])

                workers: list[tuple[mp.process.BaseProcess, Connection]] = []
                try:
                    for w_seed, atoms_spec in zip(
                        walker_seeds, window_specs, strict=True
                    ):
                        parent_conn, child_conn = ctx.Pipe(duplex=True)
                        builder = WLBuilder(
                            ce_path=str(ce_path),
                            atoms=atoms_spec,
                            energy_spacing=float(energy_spacing),
                            energy_limit_left=lo,
                            energy_limit_right=hi,
                            seed=int(w_seed),
                            ensemble_cls=ensemble_cls,
                            ensemble_kwargs=dict(extra_kwargs),
                            recency_visits_per_bin=self._recency_visits_per_bin,
                            dos_snapshot_ratio=self._dos_snapshot_ratio,
                            one_over_t_entry=self._one_over_t_entry,
                        )
                        process = ctx.Process(
                            target=_wl_worker,
                            args=(child_conn, builder),
                            daemon=True,
                        )
                        process.start()
                        child_conn.close()
                        workers.append((process, parent_conn))
                except BaseException:
                    # BaseException (not Exception): also clean up on
                    # KeyboardInterrupt during the spawn loop.
                    for proc, conn in workers:
                        conn.close()
                        proc.terminate()
                    raise
                self._slots.append(ProcessWangLandauWindow(
                    workers=workers,
                    rng=np.random.default_rng(rng_seed),
                    flatness_mode=self._flatness_mode,
                    merge_cadence=self._merge_cadence,
                    schedule=cast(
                        Schedule, extra_kwargs.get("schedule", "halving")
                    ),
                    flatness_limit=self._flatness_limit,
                    one_over_t_gate=self._one_over_t_gate,
                    bp_stall_multiple=self._bp_stall_multiple,
                ))

            for i, slot in enumerate(self._slots):
                for w, (_, conn) in enumerate(slot.workers):
                    recv_reply(conn, "STARTUP", f"window {i} walker {w}")
            self._prime_energy_cache()
        except BaseException:
            self.shutdown()
            raise
        self._ensemble_cls_fqn = (
            f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
        )
        self._ensemble_kwargs_hash = _compute_ensemble_kwargs_hash(
            ensemble_kwargs
        )

    def _check_open(self) -> None:
        if not self._slots:
            raise RuntimeError("pool is shut down")

    @property
    def is_open(self) -> bool:
        """``True`` until :meth:`shutdown` clears the worker slots."""
        return bool(self._slots)

    @property
    def ensemble_cls_fqn(self) -> str:
        """Fully-qualified name of the ensemble class used by workers."""
        return self._ensemble_cls_fqn

    @property
    def ensemble_kwargs_hash(self) -> str:
        """SHA-256 hash of the ensemble kwargs forwarded to workers."""
        return self._ensemble_kwargs_hash

    def _drain_remaining_replies(self, conns: list[Connection]) -> None:
        """Read pending replies on the given connections, ignoring contents."""
        for conn in conns:
            try:
                conn.recv()
            except (EOFError, BrokenPipeError):
                pass

    def _recv_or_abort_attach(
        self,
        conn: Connection,
        op: str,
        label: Any,
        remaining: list[Connection],
    ) -> None:
        """Receive an attach reply or abort the pool with full cleanup.

        Mirrors `ProcessPool._recv_or_abort_attach`.
        """
        try:
            recv_reply(conn, op, label)
        except (RuntimeError, TypeError):
            self._drain_remaining_replies(remaining)
            self.shutdown()
            raise

    def __len__(self) -> int:
        self._check_open()
        return len(self._slots)

    @property
    def windows(self) -> list[tuple[float | None, float | None]]:
        self._check_open()
        return list(self._windows)

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    @property
    def flatness_mode(self) -> FlatnessMode:
        """Collective-halving flatness mode this pool drives."""
        return self._flatness_mode

    @property
    def merge_cadence(self) -> MergeCadence:
        """Entropy-merge cadence this pool drives."""
        return self._merge_cadence

    @property
    def one_over_t_gate(self) -> OneOverTGate:
        """1/t-schedule halving-phase gate this pool drives."""
        return self._one_over_t_gate

    @property
    def bp_stall_multiple(self) -> float:
        """Stall-escape multiple this pool drives (consulted under flatness)."""
        return self._bp_stall_multiple

    @property
    def one_over_t_entry(self) -> OneOverTEntry:
        """1/t entry policy forwarded to every walker's replica."""
        return self._one_over_t_entry

    def advance_all(self, n_steps: int) -> None:
        # Three stages: ADVANCE (one fan-out across every worker in
        # every window), DECIDE (pure-Python coordinator decisions
        # per slot), EXECUTE (batched FORCE_HALVE / SET_ENTROPY /
        # SET_PHASE, each parallelised across slots).
        self._check_open()
        try:
            # ADVANCE: broadcast to every worker in every window in a
            # single fan-out; gather post-block state into each slot.
            all_targets: list[tuple[Connection, str]] = [
                (conn, f"window {i} walker {w}")
                for i, slot in enumerate(self._slots)
                for w, (_, conn) in enumerate(slot.workers)
            ]
            payloads = broadcast_gather(
                all_targets, ("ADVANCE", int(n_steps))
            )
            walker_addrs = [
                (slot, w)
                for slot in self._slots
                for w in range(len(slot.workers))
            ]
            for (slot, w), payload in zip(
                walker_addrs, payloads, strict=True
            ):
                slot.walker_states[w] = payload

            # Frozen mode: workers advance but the coordinator does not run.
            # No halving, entropy-merge, or phase-switch; g(E) is untouched.
            if self._frozen_measurement:
                return

            # DECIDE: per-slot coordinator decisions; no IPC.
            views = [_view_of(slot) for slot in self._slots]
            plans = [decide_block_actions(v) for v in views]

            # RECORD: capture merged entropy per halving merge while the
            # master-side plan still carries it. See wl_merge_diagnostics.
            for slot_index, (view, plan) in enumerate(
                zip(views, plans, strict=True)
            ):
                if plan.merged_entropy is not None:
                    self._merge_events.append(
                        MergeEvent(
                            slot_index=slot_index,
                            step=view.walker_states[0].step,
                            merged_entropy=dict(plan.merged_entropy),
                        )
                    )

            # EXECUTE step 1: batched FORCE_HALVE across halving slots.
            halve_targets = [
                (conn, f"window {i} walker {w}")
                for i, (slot, plan) in enumerate(zip(self._slots, plans, strict=True))
                if plan.halve
                for w, (_, conn) in enumerate(slot.workers)
            ]
            if halve_targets:
                broadcast_gather(halve_targets, ("FORCE_HALVE",))
                for slot, plan in zip(self._slots, plans, strict=True):
                    if plan.halve:
                        step = slot.walker_states[0].step
                        if slot.last_halve_step is None:
                            entries = [
                                s.window_entry_step
                                for s in slot.walker_states
                                if s.window_entry_step is not None
                            ]
                            if entries:
                                slot.first_halve_duration = step - max(entries)
                        slot.last_halve_step = step
                        slot.walker_states = [
                            replace(s, fill_factor=s.fill_factor / 2.0)
                            for s in slot.walker_states
                        ]

            # EXECUTE step 2: batched SET_ENTROPY with per-slot dicts.
            merge_targets = [
                (
                    conn,
                    f"window {i} walker {w}",
                    ("SET_ENTROPY", dict(plan.merged_entropy)),
                )
                for i, (slot, plan) in enumerate(zip(self._slots, plans, strict=True))
                if plan.merged_entropy is not None
                for w, (_, conn) in enumerate(slot.workers)
            ]
            if merge_targets:
                fanout_gather(merge_targets)
                for slot, plan in zip(self._slots, plans, strict=True):
                    if plan.merged_entropy is not None:
                        merged = dict(plan.merged_entropy)
                        slot.walker_states = [
                            replace(s, entropy=dict(merged))
                            for s in slot.walker_states
                        ]

            # EXECUTE step 3: batched SET_PHASE with per-slot phase.
            switch_targets = [
                (
                    conn,
                    f"window {i} walker {w}",
                    ("SET_PHASE", plan.switch_to_phase),
                )
                for i, (slot, plan) in enumerate(zip(self._slots, plans, strict=True))
                if plan.switch_to_phase is not None
                for w, (_, conn) in enumerate(slot.workers)
            ]
            if switch_targets:
                fanout_gather(switch_targets)
                for slot, plan in zip(self._slots, plans, strict=True):
                    if plan.switch_to_phase is not None:
                        slot.phase = plan.switch_to_phase
        except Exception:
            self.shutdown()
            raise

    def finalise_for_reporting(self) -> None:
        """End-of-run merge: fan out FINALISE_MERGE per multi-walker window.

        For each window with more than one walker: compute the merged
        entropy from ``slot.walker_states``, then send ``FINALISE_MERGE``
        to every walker in the window in a single fan-out so each
        walker's ``_entropy`` and data-container ``_last_state`` are
        updated in one round trip. Single-walker windows are skipped.
        """
        self._check_open()
        try:
            targets: list[tuple[Connection, str, tuple[Any, ...]]] = []
            per_slot_merged: list[dict[int, float] | None] = []
            for i, slot in enumerate(self._slots):
                if len(slot.workers) <= 1:
                    per_slot_merged.append(None)
                    continue
                merged = merge_entropies(slot.collect_entropy_snapshots())
                per_slot_merged.append(merged)
                for w, (_, conn) in enumerate(slot.workers):
                    targets.append((
                        conn,
                        f"window {i} walker {w}",
                        ("FINALISE_MERGE", dict(merged)),
                    ))
            if targets:
                fanout_gather(targets)
                # Refresh local snapshots so subsequent stats reads
                # also reflect the merged values.
                for slot, merged_opt in zip(
                    self._slots, per_slot_merged, strict=True,
                ):
                    if merged_opt is None:
                        continue
                    slot.walker_states = [
                        replace(s, entropy=dict(merged_opt))
                        for s in slot.walker_states
                    ]
        except Exception:
            self.shutdown()
            raise

    def current_energies(self) -> np.ndarray:
        self._check_open()
        return np.array(
            [float(slot.walker_states[0].current_energy) for slot in self._slots],
            dtype=np.float64,
        )

    def current_energy(self, i: int) -> float:
        # Live ENERGY query of window i's walker 0 (one IPC round trip),
        # so callers get an up-to-date value between block boundaries.
        self._check_open()
        return float(request(self._slots[i].workers[0][1], ("ENERGY",), i))

    def current_occupations(self, i: int) -> np.ndarray:
        self._check_open()
        return np.asarray(
            request(self._slots[i].workers[0][1], ("GET_OCC",), i)
        )

    def swap_configurations(self, i: int, j: int) -> None:
        """Swap walker 0 of window ``i`` with walker 0 of window ``j``."""
        self.apply_swaps([(i, 0, j, 0)])

    def log_g(self, i: int, energy: float) -> float:
        self._check_open()
        _, conn = self._slots[i].workers[0]
        g_at_E, _ = request(conn, ("LOG_G_AT", float(energy), float(energy)), i)
        return float(g_at_E)

    def n_walkers(self, i: int) -> int:
        """Number of walkers in window ``i``."""
        return len(self._slots[i].workers)

    def walker_energy(self, i: int, walker: int) -> float:
        """Cached current energy of ``walker`` in window ``i`` (no IPC)."""
        return float(self._slots[i].walker_states[walker].current_energy)

    def walker_log_g(self, i: int, walker: int, energy: float) -> float:
        """``ln g(E)`` for ``walker`` in window ``i`` from cached entropy (no IPC)."""
        left, right = self._windows[i]
        spacing = self._energy_spacing
        bin_left = None if left is None else int(round(left / spacing))
        bin_right = None if right is None else int(round(right / spacing))
        entropy = self._slots[i].walker_states[walker].entropy
        return log_g_at(
            entropy, energy, spacing, bin_left=bin_left, bin_right=bin_right
        )

    def candidate_pairs(
        self, i: int, j: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Random matching of windows ``i`` and ``j`` walkers for exchange.

        Each returned ``(a, b)`` pairs walker ``a`` of window ``i`` with
        walker ``b`` of window ``j``; pairs are disjoint in each
        coordinate. See :func:`mchammer_pt.exchange.matching_for_boundary`.
        """
        return matching_for_boundary(self.n_walkers(i), self.n_walkers(j), rng)

    def window_of_position(self) -> np.ndarray:
        """Window index of each ``(window, walker)`` position, in order."""
        counts = [self.n_walkers(i) for i in range(len(self._slots))]
        return np.repeat(np.arange(len(counts), dtype=np.int64), counts)

    def n_carriers(self) -> int:
        """Total number of walker positions across all windows."""
        return sum(self.n_walkers(i) for i in range(len(self._slots)))

    def apply_swaps(self, swaps: list[tuple[int, int, int, int]]) -> None:
        """Apply a batch of accepted walker-config swaps in two IPC rounds.

        ``swaps`` is a list of ``(i, a, j, b)`` meaning swap walker ``a``
        of window ``i`` with walker ``b`` of window ``j``. The controller
        guarantees the swaps are disjoint (no walker appears in two
        swaps in one cycle), so the gather/scatter target sets contain
        no aliased walkers. One ``GET_OCC`` fan-out reads every involved
        walker's occupations; one ``SET_OCC`` fan-out writes the swapped
        configurations.

        The move is not atomic across workers (the ``SET_OCC`` fan-out has
        no rollback). To avoid leaving a half-swapped pool live for a
        downstream caller, a worker IPC failure shuts the pool down and
        re-raises, matching :meth:`advance_all`.
        """
        if not swaps:
            return
        self._check_open()
        involved: list[tuple[int, int]] = []
        for i, a, j, b in swaps:
            involved.append((i, a))
            involved.append((j, b))
        if len(set(involved)) != len(involved):
            # The matching exchange guarantees disjoint swaps; an overlap
            # would silently overwrite an entry in the occupations dict
            # below and move the wrong configurations. Fail loudly instead.
            raise ValueError(
                f"apply_swaps received overlapping swaps {swaps}: a walker "
                f"appears in more than one swap. This signals a bug in the "
                f"exchange driver, not user input."
            )
        try:
            get_targets = [
                (self._slots[s].workers[w][1], f"window {s} walker {w}")
                for s, w in involved
            ]
            occ_list = broadcast_gather(get_targets, ("GET_OCC",))
            occ = {
                sw: np.asarray(o, dtype=np.int64)
                for sw, o in zip(involved, occ_list, strict=True)
            }
            set_targets = []
            for i, a, j, b in swaps:
                set_targets.append(
                    (
                        self._slots[i].workers[a][1],
                        f"window {i} walker {a}",
                        ("SET_OCC", occ[(j, b)]),
                    )
                )
                set_targets.append(
                    (
                        self._slots[j].workers[b][1],
                        f"window {j} walker {b}",
                        ("SET_OCC", occ[(i, a)]),
                    )
                )
            fanout_gather(set_targets)
        except Exception:
            self.shutdown()
            raise
        # Keep the parent-side energy cache consistent without an IPC
        # round trip: a swap moves each walker's configuration to the
        # other, so their cached current energies swap too. The entropy
        # estimate stays with the walker (only the configuration moves).
        for i, a, j, b in swaps:
            state_ia = self._slots[i].walker_states[a]
            state_jb = self._slots[j].walker_states[b]
            self._slots[i].walker_states[a] = replace(
                state_ia, current_energy=state_jb.current_energy
            )
            self._slots[j].walker_states[b] = replace(
                state_jb, current_energy=state_ia.current_energy
            )

    def swap_walker_configurations(self, i: int, a: int, j: int, b: int) -> None:
        """Swap a single walker pair (delegates to the batched path)."""
        self.apply_swaps([(i, a, j, b)])

    def converged_flags(self) -> np.ndarray:
        self._check_open()
        all_targets = [
            (conn, f"window {i} walker {w}")
            for i, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        all_flags = broadcast_gather(all_targets, ("CONVERGED",))
        result = np.empty(len(self._slots), dtype=bool)
        offset = 0
        for i, slot in enumerate(self._slots):
            result[i] = all(
                bool(all_flags[offset + w]) for w in range(len(slot.workers))
            )
            offset += len(slot.workers)
        return result

    def data_containers(self) -> list[BaseDataContainer]:
        """Flat per-walker containers in window-major / walker-minor order."""
        self._check_open()
        targets = [
            (conn, f"window {g} walker {w}")
            for g, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        return broadcast_gather(targets, ("GET_DC",))

    def per_window_stats(self) -> list[dict[str, Any]]:
        self._check_open()
        all_targets = [
            (conn, f"window {i} walker {w}")
            for i, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        all_stats = broadcast_gather(all_targets, ("WL_STATS",))
        result = []
        offset = 0
        for slot in self._slots:
            n_workers = len(slot.workers)
            slot_stats = all_stats[offset:offset + n_workers]
            offset += n_workers
            result.append(_merge_per_window_stats(slot_stats, slot._flatness_mode))
        return result

    @property
    def merge_events(self) -> tuple[MergeEvent, ...]:
        """Per-halving merged entropies; see :mod:`wl_merge_diagnostics`."""
        return tuple(self._merge_events)

    def per_window_data_containers(self) -> list[list[BaseDataContainer]]:
        """All data containers grouped by window slot.

        Returns a list of length n_windows; each entry is a list of
        ``WangLandauDataContainer`` instances, one per walker in that slot.
        """
        self._check_open()
        all_targets = [
            (conn, f"window {i} walker {w}")
            for i, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        all_dcs = broadcast_gather(all_targets, ("GET_DC",))
        result = []
        offset = 0
        for slot in self._slots:
            n_workers = len(slot.workers)
            result.append(all_dcs[offset:offset + n_workers])
            offset += n_workers
        return result

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        """Snapshot per-walker and group-level checkpoint state across workers.

        Returns:
            Dict with:
                ``"per_walker"``: list of M dicts in window-major /
                    walker-minor order, each from a worker's
                    ``SNAPSHOT_FOR_CHECKPOINT`` handler.
                ``"group_state"``: list of length N (one per window slot).
                    Dict containing ``rng_state`` and ``phase`` for W>1
                    slots (pulled from the orchestrator-side
                    :class:`ProcessWangLandauWindow`); ``None`` for W=1
                    slots.
        """
        self._check_open()
        targets = [
            (conn, f"window {g} walker {w}")
            for g, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        per_walker = broadcast_gather(targets, ("SNAPSHOT_FOR_CHECKPOINT",))
        group_state: list[dict[str, Any] | None] = []
        for slot in self._slots:
            if len(slot.workers) > 1:
                group_state.append({
                    "rng_state": _serialise_rng_state(slot.rng),
                    "phase": slot.phase,
                })
            else:
                group_state.append(None)
        return {"per_walker": per_walker, "group_state": group_state}

    def restore_replica_state(
        self,
        containers: list[BaseDataContainer],
        per_walker_extras: list[dict[str, Any]],
        group_state: list[dict[str, Any] | None],
    ) -> None:
        """Push saved per-walker and group-level state into the live pool.

        Per-walker state is fanned out to each worker over IPC; group-level
        state (rng, phase) is assigned directly to each
        :class:`ProcessWangLandauWindow` on the orchestrator side — no
        worker involvement.

        Args:
            containers: flat list of M containers in window-major /
                walker-minor order. Length must equal the total number of
                workers across all slots.
            per_walker_extras: flat list of M extras dicts, same order.
                Each must carry a ``"sites_by_species"`` key.
            group_state: list of length N (one entry per window slot).
                Must be a dict for W>1 slots — carrying ``rng_state``
                and ``phase`` — and ``None`` for W=1 slots.

        Raises:
            RuntimeError: pool is shut down, or any worker reports an
                error during restoration.
            ValueError: any input list length mismatches, or a
                ``group_state`` entry has the wrong kind for its slot.
        """
        self._check_open()
        expected_m = sum(len(slot.workers) for slot in self._slots)
        if len(containers) != expected_m:
            raise ValueError(
                f"restore_replica_state expects {expected_m} containers, "
                f"got {len(containers)}"
            )
        if len(per_walker_extras) != expected_m:
            raise ValueError(
                f"restore_replica_state expects {expected_m} extras dicts, "
                f"got {len(per_walker_extras)}"
            )
        if len(group_state) != len(self._slots):
            raise ValueError(
                f"restore_replica_state expects {len(self._slots)} "
                f"group_state entries, got {len(group_state)}"
            )

        # Phase 1: fan out RESTORE_STATE to every worker.
        offset = 0
        for slot in self._slots:
            for w, (_, conn) in enumerate(slot.workers):
                conn.send((
                    "RESTORE_STATE",
                    containers[offset + w],
                    per_walker_extras[offset + w]["sites_by_species"],
                ))
            offset += len(slot.workers)

        # Phase 2: drain acks in the same order.
        for g, slot in enumerate(self._slots):
            for w, (_, conn) in enumerate(slot.workers):
                recv_reply(conn, "RESTORE_STATE", f"window {g} walker {w}")

        # Group-level state: assign directly to ProcessWangLandauWindow.
        required_keys = {"rng_state", "phase"}
        for g, (slot, gs) in enumerate(
            zip(self._slots, group_state, strict=True)
        ):
            multi = len(slot.workers) > 1
            if gs is None:
                if multi:
                    raise ValueError(
                        "group_state entry is None for a multi-walker slot"
                    )
                continue
            if not multi:
                raise ValueError(
                    "group_state entry is non-None for a bare-replica slot"
                )
            missing = required_keys - gs.keys()
            if missing:
                raise ValueError(
                    f"window {g}: group_state missing required keys "
                    f"{sorted(missing)}; corrupted checkpoint."
                )
            slot.rng.bit_generator.state = json.loads(gs["rng_state"])
            slot.phase = gs["phase"]

        # Restored configurations change each walker's energy; refresh the
        # parent-side cache so the post-resume history snapshot is correct.
        self._prime_energy_cache()

    def _prime_energy_cache(self) -> None:
        """Fill each walker's cached ``current_energy`` from a live query.

        The per-walker energy cache (``walker_states``) is otherwise
        populated only by :meth:`advance_all`. Priming it at construction
        and after resume means :meth:`current_energies` returns the true
        initial energies for the pre-run history snapshot rather than the
        0.0 placeholder. This one-off query runs outside the per-cycle
        exchange loop, so it does not reintroduce per-exchange IPC.
        """
        targets = [
            (conn, f"window {i} walker {w}")
            for i, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        energies = broadcast_gather(targets, ("ENERGY",))
        idx = 0
        for slot in self._slots:
            for w in range(len(slot.workers)):
                slot.walker_states[w] = replace(
                    slot.walker_states[w], current_energy=float(energies[idx])
                )
                idx += 1

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an mchammer observer to selected REWL workers.

        Mirrors `ProcessPool.attach_observer`. Each selected worker
        receives its own deserialised copy via a pickle round-trip;
        the parent eagerly validates picklability before contacting
        any worker.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._slots))
        if not target_indices:
            return
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        targets: list[tuple[str, Connection]] = []
        for i in target_indices:
            for w, (_, conn) in enumerate(self._slots[i].workers):
                conn.send(("ATTACH_OBS", blob))
                targets.append((f"window {i} walker {w}", conn))
        for offset, (label, conn) in enumerate(targets):
            remaining = [c for _, c in targets[offset + 1:]]
            self._recv_or_abort_attach(conn, "ATTACH_OBS", label, remaining)

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        replicas: Sequence[int] | Literal["all"] = "all",
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to selected REWL workers.

        Mirrors `ProcessPool.attach_observer_class`. Eager
        parent-side checks: importability of ``cls``, picklability
        of ``(args, kwargs)``, and a dry-run construction that
        asserts the result is a ``BaseObserver``.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._slots))
        if not target_indices:
            return
        _check_importable(cls, kind="observer class")
        try:
            pickle.dumps((args, kwargs))
        except Exception as exc:
            raise TypeError(
                f"attach_observer_class: args/kwargs for "
                f"{cls.__name__} are not picklable ({exc})"
            ) from exc
        probe = cls(*args, **kwargs)
        if not isinstance(probe, BaseObserver):
            raise TypeError(
                f"attach_observer_class: {cls.__name__}(...) returned "
                f"{type(probe).__name__}, not a BaseObserver"
            )
        del probe
        targets: list[tuple[str, Connection]] = []
        for i in target_indices:
            for w, (_, conn) in enumerate(self._slots[i].workers):
                conn.send(("ATTACH_OBS_CLS", cls, args, kwargs))
                targets.append((f"window {i} walker {w}", conn))
        for offset, (label, conn) in enumerate(targets):
            remaining = [c for _, c in targets[offset + 1:]]
            self._recv_or_abort_attach(conn, "ATTACH_OBS_CLS", label, remaining)

    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
        *,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer constructed inside each REWL worker.

        Mirrors `ProcessPool.attach_observer_factory`. Eager
        parent-side checks: importability of ``factory`` and
        picklability of ``factory``. Worker-side construction
        failures (factory raising, or returning a non-``BaseObserver``)
        surface as ``RuntimeError`` with the worker traceback, and
        the pool shuts down so subsequent operations raise via
        ``_check_open``.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._slots))
        if not target_indices:
            return
        _check_importable(factory, kind="observer factory")
        try:
            pickle.dumps(factory)
        except Exception as exc:
            raise TypeError(
                f"attach_observer_factory: factory "
                f"{getattr(factory, '__name__', repr(factory))!r} "
                f"is not picklable ({exc})"
            ) from exc
        targets: list[tuple[str, Connection]] = []
        for i in target_indices:
            for w, (_, conn) in enumerate(self._slots[i].workers):
                conn.send(("ATTACH_OBS_FACTORY", factory))
                targets.append((f"window {i} walker {w}", conn))
        for offset, (label, conn) in enumerate(targets):
            remaining = [c for _, c in targets[offset + 1:]]
            self._recv_or_abort_attach(
                conn, "ATTACH_OBS_FACTORY", label, remaining
            )

    def record_observable(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        """Attach an observer for per-bin microcanonical moment accumulation.

        Mirrors ``attach_observer`` for the recorder path. Each selected
        worker receives its own deserialised copy of ``observer`` via a
        pickle round-trip; the worker installs it via
        ``replica.record_observable`` so the recorder is restore-aware.
        The parent eagerly validates picklability before contacting any
        worker.

        Args:
            observer: any ``mchammer.BaseObserver`` whose
                ``get_observable`` returns a scalar, sequence, or Mapping.
            replicas: ``"all"`` or an explicit sequence of window indices.

        Raises:
            TypeError: if ``observer`` is not picklable.
            RuntimeError: if the pool is shut down or a worker raises.
        """
        self._check_open()
        target_indices = _resolve_replicas(replicas, len(self._slots))
        if not target_indices:
            return
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc})"
            ) from exc
        targets: list[tuple[str, Connection]] = []
        for i in target_indices:
            for w, (_, conn) in enumerate(self._slots[i].workers):
                conn.send(("ATTACH_RECORDER", blob))
                targets.append((f"window {i} walker {w}", conn))
        for offset, (label, conn) in enumerate(targets):
            remaining = [c for _, c in targets[offset + 1:]]
            self._recv_or_abort_attach(conn, "ATTACH_RECORDER", label, remaining)

    def get_observers(self, replica_index: int) -> dict[str, BaseObserver]:
        """Return a snapshot of the observers attached to one REWL worker.

        Mirrors `ProcessPool.get_observers`. The returned dict is
        keyed by observer tag; values are independent copies via a
        worker-side pickle on send and a parent-side unpickle.

        Raises:
            IndexError: if ``replica_index`` is out of range.
            TypeError: if the observer dict cannot be round-tripped
                through pickle.
            RuntimeError: if the pool is shut down, the worker
                exited unexpectedly, or the worker reports any
                other ERR.
        """
        self._check_open()
        n = len(self._slots)
        if not 0 <= replica_index < n:
            raise IndexError(
                f"replica index {replica_index} out of range "
                f"for pool of size {n}"
            )
        _, conn = self._slots[replica_index].workers[0]
        return request(conn, ("GET_OBSERVERS",), replica_index)

    def shutdown(self) -> None:
        for slot in self._slots:
            for _, conn in slot.workers:
                try:
                    conn.send(("SHUTDOWN",))
                    conn.recv()
                except (EOFError, BrokenPipeError):
                    pass
                conn.close()
        for slot in self._slots:
            for process, _ in slot.workers:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
        self._slots = []

    def __enter__(self) -> ProcessWangLandauPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.shutdown()
