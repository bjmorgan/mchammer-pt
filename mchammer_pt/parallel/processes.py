"""Persistent-worker multiprocessing pool -- parent-side orchestration.

One OS process per walker. Workers live in spawn-mode subprocesses
implemented in ``_worker.py``; this file contains only the parent-side
``ProcessPool`` class and the per-worker ``Pipe``-based command/reply
plumbing.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import pickle
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.ensembles import (
    CanonicalEnsemble,
    WangLandauEnsemble,
)
from mchammer.observers.base_observer import (
    BaseObserver,
)

from ..checkpoint import _compute_ensemble_kwargs_hash
from ..replica import Replica
from ..wl_replica import WangLandauReplica
from ..wl_window_group import (
    _MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED,
    merge_entropies,
)
from ._comms import broadcast_gather, recv_reply, request
from ._imports import _check_importable, _resolve_replicas
from ._worker import _wl_worker, _worker


def _atoms_to_dict(atoms: Atoms) -> dict[str, Any]:
    return {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": np.asarray(atoms.positions, dtype=np.float64),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
    }


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
        seeds: one random seed per window.
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
            atoms_dicts = [_atoms_to_dict(initial_atoms)] * len(temperatures_list)
        else:
            atoms_list = list(initial_atoms)
            if len(atoms_list) != len(temperatures_list):
                raise ValueError(
                    f"initial_atoms has {len(atoms_list)} entries but "
                    f"temperatures has {len(temperatures_list)}"
                )
            atoms_dicts = [_atoms_to_dict(a) for a in atoms_list]
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
            for T, seed, ad in zip(
                self._temperatures, seeds_list, atoms_dicts, strict=True
            ):
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                process = ctx.Process(
                    target=_worker,
                    args=(
                        child_conn,
                        str(ce_path),
                        ad,
                        T,
                        int(seed),
                        ensemble_cls,
                        extra_kwargs,
                    ),
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
            containers: one container per slot, in window order.
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


@dataclasses.dataclass
class _WindowSlot:
    """Per-window state for ``ProcessWangLandauPool``."""

    workers: list[tuple[mp.process.BaseProcess, Connection]]
    exchange_idx: int
    rng: np.random.Generator

    def exchange_conn(self) -> Connection:
        """Connection to the current exchange-representative walker."""
        return self.workers[self.exchange_idx][1]


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
        initial_atoms: one starting structure per window. Single-Atoms
            broadcast is not supported (every window needs an initial
            configuration whose energy lies in that window).
        windows: per-replica energy windows.
        energy_spacing: bin size shared across replicas.
        seeds: one random seed per window.
        n_walkers_per_window: walkers per energy window; > 1 enables
            multi-walker entropy sync after each block.
        ensemble_cls: WL ensemble class. Defaults to
            `WangLandauEnsemble`. To use the 1/t schedule, pass
            ``ensemble_kwargs={'schedule': '1_over_t'}``. Spawned
            workers re-import by FQN; interactive-``__main__``
            classes are not supported.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.
            Must be picklable for the spawn boundary.
    """

    def __init__(
        self,
        ce_path: Path | str,
        initial_atoms: Sequence[Atoms],
        windows: Sequence[tuple[float | None, float | None]],
        energy_spacing: float,
        seeds: Sequence[int],
        *,
        n_walkers_per_window: int | Sequence[int] = 1,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        _check_importable(ensemble_cls, kind="ensemble_cls")
        windows_list: list[tuple[float | None, float | None]] = [
            (lo, hi) for lo, hi in windows
        ]
        seeds_list = list(seeds)
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
        atoms_dicts = [_atoms_to_dict(a) for a in atoms_list]
        extra_kwargs: dict[str, Any] = (
            dict(ensemble_kwargs) if ensemble_kwargs else {}
        )
        self._slots: list[_WindowSlot] = []

        ctx = mp.get_context("spawn")
        try:
            for (lo, hi), window_seed, ad, W_w in zip(
                windows_list, seeds_list, atoms_dicts, walkers_per_window, strict=True,
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
                for w_seed in walker_seeds:
                    parent_conn, child_conn = ctx.Pipe(duplex=True)
                    process = ctx.Process(
                        target=_wl_worker,
                        args=(
                            child_conn,
                            str(ce_path),
                            ad,
                            float(energy_spacing),
                            lo,
                            hi,
                            int(w_seed),
                            ensemble_cls,
                            extra_kwargs,
                        ),
                        daemon=True,
                    )
                    process.start()
                    child_conn.close()
                    workers.append((process, parent_conn))
                self._slots.append(_WindowSlot(
                    workers=workers,
                    exchange_idx=0,
                    rng=np.random.default_rng(rng_seed),
                ))

            for i, slot in enumerate(self._slots):
                for w, (_, conn) in enumerate(slot.workers):
                    recv_reply(conn, "STARTUP", f"window {i} walker {w}")
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

    def advance_all(self, n_steps: int) -> None:
        self._check_open()
        all_targets = [
            (conn, f"window {i} walker {w}")
            for i, slot in enumerate(self._slots)
            for w, (_, conn) in enumerate(slot.workers)
        ]
        broadcast_gather(all_targets, ("ADVANCE", int(n_steps)))

        try:
            for i, slot in enumerate(self._slots):
                if len(slot.workers) == 1:
                    continue
                slot_targets = [
                    (conn, f"window {i} walker {w}")
                    for w, (_, conn) in enumerate(slot.workers)
                ]
                states = broadcast_gather(
                    slot_targets, ("GET_ENTROPY_SYNC_STATE",)
                )
                target_len = max(
                    s["fill_factor_history_len"] for s in states
                )
                merged = merge_entropies(
                    [s["entropy"] for s in states]
                )
                for w, (_, conn) in enumerate(slot.workers):
                    extra = target_len - states[w]["fill_factor_history_len"]
                    conn.send(("APPLY_ENTROPY_SYNC", merged, extra))
                for w, (_, conn) in enumerate(slot.workers):
                    recv_reply(
                        conn, "APPLY_ENTROPY_SYNC",
                        f"window {i} walker {w}",
                    )
                slot.exchange_idx = int(
                    slot.rng.integers(0, len(slot.workers))
                )
        except Exception:
            self.shutdown()
            raise

    def current_energies(self) -> np.ndarray:
        self._check_open()
        targets = [
            (slot.exchange_conn(), i)
            for i, slot in enumerate(self._slots)
        ]
        payloads = broadcast_gather(targets, ("ENERGY",))
        return np.array(payloads, dtype=np.float64)

    def current_energy(self, i: int) -> float:
        self._check_open()
        return float(request(self._slots[i].exchange_conn(), ("ENERGY",), i))

    def current_occupations(self, i: int) -> np.ndarray:
        self._check_open()
        return np.asarray(request(self._slots[i].exchange_conn(), ("GET_OCC",), i))

    def swap_configurations(self, i: int, j: int) -> None:
        self._check_open()
        conn_i = self._slots[i].exchange_conn()
        conn_j = self._slots[j].exchange_conn()
        occ_i, occ_j = broadcast_gather(
            [(conn_i, i), (conn_j, j)], ("GET_OCC",)
        )
        request(conn_i, ("SET_OCC", np.asarray(occ_j, dtype=np.int64)), i)
        try:
            request(conn_j, ("SET_OCC", np.asarray(occ_i, dtype=np.int64)), j)
        except BaseException:
            request(conn_i, ("SET_OCC", np.asarray(occ_i, dtype=np.int64)), i)
            raise

    def log_g(self, i: int, energy: float) -> float:
        self._check_open()
        _, conn = self._slots[i].workers[0]
        g_at_E, _ = request(conn, ("LOG_G_AT", float(energy), float(energy)), i)
        return float(g_at_E)

    def log_g_pair(
        self, i: int, j: int, E_i: float, E_j: float,
    ) -> tuple[float, float, float, float]:
        self._check_open()
        _, conn_i = self._slots[i].workers[0]
        _, conn_j = self._slots[j].workers[0]
        (g_i_Ei, g_i_Ej), (g_j_Ei, g_j_Ej) = broadcast_gather(
            [(conn_i, i), (conn_j, j)],
            ("LOG_G_AT", float(E_i), float(E_j)),
        )
        return (
            float(g_i_Ei), float(g_i_Ej), float(g_j_Ei), float(g_j_Ej),
        )

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
        self._check_open()
        targets = [
            (slot.workers[0][1], i) for i, slot in enumerate(self._slots)
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
            if n_workers == 1:
                result.append(slot_stats[0])
            else:
                combined_hist: dict[int, int] = {}
                for s in slot_stats:
                    for k, v in s["histogram"].items():
                        combined_hist[k] = combined_hist.get(k, 0) + v
                result.append({
                    "fill_factor": slot_stats[0]["fill_factor"],
                    "halvings": slot_stats[0]["halvings"],
                    "histogram": combined_hist,
                    "converged": all(s["converged"] for s in slot_stats),
                })
        return result

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

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]]:
        self._check_open()
        if any(len(slot.workers) > 1 for slot in self._slots):
            raise NotImplementedError(_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED)
        targets = [
            (slot.workers[0][1], i) for i, slot in enumerate(self._slots)
        ]
        return broadcast_gather(targets, ("SNAPSHOT_FOR_CHECKPOINT",))

    def restore_replica_state(
        self,
        containers: list[BaseDataContainer],
        replica_extras: list[dict[str, Any]],
    ) -> None:
        """Push saved per-replica state into each worker.

        Sends each worker its corresponding `WangLandauDataContainer` and
        the ``sites_by_species`` cache from ``replica_extras``; the worker
        applies them via `WangLandauReplica.restore_state`. Used by
        `WangLandauParallelTempering.resume_process_pool` to bring a
        freshly-spawned process pool to the saved state.

        Args:
            containers: one container per worker, in slot order.
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
        for slot, container, extra in zip(
            self._slots, containers, replica_extras, strict=True
        ):
            _, conn = slot.workers[0]
            conn.send(
                ("RESTORE_STATE", container, extra["sites_by_species"])
            )
        for i, slot in enumerate(self._slots):
            _, conn = slot.workers[0]
            recv_reply(conn, "RESTORE_STATE", i)

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
