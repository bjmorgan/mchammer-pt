"""Persistent-worker multiprocessing pool -- parent-side orchestration.

One OS process per replica. Workers live in spawn-mode subprocesses
implemented in ``_worker.py``; this file contains only the parent-side
``ProcessPool`` class and the per-worker ``Pipe``-based command/reply
plumbing.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal, NoReturn

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
from ._imports import _check_importable, _resolve_replicas
from ._worker import _wl_worker, _worker

_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED = (
    "checkpointing is not yet supported for n_walkers_per_window > 1; "
    "pass data_container_file=None and avoid save_checkpoint() / "
    "attach_checkpoint_writer() when using multiple walkers per window."
)


def _merge_entropies(entropies: list[dict[int, float]]) -> dict[int, float]:
    """Average bin-wise entropy estimates across multiple walkers.

    Args:
        entropies: list of {bin_index: entropy_value} dicts from each walker.

    Returns:
        Merged entropy dict with bin-wise averages; missing bins contribute 0.0.
        Unvisited bins are deliberately suppressed: frontier regions entered by
        only a subset of walkers contribute a reduced entropy estimate until all
        walkers reach them.
    """
    all_bins: set[int] = set()
    for e in entropies:
        all_bins.update(e.keys())
    n = len(entropies)
    return {b: sum(e.get(b, 0.0) for e in entropies) / n for b in all_bins}


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
            for _, conn in self._workers:
                status, payload = conn.recv()
                if status != "OK":
                    raise RuntimeError(f"worker startup failed:\n{payload}")
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

    def _abort_partial_attach(
        self,
        op: str,
        payload: str,
        remaining: list[int],
    ) -> NoReturn:
        """Shut the pool down after a worker reports ERR during attach.

        Partial-attach state is unrecoverable — mchammer has no detach
        API — so the pool is shut down and further operations refuse
        via _check_open. The drain step prevents the SHUTDOWN handshake
        from racing against unread attach replies.
        """
        self._drain_remaining_replies(remaining)
        self.shutdown()
        raise RuntimeError(f"worker {op} failed: {payload}")

    def _recv_or_abort_attach(
        self,
        conn: Connection,
        op: str,
        i: int,
        remaining: list[int],
    ) -> None:
        """Receive an attach reply or abort the pool with full cleanup.

        Three failure paths produce the same outcome — drain pending
        replies on later workers, shut the pool down, raise a framed
        RuntimeError:

        - Worker reports ERR (factory raised, isinstance check
          failed, etc.) — message includes the worker traceback.
        - Pipe closes (worker died via Ctrl-C, OOM, segfault) —
          message says the worker exited unexpectedly during attach.

        After this returns the pool is guaranteed shut down and
        further operations refuse via _check_open.
        """
        try:
            status, payload = conn.recv()
        except EOFError as exc:
            self._drain_remaining_replies(remaining)
            self.shutdown()
            raise RuntimeError(
                f"worker {op} (replica i={i}) exited unexpectedly during attach"
            ) from exc
        if status != "OK":
            self._abort_partial_attach(op, payload, remaining)

    def _recv_or_raise(self, conn: Connection, op: str, i: int) -> Any:
        """Receive a (status, payload) reply or raise a clear exception.

        Three reply shapes are recognised:

        - ``("OK", payload)`` — return ``payload``.
        - ``("ERR_PICKLE", traceback)`` — the worker's reply payload
          could not be pickled (e.g. an attached observer accumulated
          a non-picklable attribute). Raise ``TypeError`` so callers
          see the same exception type as the parent-side eager pickle
          checks on the attach paths.
        - ``("ERR", traceback)`` — any other worker-side failure.
          Raise ``RuntimeError`` carrying the worker traceback.

        A pipe-closed ``EOFError`` (worker died, e.g. via
        ``KeyboardInterrupt``) is translated into a framed
        ``RuntimeError`` so the parent never sees a bare
        ``EOFError`` from the recv path.
        """
        try:
            status, payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"worker {op} (replica i={i}) exited unexpectedly"
            ) from exc
        if status == "ERR_PICKLE":
            raise TypeError(
                f"reply from worker {op} (replica i={i}) could not be "
                f"round-tripped through pickle: {payload}"
            )
        if status != "OK":
            raise RuntimeError(f"worker {op} (replica i={i}) failed: {payload}")
        return payload

    def __len__(self) -> int:
        self._check_open()
        return len(self._workers)

    @property
    def temperatures(self) -> list[float]:
        self._check_open()
        return list(self._temperatures)

    def advance_all(self, n_steps: int) -> None:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("ADVANCE", int(n_steps)))
        for i, (_, conn) in enumerate(self._workers):
            self._recv_or_raise(conn, "ADVANCE", i)

    def current_energies(self) -> np.ndarray:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("ENERGY",))
        result = np.empty(len(self._workers), dtype=np.float64)
        for i, (_, conn) in enumerate(self._workers):
            result[i] = float(self._recv_or_raise(conn, "ENERGY", i))
        return result

    def current_energy(self, i: int) -> float:
        self._check_open()
        _, conn = self._workers[i]
        conn.send(("ENERGY",))
        return float(self._recv_or_raise(conn, "ENERGY", i))

    def current_occupations(self, i: int) -> np.ndarray:
        self._check_open()
        _, conn = self._workers[i]
        conn.send(("GET_OCC",))
        return np.asarray(self._recv_or_raise(conn, "GET_OCC", i))

    def swap_configurations(self, i: int, j: int) -> None:
        self._check_open()
        _, conn_i = self._workers[i]
        _, conn_j = self._workers[j]
        conn_i.send(("GET_OCC",))
        conn_j.send(("GET_OCC",))
        occ_i = self._recv_or_raise(conn_i, "GET_OCC", i)
        occ_j = self._recv_or_raise(conn_j, "GET_OCC", j)
        conn_i.send(("SET_OCC", np.asarray(occ_j, dtype=np.int64)))
        self._recv_or_raise(conn_i, "SET_OCC", i)
        conn_j.send(("SET_OCC", np.asarray(occ_i, dtype=np.int64)))
        try:
            self._recv_or_raise(conn_j, "SET_OCC", j)
        except BaseException:
            conn_i.send(("SET_OCC", np.asarray(occ_i, dtype=np.int64)))
            self._recv_or_raise(conn_i, "SET_OCC", i)
            raise

    def data_containers(self) -> list[BaseDataContainer]:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("GET_DC",))
        containers: list[BaseDataContainer] = []
        for i, (_, conn) in enumerate(self._workers):
            containers.append(self._recv_or_raise(conn, "GET_DC", i))
        return containers

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]]:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("SNAPSHOT_FOR_CHECKPOINT",))
        extras: list[dict[str, Any]] = []
        for i, (_, conn) in enumerate(self._workers):
            extras.append(
                self._recv_or_raise(conn, "SNAPSHOT_FOR_CHECKPOINT", i)
            )
        return extras

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
            self._recv_or_raise(conn, "RESTORE_STATE", i)

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
        conn.send(("GET_OBSERVERS",))
        return self._recv_or_raise(conn, "GET_OBSERVERS", replica_index)

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


class ProcessWangLandauPool:
    """Persistent-worker REWL pool.

    One OS process per replica. Satisfies `WangLandauObservablePool`:
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
        seeds: one random seed per replica.
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
        self._slots: list[list[tuple[mp.process.BaseProcess, Connection]]] = []
        self._exchange_idx: list[int] = [0] * len(windows_list)
        self._slot_rngs: list[np.random.Generator] = []

        ctx = mp.get_context("spawn")
        try:
            for (lo, hi), window_seed, ad, W_w in zip(
                windows_list, seeds_list, atoms_dicts, walkers_per_window, strict=True,
            ):
                if W_w == 1:
                    walker_seeds = [int(window_seed)]
                    rng_seed = int(window_seed)  # RNG is never called for W=1
                else:
                    sub = np.random.SeedSequence(int(window_seed))
                    children = sub.spawn(W_w + 1)
                    walker_seeds = [int(c.generate_state(1)[0]) for c in children[:W_w]]
                    rng_seed = int(children[W_w].generate_state(1)[0])
                self._slot_rngs.append(np.random.default_rng(rng_seed))

                slot: list[tuple[mp.process.BaseProcess, Connection]] = []
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
                    slot.append((process, parent_conn))
                self._slots.append(slot)

            for slot in self._slots:
                for _, conn in slot:
                    status, payload = conn.recv()
                    if status != "OK":
                        raise RuntimeError(f"worker startup failed:\n{payload}")
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

    def _recv_or_raise(self, conn: Connection, op: str, label: Any) -> Any:
        """Receive a (status, payload) reply or raise a clear exception.

        Mirrors `ProcessPool._recv_or_raise`; duplicated here for
        clarity since `ProcessWangLandauPool` is self-contained. A
        future refactor extracting the parent-side pipe plumbing
        would consolidate this and the abort/drain helpers across
        both pool classes.
        """
        try:
            status, payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"worker {op} ({label}) exited unexpectedly"
            ) from exc
        if status == "ERR_PICKLE":
            raise TypeError(
                f"reply from worker {op} ({label}) could not be "
                f"round-tripped through pickle: {payload}"
            )
        if status != "OK":
            raise RuntimeError(f"worker {op} ({label}) failed: {payload}")
        return payload

    def _drain_remaining_replies(self, conns: list[Connection]) -> None:
        """Read pending replies on the given connections, ignoring contents."""
        for conn in conns:
            try:
                conn.recv()
            except (EOFError, BrokenPipeError):
                pass

    def _abort_partial_attach(
        self,
        op: str,
        payload: str,
        remaining: list[Connection],
    ) -> NoReturn:
        """Shut the pool down after a worker reports ERR during attach.

        Mirrors `ProcessPool._abort_partial_attach`. Partial-attach
        state is unrecoverable — mchammer has no detach API — so the
        pool is shut down and further operations refuse via
        `_check_open`. The drain step prevents the SHUTDOWN handshake
        from racing against unread attach replies.
        """
        self._drain_remaining_replies(remaining)
        self.shutdown()
        raise RuntimeError(f"worker {op} failed: {payload}")

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
            status, payload = conn.recv()
        except EOFError as exc:
            self._drain_remaining_replies(remaining)
            self.shutdown()
            raise RuntimeError(
                f"worker {op} ({label}) exited unexpectedly during attach"
            ) from exc
        if status != "OK":
            self._abort_partial_attach(op, payload, remaining)

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
        for slot in self._slots:
            for _, conn in slot:
                conn.send(("ADVANCE", int(n_steps)))
        for i, slot in enumerate(self._slots):
            for w, (_, conn) in enumerate(slot):
                self._recv_or_raise(conn, "ADVANCE", f"window {i} walker {w}")

        # Phase 2: entropy sync for multi-walker slots
        try:
            for i, slot in enumerate(self._slots):
                if len(slot) == 1:
                    continue
                for _, conn in slot:
                    conn.send(("GET_ENTROPY_SYNC_STATE",))
                states = [
                    self._recv_or_raise(
                        conn, "GET_ENTROPY_SYNC_STATE", f"window {i} walker {w}"
                    )
                    for w, (_, conn) in enumerate(slot)
                ]
                target_len = max(s["fill_factor_history_len"] for s in states)
                merged = _merge_entropies([s["entropy"] for s in states])
                for w, (_, conn) in enumerate(slot):
                    extra = target_len - states[w]["fill_factor_history_len"]
                    conn.send(("APPLY_ENTROPY_SYNC", merged, extra))
                for w, (_, conn) in enumerate(slot):
                    self._recv_or_raise(
                        conn, "APPLY_ENTROPY_SYNC", f"window {i} walker {w}"
                    )
                self._exchange_idx[i] = int(self._slot_rngs[i].integers(0, len(slot)))
        except Exception:
            self.shutdown()
            raise

    def current_energies(self) -> np.ndarray:
        self._check_open()
        for i, slot in enumerate(self._slots):
            _, conn = slot[self._exchange_idx[i]]
            conn.send(("ENERGY",))
        result = np.empty(len(self._slots), dtype=np.float64)
        for i, slot in enumerate(self._slots):
            _, conn = slot[self._exchange_idx[i]]
            result[i] = float(self._recv_or_raise(conn, "ENERGY", i))
        return result

    def current_energy(self, i: int) -> float:
        self._check_open()
        _, conn = self._slots[i][self._exchange_idx[i]]
        conn.send(("ENERGY",))
        return float(self._recv_or_raise(conn, "ENERGY", i))

    def current_occupations(self, i: int) -> np.ndarray:
        self._check_open()
        _, conn = self._slots[i][self._exchange_idx[i]]
        conn.send(("GET_OCC",))
        return np.asarray(self._recv_or_raise(conn, "GET_OCC", i))

    def swap_configurations(self, i: int, j: int) -> None:
        self._check_open()
        _, conn_i = self._slots[i][self._exchange_idx[i]]
        _, conn_j = self._slots[j][self._exchange_idx[j]]
        conn_i.send(("GET_OCC",))
        conn_j.send(("GET_OCC",))
        occ_i = self._recv_or_raise(conn_i, "GET_OCC", i)
        occ_j = self._recv_or_raise(conn_j, "GET_OCC", j)
        conn_i.send(("SET_OCC", np.asarray(occ_j, dtype=np.int64)))
        self._recv_or_raise(conn_i, "SET_OCC", i)
        conn_j.send(("SET_OCC", np.asarray(occ_i, dtype=np.int64)))
        try:
            self._recv_or_raise(conn_j, "SET_OCC", j)
        except BaseException:
            conn_i.send(("SET_OCC", np.asarray(occ_i, dtype=np.int64)))
            self._recv_or_raise(conn_i, "SET_OCC", i)
            raise

    def log_g(self, i: int, energy: float) -> float:
        self._check_open()
        # walker 0; all walkers' entropies are equal after advance_all returns
        _, conn = self._slots[i][0]
        conn.send(("LOG_G_AT", float(energy), float(energy)))
        g_at_E, _ = self._recv_or_raise(conn, "LOG_G_AT", i)
        return float(g_at_E)

    def log_g_pair(
        self, i: int, j: int, E_i: float, E_j: float,
    ) -> tuple[float, float, float, float]:
        self._check_open()
        # walker 0; all walkers' entropies are equal after advance_all returns
        _, conn_i = self._slots[i][0]
        _, conn_j = self._slots[j][0]
        conn_i.send(("LOG_G_AT", float(E_i), float(E_j)))
        conn_j.send(("LOG_G_AT", float(E_i), float(E_j)))
        g_i_Ei, g_i_Ej = self._recv_or_raise(conn_i, "LOG_G_AT", i)
        g_j_Ei, g_j_Ej = self._recv_or_raise(conn_j, "LOG_G_AT", j)
        return (
            float(g_i_Ei), float(g_i_Ej), float(g_j_Ei), float(g_j_Ej),
        )

    def converged_flags(self) -> np.ndarray:
        self._check_open()
        for slot in self._slots:
            for _, conn in slot:
                conn.send(("CONVERGED",))
        result = np.empty(len(self._slots), dtype=bool)
        for i, slot in enumerate(self._slots):
            result[i] = all(
                bool(self._recv_or_raise(conn, "CONVERGED", f"window {i} walker {w}"))
                for w, (_, conn) in enumerate(slot)
            )
        return result

    def data_containers(self) -> list[BaseDataContainer]:
        self._check_open()
        for slot in self._slots:
            _, conn = slot[0]
            conn.send(("GET_DC",))
        result = []
        for i, slot in enumerate(self._slots):
            _, conn = slot[0]
            result.append(self._recv_or_raise(conn, "GET_DC", i))
        return result

    def per_window_stats(self) -> list[dict[str, Any]]:
        self._check_open()
        for slot in self._slots:
            for _, conn in slot:
                conn.send(("WL_STATS",))
        result = []
        for i, slot in enumerate(self._slots):
            slot_stats = [
                self._recv_or_raise(conn, "WL_STATS", f"window {i} walker {w}")
                for w, (_, conn) in enumerate(slot)
            ]
            if len(slot_stats) == 1:
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
        for slot in self._slots:
            for _, conn in slot:
                conn.send(("GET_DC",))
        return [
            [
                self._recv_or_raise(conn, "GET_DC", f"window {i} walker {w}")
                for w, (_, conn) in enumerate(slot)
            ]
            for i, slot in enumerate(self._slots)
        ]

    def snapshot_for_checkpoint(self) -> list[dict[str, Any]]:
        self._check_open()
        if any(len(slot) > 1 for slot in self._slots):
            raise NotImplementedError(_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED)
        for slot in self._slots:
            _, conn = slot[0]
            conn.send(("SNAPSHOT_FOR_CHECKPOINT",))
        result = []
        for i, slot in enumerate(self._slots):
            _, conn = slot[0]
            result.append(
                self._recv_or_raise(conn, "SNAPSHOT_FOR_CHECKPOINT", i)
            )
        return result

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
            _, conn = slot[0]
            conn.send(("RESTORE_STATE", container, extra["sites_by_species"]))
        for i, slot in enumerate(self._slots):
            _, conn = slot[0]
            self._recv_or_raise(conn, "RESTORE_STATE", i)

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
            for w, (_, conn) in enumerate(self._slots[i]):
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
            for w, (_, conn) in enumerate(self._slots[i]):
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
            for w, (_, conn) in enumerate(self._slots[i]):
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
        _, conn = self._slots[replica_index][0]
        conn.send(("GET_OBSERVERS",))
        return self._recv_or_raise(conn, "GET_OBSERVERS", replica_index)

    def shutdown(self) -> None:
        for slot in self._slots:
            for _, conn in slot:
                try:
                    conn.send(("SHUTDOWN",))
                    conn.recv()
                except (EOFError, BrokenPipeError):
                    pass
                conn.close()
        for slot in self._slots:
            for process, _ in slot:
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
