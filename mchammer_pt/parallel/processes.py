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
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from ..replica import Replica
from ._imports import _check_importable, _resolve_replicas
from ._worker import _worker


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
        initial_atoms: starting structure; each worker receives an
            independent copy.
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
        initial_atoms: Atoms,
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
        atoms_dict = _atoms_to_dict(initial_atoms)
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
            for T, seed in zip(self._temperatures, seeds_list, strict=True):
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                process = ctx.Process(
                    target=_worker,
                    args=(
                        child_conn,
                        str(ce_path),
                        atoms_dict,
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

    def _check_open(self) -> None:
        if not self._workers:
            raise RuntimeError("pool is shut down")

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
        # Interleaved send/recv to halve round-trip latency.
        _, conn_i = self._workers[i]
        _, conn_j = self._workers[j]
        conn_i.send(("GET_OCC",))
        conn_j.send(("GET_OCC",))
        occ_i = self._recv_or_raise(conn_i, "GET_OCC", i)
        occ_j = self._recv_or_raise(conn_j, "GET_OCC", j)
        conn_i.send(("SET_OCC", np.asarray(occ_j, dtype=np.int64)))
        conn_j.send(("SET_OCC", np.asarray(occ_i, dtype=np.int64)))
        self._recv_or_raise(conn_i, "SET_OCC", i)
        self._recv_or_raise(conn_j, "SET_OCC", j)

    def data_containers(self) -> list[BaseDataContainer]:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("GET_DC",))
        containers: list[BaseDataContainer] = []
        for i, (_, conn) in enumerate(self._workers):
            containers.append(self._recv_or_raise(conn, "GET_DC", i))
        return containers

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
