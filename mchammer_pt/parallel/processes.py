"""Multiprocessing-based replica pool.

One persistent OS process per replica. Each worker builds its own
`Replica` on startup (since `mchammer` calculators hold C++ bindings
that do not pickle) and enters a command loop. The parent communicates
with workers via per-worker duplex `Pipe`s; only integer occupation
arrays and scalar energies cross the process boundary during the run.

Worker commands:

- ``("ADVANCE", n_steps)``
- ``("ENERGY",)`` -> replies ``("OK", float)`` with total CE energy
- ``("GET_OCC",)`` -> replies ``("OK", np.ndarray)`` with occupations
- ``("SET_OCC", occupations)`` -> overwrites state
- ``("GET_DC",)`` -> replies ``("OK", BaseDataContainer)`` (pickled)
- ``("ATTACH_OBS", blob)`` -> deserialises a pickled ``BaseObserver``
  and attaches it to the replica; replies ``("OK", None)``
- ``("ATTACH_OBS_CLS", cls, args, kwargs)`` -> constructs
  ``cls(*args, **kwargs)`` locally and attaches it; replies ``("OK", None)``
- ``("ATTACH_OBS_FACTORY", factory)`` -> calls ``factory(replica)``
  locally, checks the result is a ``BaseObserver``, and attaches it;
  replies ``("OK", None)``
- ``("SHUTDOWN",)`` -> worker exits cleanly

Every reply is of the form ``(status, payload)`` with status either
``"OK"`` or ``"ERR"``; ``"ERR"`` payloads are the formatted traceback
from the worker's exception.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import traceback
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from ..replica import Replica
from ._imports import _check_importable


def _worker(
    conn: Connection,
    ce_path: str,
    atoms_dict: dict[str, Any],
    temperature: float,
    seed: int,
    ensemble_cls: type[CanonicalEnsemble],
    ensemble_kwargs: dict[str, Any],
) -> None:
    """Worker entry point: build a Replica, then serve commands.

    After successful Replica construction the worker sends a single
    ("OK", None) ready-handshake back to the parent, so the parent can
    verify startup success synchronously rather than discovering it on
    the first ADVANCE. Any exception during startup — including
    Replica construction — is caught and sent back as ("ERR", tb)
    instead, and the worker exits.
    """
    try:
        atoms = Atoms(
            numbers=atoms_dict["numbers"],
            positions=atoms_dict["positions"],
            cell=atoms_dict["cell"],
            pbc=atoms_dict["pbc"],
        )
        ce = ClusterExpansion.read(ce_path)
        replica = Replica(
            cluster_expansion=ce,
            atoms=atoms,
            temperature=temperature,
            random_seed=seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
        )
    except BaseException:
        conn.send(("ERR", traceback.format_exc()))
        conn.close()
        return

    conn.send(("OK", None))

    while True:
        try:
            cmd = conn.recv()
        except EOFError:
            return
        op = cmd[0]
        try:
            if op == "ADVANCE":
                replica.advance(cmd[1])
                conn.send(("OK", None))
            elif op == "ENERGY":
                conn.send(("OK", replica.current_energy()))
            elif op == "GET_OCC":
                conn.send(("OK", replica.current_occupations()))
            elif op == "SET_OCC":
                replica.set_occupations(cmd[1])
                conn.send(("OK", None))
            elif op == "GET_DC":
                conn.send(("OK", replica.data_container()))
            elif op == "ATTACH_OBS":
                observer = pickle.loads(cmd[1])
                replica.attach_mchammer_observer(observer)
                conn.send(("OK", None))
            elif op == "ATTACH_OBS_CLS":
                cls, args, kwargs = cmd[1], cmd[2], cmd[3]
                replica.attach_mchammer_observer(cls(*args, **kwargs))
                conn.send(("OK", None))
            elif op == "ATTACH_OBS_FACTORY":
                factory = cmd[1]
                observer = factory(replica)
                if not isinstance(observer, BaseObserver):
                    raise TypeError(
                        f"attach_observer_factory: factory returned "
                        f"{type(observer).__name__}, not a BaseObserver"
                    )
                replica.attach_mchammer_observer(observer)
                conn.send(("OK", None))
            elif op == "SHUTDOWN":
                conn.send(("OK", None))
                conn.close()
                return
            else:
                conn.send(("ERR", f"unknown command: {op!r}"))
        except Exception:
            conn.send(("ERR", traceback.format_exc()))


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
      inside each worker with that worker's ``Replica`` and reaches
      icet objects via ``replica.ensemble.calculator.cluster_expansion``.

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
        for _, conn in self._workers:
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker ADVANCE failed: {payload}")

    def current_energies(self) -> np.ndarray:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("ENERGY",))
        result = np.empty(len(self._workers), dtype=np.float64)
        for i, (_, conn) in enumerate(self._workers):
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker ENERGY failed: {payload}")
            result[i] = float(payload)
        return result

    def current_energy(self, i: int) -> float:
        self._check_open()
        _, conn = self._workers[i]
        conn.send(("ENERGY",))
        status, payload = conn.recv()
        if status != "OK":
            raise RuntimeError(f"worker ENERGY failed: {payload}")
        return float(payload)

    def current_occupations(self, i: int) -> np.ndarray:
        self._check_open()
        _, conn = self._workers[i]
        conn.send(("GET_OCC",))
        status, payload = conn.recv()
        if status != "OK":
            raise RuntimeError(f"worker GET_OCC failed: {payload}")
        return np.asarray(payload)

    def swap_configurations(self, i: int, j: int) -> None:
        self._check_open()
        # Interleaved send/recv to halve round-trip latency.
        _, conn_i = self._workers[i]
        _, conn_j = self._workers[j]
        conn_i.send(("GET_OCC",))
        conn_j.send(("GET_OCC",))
        status_i, occ_i = conn_i.recv()
        status_j, occ_j = conn_j.recv()
        if status_i != "OK":
            raise RuntimeError(f"worker GET_OCC failed: {occ_i}")
        if status_j != "OK":
            raise RuntimeError(f"worker GET_OCC failed: {occ_j}")
        conn_i.send(("SET_OCC", np.asarray(occ_j, dtype=np.int64)))
        conn_j.send(("SET_OCC", np.asarray(occ_i, dtype=np.int64)))
        status_i, payload_i = conn_i.recv()
        status_j, payload_j = conn_j.recv()
        if status_i != "OK":
            raise RuntimeError(f"worker SET_OCC failed: {payload_i}")
        if status_j != "OK":
            raise RuntimeError(f"worker SET_OCC failed: {payload_j}")

    def data_containers(self) -> list[BaseDataContainer]:
        self._check_open()
        for _, conn in self._workers:
            conn.send(("GET_DC",))
        containers: list[BaseDataContainer] = []
        for _, conn in self._workers:
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker GET_DC failed: {payload}")
            containers.append(payload)
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
        target_indices = (
            range(len(self._workers))
            if replicas == "all"
            else [int(i) for i in replicas]
        )
        if not target_indices:
            return
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        target_indices = list(target_indices)
        for i in target_indices:
            _, conn = self._workers[i]
            conn.send(("ATTACH_OBS", blob))
        for offset, i in enumerate(target_indices):
            _, conn = self._workers[i]
            status, payload = conn.recv()
            if status != "OK":
                # Partial-attach state is unrecoverable: mchammer has no detach,
                # so we shut down the pool and refuse further operations.
                # Subsequent calls raise via _check_open.
                self._drain_remaining_replies(target_indices[offset + 1:])
                self.shutdown()
                raise RuntimeError(f"worker ATTACH_OBS failed: {payload}")

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
        checks: importability of ``cls``, picklability of
        ``(args, kwargs)``, and a dry-run construction that asserts the
        result is a ``BaseObserver``. The constructor therefore fires
        ``1 + N`` times for ``N`` selected workers; it must be free of
        externally-visible side effects.
        """
        self._check_open()
        target_indices = (
            range(len(self._workers))
            if replicas == "all"
            else [int(i) for i in replicas]
        )
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
        target_indices = list(target_indices)
        for i in target_indices:
            _, conn = self._workers[i]
            conn.send(("ATTACH_OBS_CLS", cls, args, kwargs))
        for offset, i in enumerate(target_indices):
            _, conn = self._workers[i]
            status, payload = conn.recv()
            if status != "OK":
                # Partial-attach state is unrecoverable: mchammer has no detach,
                # so we shut down the pool and refuse further operations.
                # Subsequent calls raise via _check_open.
                self._drain_remaining_replies(target_indices[offset + 1:])
                self.shutdown()
                raise RuntimeError(f"worker ATTACH_OBS_CLS failed: {payload}")

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
        ``ClusterExpansion``) that do not pickle: the worker reaches
        them via ``replica.ensemble.calculator.cluster_expansion``,
        and they never cross the process boundary.

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
        target_indices = (
            range(len(self._workers))
            if replicas == "all"
            else [int(i) for i in replicas]
        )
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
        target_indices = list(target_indices)
        for i in target_indices:
            _, conn = self._workers[i]
            conn.send(("ATTACH_OBS_FACTORY", factory))
        for offset, i in enumerate(target_indices):
            _, conn = self._workers[i]
            status, payload = conn.recv()
            if status != "OK":
                # Partial-attach state is unrecoverable: mchammer has no detach,
                # so we shut down the pool and refuse further operations.
                # Subsequent calls raise via _check_open.
                self._drain_remaining_replies(target_indices[offset + 1:])
                self.shutdown()
                raise RuntimeError(f"worker ATTACH_OBS_FACTORY failed: {payload}")

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
