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
- ``("SHUTDOWN",)`` -> worker exits cleanly

Every reply is of the form ``(status, payload)`` with status either
``"OK"`` or ``"ERR"``; ``"ERR"`` payloads are the formatted traceback
from the worker's exception.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from ..replica import Replica


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
            elif op == "SHUTDOWN":
                conn.send(("OK", None))
                conn.close()
                return
            else:
                conn.send(("ERR", f"unknown command: {op!r}"))
        except BaseException:
            conn.send(("ERR", traceback.format_exc()))


def _check_ensemble_cls_importable(ensemble_cls: type) -> None:
    """Reject ``ensemble_cls`` defined in an interactive ``__main__``.

    Spawn workers re-import ``__main__`` from its file. A class defined
    at the top of ``python script.py`` works (``__main__.__file__`` is
    the script path, the worker re-imports it). A class defined in a
    Jupyter cell or REPL does not (``__main__`` has no importable
    ``__file__``). The two failure paths if we let this through are
    both unhelpful: ``PicklingError`` from the multiprocessing
    internals at parent ``process.start()``, or the worker exiting
    before ``_worker`` runs and the parent's ``recv()`` raising
    ``EOFError`` with no mention of ``ensemble_cls``.

    The heuristic is: ``__module__`` is ``"__main__"`` and
    ``sys.modules["__main__"].__file__`` is either absent or not a
    ``.py`` path. That distinguishes script callers (which work) from
    notebook/REPL callers (which don't).
    """
    if ensemble_cls.__module__ != "__main__":
        return
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file is not None and main_file.endswith(".py"):
        return
    raise ValueError(
        f"ensemble_cls={ensemble_cls.__name__!r} is defined in __main__ "
        f"in a session whose __main__ cannot be re-imported by spawn "
        f"workers (typically Jupyter or a REPL). Move the class into a "
        f".py module that both your session and the workers can import, "
        f"e.g. ``my_moves.py``, and pass it as ``my_moves.MyEnsemble``."
    )


def _atoms_to_dict(atoms: Atoms) -> dict[str, Any]:
    return {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": np.asarray(atoms.positions, dtype=np.float64),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
    }


class ProcessPool:
    """Persistent-worker multiprocessing pool.

    One OS process per replica. Satisfies `ReplicaPool`. Does NOT
    implement `ObservablePool` because `mchammer.BaseObserver`
    instances do not pickle reliably across the spawn boundary; for
    observer support use `SerialPool`.

    Args:
        ce_path: path to a CE file readable by ``ClusterExpansion.read``.
        initial_atoms: starting structure; each worker receives an
            independent copy.
        temperatures: one temperature per replica.
        seeds: one random seed per replica.
        ensemble_cls: `CanonicalEnsemble` or a subclass thereof, used by
            every worker's Replica. Spawn workers re-import the class
            by fully qualified name, so it must live in an importable
            module: top-level classes in a ``python script.py``
            invocation work (the worker re-runs the script as
            ``__main__``); classes defined in a Jupyter cell or REPL
            do not. Move such classes to a ``.py`` module file. The
            interactive-``__main__`` case is rejected up-front in
            ``__init__`` rather than producing a deep multiprocessing
            traceback.
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
        _check_ensemble_cls_importable(ensemble_cls)
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

    def __len__(self) -> int:
        return len(self._workers)

    @property
    def temperatures(self) -> list[float]:
        return list(self._temperatures)

    def advance_all(self, n_steps: int) -> None:
        for _, conn in self._workers:
            conn.send(("ADVANCE", int(n_steps)))
        for _, conn in self._workers:
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker ADVANCE failed: {payload}")

    def current_energies(self) -> np.ndarray:
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
        _, conn = self._workers[i]
        conn.send(("ENERGY",))
        status, payload = conn.recv()
        if status != "OK":
            raise RuntimeError(f"worker ENERGY failed: {payload}")
        return float(payload)

    def current_occupations(self, i: int) -> np.ndarray:
        _, conn = self._workers[i]
        conn.send(("GET_OCC",))
        status, payload = conn.recv()
        if status != "OK":
            raise RuntimeError(f"worker GET_OCC failed: {payload}")
        return np.asarray(payload)

    def swap_configurations(self, i: int, j: int) -> None:
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
        for _, conn in self._workers:
            conn.send(("GET_DC",))
        containers: list[BaseDataContainer] = []
        for _, conn in self._workers:
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker GET_DC failed: {payload}")
            containers.append(payload)
        return containers

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
