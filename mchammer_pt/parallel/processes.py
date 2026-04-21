"""Multiprocessing-based replica advance.

One persistent OS process per replica. The worker builds its own
`Replica` on startup (since `mchammer` calculators hold C++ bindings
that do not pickle) and enters a command loop. The orchestrator
communicates with workers via per-worker duplex `Pipe`s.

Commands:

- ``("ADVANCE", n_steps)``: run N trial steps; reply ``("OK", None)``.
- ``("GET_OCC",)``: reply ``("OK", occupations)``.
- ``("SET_OCC", occupations)``: overwrite state; reply ``("OK", None)``.
- ``("GET_ENERGY",)``: reply ``("OK", total_energy)``.
- ``("SHUTDOWN",)``: break the loop; worker exits cleanly.

Only integer occupation arrays cross the process boundary during a
run; the CE itself is passed once as a file path at worker startup.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]

from ..replica import Replica


def _worker(
    conn: mp.connection.Connection,
    ce_path: str,
    atoms_dict: dict[str, Any],
    temperature: float,
    seed: int,
) -> None:
    """Worker entry point: build a Replica, then serve commands."""
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
    )
    while True:
        cmd = conn.recv()
        op = cmd[0]
        if op == "ADVANCE":
            replica.advance(cmd[1])
            conn.send(("OK", None))
        elif op == "GET_OCC":
            conn.send(("OK", replica.current_occupations()))
        elif op == "SET_OCC":
            replica.set_occupations(cmd[1])
            conn.send(("OK", None))
        elif op == "GET_ENERGY":
            conn.send(("OK", replica.current_energy()))
        elif op == "SHUTDOWN":
            conn.send(("OK", None))
            conn.close()
            return
        else:
            conn.send(("ERR", f"unknown command: {op!r}"))


def _atoms_to_dict(atoms: Atoms) -> dict[str, Any]:
    return {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": np.asarray(atoms.positions, dtype=np.float64),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
    }


class ProcessBackend:
    """Persistent-worker parallel backend.

    Args:
        ce_path: path to a CE file readable by ``ClusterExpansion.read``.
            The caller is responsible for writing one if starting from
            an in-memory CE.
        initial_atoms: starting structure; each worker receives an
            independent copy.
        temperatures: one temperature per replica.
        seeds: one random seed per replica.
    """

    def __init__(
        self,
        ce_path: Path | str,
        initial_atoms: Atoms,
        temperatures: list[float],
        seeds: list[int],
    ) -> None:
        if len(temperatures) != len(seeds):
            raise ValueError("temperatures and seeds must be the same length")
        self._workers: list[
            tuple[mp.process.BaseProcess, mp.connection.Connection]
        ] = []
        atoms_dict = _atoms_to_dict(initial_atoms)
        ctx = mp.get_context("spawn")
        for T, seed in zip(temperatures, seeds, strict=True):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            process = ctx.Process(
                target=_worker,
                args=(child_conn, str(ce_path), atoms_dict, float(T), int(seed)),
                daemon=True,
            )
            process.start()
            child_conn.close()  # the parent does not need the child end
            self._workers.append((process, parent_conn))

    def advance_all(self, replicas: list[Replica], n_steps: int) -> None:
        """Tell every worker to advance `n_steps`, then wait for all.

        The `replicas` argument is accepted for interface symmetry
        with `SerialBackend` but is not used: the worker processes
        own their own Replica state. Callers that need the in-process
        Replicas to reflect the worker state should call
        ``current_occupations()`` and then ``replica.set_occupations``
        on each.
        """
        for _, conn in self._workers:
            conn.send(("ADVANCE", int(n_steps)))
        for _, conn in self._workers:
            status, _ = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker ADVANCE failed: {status}")

    def current_occupations(self) -> list[np.ndarray]:
        """Fetch the current occupation vectors from every worker."""
        result: list[np.ndarray] = []
        for _, conn in self._workers:
            conn.send(("GET_OCC",))
        for _, conn in self._workers:
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker GET_OCC failed: {status}")
            result.append(payload)
        return result

    def current_energies(self) -> list[float]:
        """Fetch the current total energies from every worker."""
        result: list[float] = []
        for _, conn in self._workers:
            conn.send(("GET_ENERGY",))
        for _, conn in self._workers:
            status, payload = conn.recv()
            if status != "OK":
                raise RuntimeError(f"worker GET_ENERGY failed: {status}")
            result.append(float(payload))
        return result

    def set_occupations(self, i: int, occupations: np.ndarray) -> None:
        """Overwrite the state of the worker at replica index ``i``."""
        _, conn = self._workers[i]
        conn.send(("SET_OCC", np.asarray(occupations, dtype=np.int64)))
        status, _ = conn.recv()
        if status != "OK":
            raise RuntimeError(f"worker SET_OCC failed: {status}")

    def shutdown(self) -> None:
        """Ask all workers to exit and clear the worker list."""
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
