"""Worker-side implementation of the persistent multiprocessing pool.

Each worker process starts by building a `Replica`, sending a single
("OK", None) ready-handshake to the parent, then entering the
following command loop:

- ``("ADVANCE", n_steps)`` -> replies ``("OK", None)`` after the run
- ``("ENERGY",)`` -> replies ``("OK", float)`` with total CE energy
- ``("GET_OCC",)`` -> replies ``("OK", np.ndarray)`` with occupations
- ``("SET_OCC", occupations)`` -> overwrites state
- ``("GET_DC",)`` -> replies ``("OK", BaseDataContainer)`` (pickled)
- ``("ATTACH_OBS", pickled_blob)`` -> deserialises and attaches an
  observer; replies ``("OK", None)``
- ``("ATTACH_OBS_CLS", cls, args, kwargs)`` -> constructs
  ``cls(*args, **kwargs)`` and attaches; replies ``("OK", None)``
- ``("ATTACH_OBS_FACTORY", factory)`` -> constructs ``factory(replica)``
  and attaches; replies ``("OK", None)``
- ``("SHUTDOWN",)`` -> replies ``("OK", None)`` then exits

Every reply is of the form ``(status, payload)`` with status either
``"OK"`` or ``"ERR"``; ``"ERR"`` payloads are the formatted traceback
from the worker's exception. Startup failures (Replica construction)
are caught with ``BaseException`` so the parent sees the actual
exception via the handshake; in-loop failures use ``Exception`` so
``KeyboardInterrupt`` propagates and exits the worker rather than
being absorbed.
"""

from __future__ import annotations

import pickle
import traceback
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from ..replica import Replica


def _atoms_to_dict(atoms: Atoms) -> dict[str, Any]:
    return {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": np.asarray(atoms.positions, dtype=np.float64),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
    }


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
                _, cls, args, kwargs = cmd
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
