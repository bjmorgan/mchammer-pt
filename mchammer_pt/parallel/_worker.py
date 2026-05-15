"""Worker-side implementation of the persistent multiprocessing pools.

Two worker entry points live here: `_worker` for canonical replicas
(driven by `ProcessPool`) and `_wl_worker` for Wang-Landau replicas
(driven by `ProcessWangLandauPool`). Each builds its replica,
sends a single ``("OK", None)`` ready-handshake to the parent, and
then enters a command loop reading ``(opcode, *args)`` tuples and
replying with ``(status, payload)`` tuples.

Shared opcodes (both workers):

- ``("ADVANCE", n_steps)`` -> replies ``("OK", None)`` after the run
- ``("ENERGY",)`` -> replies ``("OK", float)`` with total CE energy
- ``("GET_OCC",)`` -> replies ``("OK", np.ndarray)`` with occupations
- ``("SET_OCC", occupations)`` -> overwrites state
- ``("GET_DC",)`` -> replies ``("OK", BaseDataContainer)`` (pickled)
- ``("SNAPSHOT_FOR_CHECKPOINT",)`` -> refreshes the replica's
  ``_data_container._last_state`` and replies ``("OK", dict)`` with the
  per-replica extras the checkpoint writer embeds (notably
  ``"sites_by_species"``)
- ``("RESTORE_STATE", container, sites_by_species)`` -> swaps the saved
  data container onto the replica's ensemble, drives mchammer's
  ``_restart_ensemble``, and (if ``sites_by_species`` is non-None)
  restores the ``ConfigurationManager._sites_by_species`` cache;
  replies ``("OK", None)``
- ``("SHUTDOWN",)`` -> replies ``("OK", None)`` then exits

Observer-attach opcodes (both workers):

- ``("ATTACH_OBS", pickled_blob)`` -> deserialises and attaches an
  observer; replies ``("OK", None)``
- ``("ATTACH_OBS_CLS", cls, args, kwargs)`` -> constructs
  ``cls(*args, **kwargs)`` and attaches; replies ``("OK", None)``
- ``("ATTACH_OBS_FACTORY", factory)`` -> constructs ``factory(replica)``
  and attaches; replies ``("OK", None)``
- ``("GET_OBSERVERS",)`` -> replies ``("OK", dict[str, BaseObserver])``
  with the replica's currently-attached observers (pickled on send)

REWL-only opcodes (`_wl_worker` only):

- ``("LOG_G_AT", E_i, E_j)`` -> replies ``("OK", (g_at_E_i, g_at_E_j))``
  with the replica's `log_g` evaluated at both energies
- ``("CONVERGED",)`` -> replies ``("OK", bool)`` with the replica's
  converged flag
- ``("WL_STATS",)`` -> replies ``("OK", dict)`` with lightweight
  convergence metrics (fill_factor, halvings, histogram, converged)
- ``("GET_ENTROPY_SYNC_STATE",)`` -> replies ``("OK", dict)`` with
  entropy, fill_factor_history_len, and histogram for multi-walker sync
- ``("APPLY_ENTROPY_SYNC", merged_entropy, extra_halvings)`` -> applies
  ``extra_halvings`` fill-factor halvings and writes merged entropy to the
  replica; replies ``("OK", None)``

Every reply is of the form ``(status, payload)``. ``status`` is one
of ``"OK"`` (payload is the result), ``"ERR_PICKLE"`` (the reply
payload could not be pickled — used by ``GET_OBSERVERS`` after
eagerly checking; the parent translates this to ``TypeError``), or
``"ERR"`` (any other worker-side failure; parent translates to
``RuntimeError``). ``"ERR_PICKLE"`` and ``"ERR"`` payloads are the
formatted traceback from the worker's exception.
Startup failures (Replica construction) are caught with
``BaseException`` so the parent sees the actual exception via the
handshake; in-loop failures use ``Exception`` so
``KeyboardInterrupt`` propagates and exits the worker rather than
being absorbed.
"""

from __future__ import annotations

import pickle
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from ase import Atoms
from icet import ClusterExpansion
from mchammer.ensembles import (
    CanonicalEnsemble,
    WangLandauEnsemble,
)
from mchammer.observers.base_observer import (
    BaseObserver,
)

from ..replica import Replica
from ..wl_replica import WangLandauReplica
from ._comms import Reply


class _Shutdown(BaseException):
    """Raised by SHUTDOWN handler to exit the command loop."""


class BaseWorker:
    """Command-loop base class for persistent pool workers.

    Subclasses implement ``_build_replica`` and optionally extend
    ``self._handlers`` with extra opcode handlers.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._op: str = ""
        self._replica: Any = None
        self._handlers: dict[str, Callable[[tuple], None]] = {
            "ADVANCE": self._handle_advance,
            "ENERGY": self._handle_energy,
            "GET_OCC": self._handle_get_occ,
            "SET_OCC": self._handle_set_occ,
            "GET_DC": self._handle_get_dc,
            "SNAPSHOT_FOR_CHECKPOINT": self._handle_snapshot_for_checkpoint,
            "RESTORE_STATE": self._handle_restore_state,
            "ATTACH_OBS": self._handle_attach_obs,
            "ATTACH_OBS_CLS": self._handle_attach_obs_cls,
            "ATTACH_OBS_FACTORY": self._handle_attach_obs_factory,
            "GET_OBSERVERS": self._handle_get_observers,
            "SHUTDOWN": self._handle_shutdown,
        }

    def _build_replica(self) -> Any:
        raise NotImplementedError

    def run(self) -> None:
        """Build the replica, handshake, and enter the command loop."""
        try:
            self._replica = self._build_replica()
        except BaseException:
            self._conn.send(Reply("ERR", "STARTUP", traceback.format_exc()))
            self._conn.close()
            return

        self._conn.send(Reply("OK", "STARTUP", None))

        while True:
            try:
                cmd = self._conn.recv()
            except EOFError:
                return
            self._op = cmd[0]
            handler = self._handlers.get(self._op)
            if handler is None:
                self._conn.send(
                    Reply("ERR", self._op, f"unknown command: {self._op!r}")
                )
                continue
            try:
                handler(cmd)
            except _Shutdown:
                return
            except Exception:
                self._reply_error(traceback.format_exc())

    def _reply(self, payload: Any) -> None:
        self._conn.send(Reply("OK", self._op, payload))

    def _reply_error(self, tb: str) -> None:
        self._conn.send(Reply("ERR", self._op, tb))

    def _reply_pickle_error(self, tb: str) -> None:
        self._conn.send(Reply("ERR_PICKLE", self._op, tb))

    def _handle_advance(self, cmd: tuple) -> None:
        self._replica.advance(cmd[1])
        self._reply(None)

    def _handle_energy(self, cmd: tuple) -> None:
        self._reply(self._replica.current_energy())

    def _handle_get_occ(self, cmd: tuple) -> None:
        self._reply(self._replica.current_occupations())

    def _handle_set_occ(self, cmd: tuple) -> None:
        self._replica.set_occupations(cmd[1])
        self._reply(None)

    def _handle_get_dc(self, cmd: tuple) -> None:
        self._reply(self._replica.data_container())

    def _handle_snapshot_for_checkpoint(self, cmd: tuple) -> None:
        self._reply(self._replica.snapshot_for_checkpoint())

    def _handle_restore_state(self, cmd: tuple) -> None:
        _, container, sites_by_species = cmd
        self._replica.restore_state(
            container, sites_by_species=sites_by_species
        )
        self._reply(None)

    def _handle_attach_obs(self, cmd: tuple) -> None:
        observer = pickle.loads(cmd[1])
        self._replica.attach_mchammer_observer(observer)
        self._reply(None)

    def _handle_attach_obs_cls(self, cmd: tuple) -> None:
        _, cls, args, kwargs = cmd
        self._replica.attach_mchammer_observer(cls(*args, **kwargs))
        self._reply(None)

    def _handle_attach_obs_factory(self, cmd: tuple) -> None:
        factory = cmd[1]
        observer = factory(self._replica)
        if not isinstance(observer, BaseObserver):
            raise TypeError(
                f"attach_observer_factory: factory returned "
                f"{type(observer).__name__}, not a BaseObserver"
            )
        self._replica.attach_mchammer_observer(observer)
        self._reply(None)

    def _handle_get_observers(self, cmd: tuple) -> None:
        observers = self._replica.ensemble.observers
        try:
            pickle.dumps(observers)
        except Exception:
            self._reply_pickle_error(traceback.format_exc())
        else:
            self._reply(observers)

    def _handle_shutdown(self, cmd: tuple) -> None:
        self._reply(None)
        self._conn.close()
        raise _Shutdown


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
            cluster_expansion_path=ce_path,
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
            elif op == "SNAPSHOT_FOR_CHECKPOINT":
                conn.send(("OK", replica.snapshot_for_checkpoint()))
            elif op == "RESTORE_STATE":
                _, container, sites_by_species = cmd
                replica.restore_state(
                    container, sites_by_species=sites_by_species
                )
                conn.send(("OK", None))
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
            elif op == "GET_OBSERVERS":
                # Pickling the live observer dict is safe because the
                # worker is single-threaded and idle here; a future
                # refactor adding background work would need to copy.
                observers = replica.ensemble.observers
                try:
                    pickle.dumps(observers)
                except Exception:
                    conn.send(("ERR_PICKLE", traceback.format_exc()))
                else:
                    conn.send(("OK", observers))
            elif op == "SHUTDOWN":
                conn.send(("OK", None))
                conn.close()
                return
            else:
                conn.send(("ERR", f"unknown command: {op!r}"))
        except Exception:
            conn.send(("ERR", traceback.format_exc()))


def _wl_worker(
    conn: Connection,
    ce_path: str,
    atoms_dict: dict[str, Any],
    energy_spacing: float,
    energy_limit_left: float | None,
    energy_limit_right: float | None,
    seed: int,
    ensemble_cls: type[WangLandauEnsemble],
    ensemble_kwargs: dict[str, Any],
) -> None:
    """REWL worker entry point: build a WangLandauReplica, then serve commands.

    Recognises the data/state opcodes shared with the canonical worker
    (ADVANCE, ENERGY, GET_OCC, SET_OCC, GET_DC, SNAPSHOT_FOR_CHECKPOINT,
    RESTORE_STATE, SHUTDOWN) plus three REWL-specific ones:

    - ``("LOG_G_AT", E_i, E_j)`` -> ``("OK", (g_at_E_i, g_at_E_j))``
      The worker evaluates its replica's `log_g` at the two energies
      in one round trip.
    - ``("CONVERGED",)`` -> ``("OK", bool)`` the replica's `converged`
      flag.
    - ``("WL_STATS",)`` -> ``("OK", dict)`` lightweight convergence
      metrics: fill_factor, halvings, histogram, converged.
    """
    try:
        atoms = Atoms(
            numbers=atoms_dict["numbers"],
            positions=atoms_dict["positions"],
            cell=atoms_dict["cell"],
            pbc=atoms_dict["pbc"],
        )
        ce = ClusterExpansion.read(ce_path)
        replica = WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=energy_spacing,
            energy_limit_left=energy_limit_left,
            energy_limit_right=energy_limit_right,
            random_seed=seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            cluster_expansion_path=ce_path,
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
            elif op == "SNAPSHOT_FOR_CHECKPOINT":
                conn.send(("OK", replica.snapshot_for_checkpoint()))
            elif op == "RESTORE_STATE":
                _, container, sites_by_species = cmd
                replica.restore_state(
                    container, sites_by_species=sites_by_species
                )
                conn.send(("OK", None))
            elif op == "LOG_G_AT":
                _, E_i, E_j = cmd
                conn.send(("OK", (replica.log_g(E_i), replica.log_g(E_j))))
            elif op == "CONVERGED":
                conn.send(("OK", replica.converged))
            elif op == "WL_STATS":
                conn.send(("OK", replica.window_stats()))
            elif op == "GET_ENTROPY_SYNC_STATE":
                e = replica.ensemble
                conn.send(("OK", {
                    "entropy": dict(e._entropy),
                    "fill_factor_history_len": len(e._fill_factor_history),
                    "histogram": dict(e._histogram),
                }))
            elif op == "APPLY_ENTROPY_SYNC":
                _, merged_entropy, extra_halvings = cmd
                e = replica.ensemble
                for _ in range(extra_halvings):
                    e._fill_factor /= 2.0
                    next_key = max(e._fill_factor_history, default=-1) + 1
                    e._fill_factor_history[next_key] = e._fill_factor
                    e._histogram = dict.fromkeys(e._histogram, 0)
                e._entropy = dict(merged_entropy)
                conn.send(("OK", None))
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
            elif op == "GET_OBSERVERS":
                # Pickling the live observer dict is safe because the
                # worker is single-threaded and idle here; a future
                # refactor adding background work would need to copy.
                observers = replica.ensemble.observers
                try:
                    pickle.dumps(observers)
                except Exception:
                    conn.send(("ERR_PICKLE", traceback.format_exc()))
                else:
                    conn.send(("OK", observers))
            elif op == "SHUTDOWN":
                conn.send(("OK", None))
                conn.close()
                return
            else:
                conn.send(("ERR", f"unknown command: {op!r}"))
        except Exception:
            conn.send(("ERR", traceback.format_exc()))
