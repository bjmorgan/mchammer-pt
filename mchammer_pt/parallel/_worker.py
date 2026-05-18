"""Worker-side implementation of the persistent multiprocessing pools.

``BaseWorker`` implements the command loop and shared opcodes.
``CanonicalWorker`` and ``WangLandauWorker`` extend it with
replica-specific construction and extra opcodes. The two thin
entry points ``_worker`` and ``_wl_worker`` satisfy
``Process(target=...)``.

Shared opcodes (``BaseWorker``):

- ``("ADVANCE", n_steps)`` -> ``Reply("OK", ..., None)``
- ``("ENERGY",)`` -> ``Reply("OK", ..., float)``
- ``("GET_OCC",)`` -> ``Reply("OK", ..., np.ndarray)``
- ``("SET_OCC", occupations)`` -> ``Reply("OK", ..., None)``
- ``("GET_DC",)`` -> ``Reply("OK", ..., BaseDataContainer)``
- ``("SNAPSHOT_FOR_CHECKPOINT",)`` -> ``Reply("OK", ..., dict)``
- ``("RESTORE_STATE", container, sites_by_species)``
  -> ``Reply("OK", ..., None)``
- ``("ATTACH_OBS", blob)`` -> ``Reply("OK", ..., None)``
- ``("ATTACH_OBS_CLS", cls, args, kwargs)``
  -> ``Reply("OK", ..., None)``
- ``("ATTACH_OBS_FACTORY", factory)``
  -> ``Reply("OK", ..., None)``
- ``("GET_OBSERVERS",)``
  -> ``Reply("OK", ..., dict[str, BaseObserver])``
- ``("SHUTDOWN",)`` -> ``Reply("OK", ..., None)`` then exits

REWL-only opcodes (``WangLandauWorker``):

- ``("ADVANCE", n_steps)``
  -> ``Reply("OK", ..., (is_flat, fill_factor, entropy, step,
  window_entry_step))``
- ``("LOG_G_AT", E_i, E_j)``
  -> ``Reply("OK", ..., (g_at_E_i, g_at_E_j))``
- ``("CONVERGED",)`` -> ``Reply("OK", ..., bool)``
- ``("WL_STATS",)`` -> ``Reply("OK", ..., dict)``
- ``("GET_ENTROPY",)`` -> ``Reply("OK", ..., dict)``
- ``("SET_ENTROPY", merged_entropy)`` -> ``Reply("OK", ..., None)``
- ``("FORCE_HALVE",)`` -> ``Reply("OK", ..., None)``
- ``("SET_PHASE", phase)`` -> ``Reply("OK", ..., None)``
- ``("FINALISE_MERGE", merged_entropy)`` -> ``Reply("OK", ..., None)``

Every reply is a ``Reply(status, op, payload)`` named tuple.
``status`` is ``"OK"``, ``"ERR_PICKLE"`` (unpicklable reply;
parent translates to ``TypeError``), or ``"ERR"`` (worker-side
failure; parent translates to ``RuntimeError``).
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
from ..wl_window_group import WalkerPostBlockState
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
        self._handlers: dict[str, Callable[[tuple[Any, ...]], None]] = {
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

    def _handle_advance(self, cmd: tuple[Any, ...]) -> None:
        self._replica.advance(cmd[1])
        self._reply(None)

    def _handle_energy(self, cmd: tuple[Any, ...]) -> None:
        self._reply(self._replica.current_energy())

    def _handle_get_occ(self, cmd: tuple[Any, ...]) -> None:
        self._reply(self._replica.current_occupations())

    def _handle_set_occ(self, cmd: tuple[Any, ...]) -> None:
        self._replica.set_occupations(cmd[1])
        self._reply(None)

    def _handle_get_dc(self, cmd: tuple[Any, ...]) -> None:
        self._reply(self._replica.data_container())

    def _handle_snapshot_for_checkpoint(self, cmd: tuple[Any, ...]) -> None:
        self._reply(self._replica.snapshot_for_checkpoint())

    def _handle_restore_state(self, cmd: tuple[Any, ...]) -> None:
        _, container, sites_by_species = cmd
        self._replica.restore_state(
            container, sites_by_species=sites_by_species
        )
        self._reply(None)

    def _handle_attach_obs(self, cmd: tuple[Any, ...]) -> None:
        observer = pickle.loads(cmd[1])
        self._replica.attach_mchammer_observer(observer)
        self._reply(None)

    def _handle_attach_obs_cls(self, cmd: tuple[Any, ...]) -> None:
        _, cls, args, kwargs = cmd
        self._replica.attach_mchammer_observer(cls(*args, **kwargs))
        self._reply(None)

    def _handle_attach_obs_factory(self, cmd: tuple[Any, ...]) -> None:
        factory = cmd[1]
        observer = factory(self._replica)
        if not isinstance(observer, BaseObserver):
            raise TypeError(
                f"attach_observer_factory: factory returned "
                f"{type(observer).__name__}, not a BaseObserver"
            )
        self._replica.attach_mchammer_observer(observer)
        self._reply(None)

    def _handle_get_observers(self, cmd: tuple[Any, ...]) -> None:
        observers = self._replica.ensemble.observers
        try:
            pickle.dumps(observers)
        except Exception:
            self._reply_pickle_error(traceback.format_exc())
        else:
            self._reply(observers)

    def _handle_shutdown(self, cmd: tuple[Any, ...]) -> None:
        self._reply(None)
        self._conn.close()
        raise _Shutdown


class CanonicalWorker(BaseWorker):
    """Worker for canonical-ensemble replicas."""

    def __init__(
        self,
        conn: Connection,
        ce_path: str,
        atoms_dict: dict[str, Any],
        temperature: float,
        seed: int,
        ensemble_cls: type[CanonicalEnsemble],
        ensemble_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(conn)
        self._ce_path = ce_path
        self._atoms_dict = atoms_dict
        self._temperature = temperature
        self._seed = seed
        self._ensemble_cls = ensemble_cls
        self._ensemble_kwargs = ensemble_kwargs

    def _build_replica(self) -> Replica:
        atoms = Atoms(
            numbers=self._atoms_dict["numbers"],
            positions=self._atoms_dict["positions"],
            cell=self._atoms_dict["cell"],
            pbc=self._atoms_dict["pbc"],
        )
        ce = ClusterExpansion.read(self._ce_path)
        return Replica(
            cluster_expansion=ce,
            atoms=atoms,
            temperature=self._temperature,
            random_seed=self._seed,
            ensemble_cls=self._ensemble_cls,
            ensemble_kwargs=self._ensemble_kwargs,
            cluster_expansion_path=self._ce_path,
        )


def _worker(
    conn: Connection,
    ce_path: str,
    atoms_dict: dict[str, Any],
    temperature: float,
    seed: int,
    ensemble_cls: type[CanonicalEnsemble],
    ensemble_kwargs: dict[str, Any],
) -> None:
    """Canonical worker entry point for Process(target=...)."""
    CanonicalWorker(
        conn, ce_path, atoms_dict, temperature, seed,
        ensemble_cls, ensemble_kwargs,
    ).run()


class WangLandauWorker(BaseWorker):
    """Worker for Wang-Landau (REWL) replicas."""

    def __init__(
        self,
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
        super().__init__(conn)
        self._ce_path = ce_path
        self._atoms_dict = atoms_dict
        self._energy_spacing = energy_spacing
        self._energy_limit_left = energy_limit_left
        self._energy_limit_right = energy_limit_right
        self._seed = seed
        self._ensemble_cls = ensemble_cls
        self._ensemble_kwargs = ensemble_kwargs
        self._handlers.update({
            "LOG_G_AT": self._handle_log_g_at,
            "CONVERGED": self._handle_converged,
            "WL_STATS": self._handle_wl_stats,
            "GET_ENTROPY": self._handle_get_entropy,
            "SET_ENTROPY": self._handle_set_entropy,
            "FORCE_HALVE": self._handle_force_halve,
            "SET_PHASE": self._handle_set_phase,
            "FINALISE_MERGE": self._handle_finalise_merge,
        })

    def _build_replica(self) -> WangLandauReplica:
        atoms = Atoms(
            numbers=self._atoms_dict["numbers"],
            positions=self._atoms_dict["positions"],
            cell=self._atoms_dict["cell"],
            pbc=self._atoms_dict["pbc"],
        )
        ce = ClusterExpansion.read(self._ce_path)
        return WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=self._energy_spacing,
            energy_limit_left=self._energy_limit_left,
            energy_limit_right=self._energy_limit_right,
            random_seed=self._seed,
            ensemble_cls=self._ensemble_cls,
            ensemble_kwargs=self._ensemble_kwargs,
            cluster_expansion_path=self._ce_path,
        )

    def _handle_get_dc(self, cmd: tuple[Any, ...]) -> None:
        self._replica.refresh_last_state()
        self._reply(self._replica.data_container())

    def _handle_advance(self, cmd: tuple[Any, ...]) -> None:
        n_steps = int(cmd[1])
        self._replica.advance(n_steps)
        e = self._replica.ensemble
        self._reply(WalkerPostBlockState(
            is_flat=self._replica.is_flat(),
            fill_factor=float(e._fill_factor),
            entropy=dict(e._entropy),
            step=int(e.step),
            window_entry_step=(
                None
                if e._window_entry_step is None
                else int(e._window_entry_step)
            ),
            histogram=dict(e._histogram),
        ))

    def _handle_log_g_at(self, cmd: tuple[Any, ...]) -> None:
        _, E_i, E_j = cmd
        self._reply((self._replica.log_g(E_i), self._replica.log_g(E_j)))

    def _handle_converged(self, cmd: tuple[Any, ...]) -> None:
        self._reply(self._replica.converged)

    def _handle_wl_stats(self, cmd: tuple[Any, ...]) -> None:
        self._reply(self._replica.window_stats())

    def _handle_get_entropy(self, cmd: tuple[Any, ...]) -> None:
        self._reply(dict(self._replica.ensemble._entropy))

    def _handle_set_entropy(self, cmd: tuple[Any, ...]) -> None:
        merged = cmd[1]
        self._replica.ensemble._entropy = dict(merged)
        self._reply(None)

    def _handle_force_halve(self, cmd: tuple[Any, ...]) -> None:
        self._replica.force_halve()
        self._reply(None)

    def _handle_set_phase(self, cmd: tuple[Any, ...]) -> None:
        # Flips ``_phase`` and, on transition to ``1_over_t``, sets
        # ``_fill_factor`` to the current ``1/t``. Does not write to
        # ``_fill_factor_history``: history records halve events
        # (shared keys with ``_entropy_history``); in the 1/t phase
        # ``_fill_factor`` is reconstructed from
        # ``step - _window_entry_step + 1``.
        phase = str(cmd[1])
        e = self._replica.ensemble
        e._phase = phase
        if phase == "1_over_t":
            entry = e._window_entry_step
            if entry is not None:
                t = e.step - entry + 1
                e._fill_factor = 1.0 / t
        self._reply(None)

    def _handle_finalise_merge(self, cmd: tuple[Any, ...]) -> None:
        """Write the supplied merged entropy dict; refresh ``_last_state``.

        Used at end of run by the coordinator's finalise-for-reporting
        path. Mirrors ``SET_ENTROPY`` but also calls
        ``refresh_last_state`` so the data container picks up the
        merged values without an additional round-trip.
        """
        merged = cmd[1]
        self._replica.ensemble._entropy = dict(merged)
        self._replica.refresh_last_state()
        self._reply(None)


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
    """REWL worker entry point for Process(target=...)."""
    WangLandauWorker(
        conn, ce_path, atoms_dict, energy_spacing,
        energy_limit_left, energy_limit_right, seed,
        ensemble_cls, ensemble_kwargs,
    ).run()
