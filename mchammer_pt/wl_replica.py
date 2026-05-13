"""Per-window Wang-Landau replica handle.

Sibling of `mchammer_pt.replica.Replica`. Wraps a single
`icet.mchammer.ensembles.WangLandauEnsemble` for use inside the REWL
orchestrator. To use the Belardinelli-Pereyra 1/t schedule, pass
``ensemble_kwargs={'schedule': '1_over_t'}``; the default
``schedule='halving'`` gives the standard WL fill-factor scheme.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import (
    ClusterExpansionCalculator,
)
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)
from mchammer.ensembles import WangLandauEnsemble
from mchammer.observers.base_observer import (
    BaseObserver,
)

_RESERVED_ENSEMBLE_KWARGS: frozenset[str] = frozenset(
    {
        "structure",
        "calculator",
        "energy_spacing",
        "energy_limit_left",
        "energy_limit_right",
        "random_seed",
        "dc_filename",
    }
)

# `_last_state` fields whose dict keys are integer bin indices.
# JSON round-trips coerce these to strings; the conversion has to be
# reversed before mchammer's `_restart_ensemble` reads them. Matches
# the set `WangLandauDataContainer.read` converts upstream.
_WL_INT_KEY_FIELDS: frozenset[str] = frozenset(
    {"histogram", "entropy", "fill_factor_history", "entropy_history"}
)


def _coerce_wl_last_state_keys_to_int(last_state: dict[str, Any]) -> None:
    """Convert string dict keys back to ints in WL `_last_state` fields.

    Mirrors the conversion `WangLandauDataContainer.read` applies
    inline; used on the WL resume path when the container was
    deserialised via `BaseDataContainer.read` (which does not know
    about WL).
    """
    for tag in _WL_INT_KEY_FIELDS:
        if tag not in last_state:
            continue
        value = last_state[tag]
        if not value:
            continue
        first_key = next(iter(value))
        if isinstance(first_key, int):
            continue  # already int-keyed
        try:
            converted: dict[int, Any] = {}
            for key, val in value.items():
                if isinstance(val, dict):
                    val = {int(k): v for k, v in val.items()}
                converted[int(key)] = val
        except ValueError as exc:
            raise ValueError(
                f"WL `_last_state` field {tag!r} contains a non-integer "
                f"bin key; the checkpoint is malformed. Original error: "
                f"{exc}"
            ) from exc
        last_state[tag] = converted


class WangLandauReplica:
    """One Wang-Landau ensemble at one energy window, wrapped for REWL use.

    Invariant: a `WangLandauReplica` always has a configuration whose
    energy lies inside its assigned window. The constructor validates
    this at startup, and both `set_occupations` and `restore_state`
    validate the proposed energy before mutating any state so a
    window-violating call leaves the replica untouched. The REWL
    acceptance formula in `WangLandauParallelTempering._log_prob_ratio`
    relies on this invariant to short-circuit cleanly when only the
    "cross-bin" terms can be -inf.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: starting structure. Its energy must lie inside the
            window (validated at construction).
        energy_spacing: bin size of the WL energy grid.
        energy_limit_left: lower window edge, or None for unbounded.
        energy_limit_right: upper window edge, or None for unbounded.
        random_seed: seed for this replica's MC random generator.
        ensemble_cls: WL ensemble class. Defaults to
            `WangLandauEnsemble`. To use the 1/t schedule, pass
            ``ensemble_kwargs={'schedule': '1_over_t'}``.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.
            Reserved names (see `_RESERVED_ENSEMBLE_KWARGS`) cannot
            appear here — they are set by the wrapper.
        cluster_expansion_path: same semantics as
            `mchammer_pt.replica.Replica`.

    Raises:
        ValueError: if `ensemble_kwargs` contains a reserved name, or
            if the initial configuration's energy is outside the
            window.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        energy_spacing: float,
        energy_limit_left: float | None,
        energy_limit_right: float | None,
        random_seed: int,
        *,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
        cluster_expansion_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._energy_spacing = float(energy_spacing)
        self._energy_limit_left = (
            None if energy_limit_left is None else float(energy_limit_left)
        )
        self._energy_limit_right = (
            None if energy_limit_right is None else float(energy_limit_right)
        )
        self._cluster_expansion_path = (
            None
            if cluster_expansion_path is None
            else os.fspath(cluster_expansion_path)
        )
        extra = dict(ensemble_kwargs) if ensemble_kwargs else {}
        clash = _RESERVED_ENSEMBLE_KWARGS & extra.keys()
        if clash:
            raise ValueError(
                f"ensemble_kwargs must not contain {sorted(clash)}; "
                f"these are set by WangLandauReplica from its own "
                f"arguments (structure/calculator from "
                f"cluster_expansion+atoms; energy_spacing, "
                f"energy_limit_left, energy_limit_right, "
                f"random_seed from their dedicated parameters; "
                f"dc_filename is always pinned to None to disable "
                f"periodic on-disk writes)."
            )
        atoms_copy: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
        calculator = ClusterExpansionCalculator(atoms_copy, cluster_expansion)
        caller_state = random.getstate()
        try:
            self._ensemble: WangLandauEnsemble = ensemble_cls(
                structure=atoms_copy,
                calculator=calculator,
                energy_spacing=self._energy_spacing,
                energy_limit_left=self._energy_limit_left,
                energy_limit_right=self._energy_limit_right,
                random_seed=int(random_seed),
                dc_filename=None,
                **extra,
            )
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)

        # Validate the initial configuration lies inside the window.
        e = self._ensemble
        bin_init = e._get_bin_index(e._potential)
        if bin_init is None or not e._inside_energy_window(bin_init):
            raise ValueError(
                f"initial energy {e._potential} (bin {bin_init}) is "
                f"outside window "
                f"[{self._energy_limit_left}, {self._energy_limit_right}]; "
                f"each WL replica must start with a configuration whose "
                f"energy lies in its window."
            )

    @property
    def energy_window(self) -> tuple[float | None, float | None]:
        return (self._energy_limit_left, self._energy_limit_right)

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    @property
    def ensemble(self) -> WangLandauEnsemble:
        return self._ensemble

    def current_energy(self) -> float:
        """Cached running total of the WL ensemble (eV)."""
        return float(self._ensemble._potential)

    def current_occupations(self) -> np.ndarray:
        return self._ensemble.configuration.occupations.copy()

    def log_g(self, energy: float) -> float:
        """Return ln g at the given energy, or -inf if outside the window.

        Returns 0.0 (i.e. g = 1) for unvisited in-window bins. This is the
        standard REWL convention: an unvisited bin is treated as singly-
        degenerate, so a swap proposal that would target it incurs the
        full WL acceptance bias. Returning -inf here would forbid the
        swap until the bin is first visited by within-window MC, which
        is the wrong physics for REWL exchange acceptance.
        """
        e = self._ensemble
        bin_idx = e._get_bin_index(energy)
        if bin_idx is None or not e._inside_energy_window(bin_idx):
            return -float(np.inf)
        return float(e._entropy.get(bin_idx, 0.0))

    def set_occupations(self, occupations: np.ndarray) -> None:
        """Overwrite the replica's configuration and refresh WL-specific caches.

        `WangLandauEnsemble` caches `_potential` as a running total
        mutated in-place by the acceptance loop. `update_occupations`
        alone does NOT refresh it, so this method also recomputes the
        cached potential and the `_reached_energy_window` flag from
        the new configuration. Without this refresh, the next trial
        step would look up the wrong bin in `_entropy`, silently
        corrupting the entropy estimate.

        Validates the proposed configuration's energy before mutating
        any state, so a window violation leaves the replica untouched.
        """
        occ = np.asarray(occupations, dtype=int)
        e = self._ensemble
        proposed_potential = float(e.calculator.calculate_total(occupations=occ))
        new_bin = e._get_bin_index(proposed_potential)
        if new_bin is None or not e._inside_energy_window(new_bin):
            raise ValueError(
                f"set_occupations would leave replica at energy "
                f"{proposed_potential} (bin {new_bin}), outside window "
                f"[{self._energy_limit_left}, {self._energy_limit_right}]."
            )
        e.update_occupations(sites=list(range(len(occ))), species=list(occ))
        e._potential = proposed_potential
        e._reached_energy_window = True

    def advance(self, n_steps: int) -> None:
        """Run `n_steps` WL trial steps, isolating this replica's RNG stream.

        Mirrors the save/restore discipline used by
        `mchammer_pt.Replica.advance`. icet's `BaseEnsemble.run` may
        short-circuit early once the underlying WL ensemble has
        converged (`_terminate_sampling`); the orchestrator handles
        this case by stopping the global loop when all replicas
        report `converged`. See `converged`.
        """
        previous_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble.run(int(n_steps))
            self._rng_state = random.getstate()
        finally:
            random.setstate(previous_state)

    @property
    def converged(self) -> bool:
        """True once the underlying WL ensemble has flagged convergence."""
        return bool(self._ensemble.converged or False)

    def data_container(self) -> WangLandauDataContainer:
        """The replica's live `WangLandauDataContainer`."""
        return self._ensemble.data_container

    def attach_mchammer_observer(self, observer: BaseObserver) -> None:
        """Attach an mchammer observer; fires inside `advance(...)`."""
        self._ensemble.attach_observer(observer)

    @property
    def cluster_expansion_path(self) -> str | None:
        return self._cluster_expansion_path

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        """Refresh `_last_state` on the live container and return extras.

        Populates the fields icet's WL `_restart_ensemble` reads on resume
        (`last_step`, `occupations`, `accepted_trials`, `random_state`,
        `fill_factor`, `fill_factor_history`, `entropy_history`,
        `histogram`, `entropy`), plus the 1/t-schedule fields when the
        ensemble subclass exposes them. Returns the
        `sites_by_species` extras the orchestrator-side checkpoint code
        embeds alongside the container.
        """
        from collections import OrderedDict

        e = self._ensemble
        e._data_container._update_last_state(
            last_step=e.step,
            occupations=e.configuration.occupations.tolist(),
            accepted_trials=e._accepted_trials,
            random_state=self._rng_state,
            fill_factor=e._fill_factor,
            fill_factor_history=e._fill_factor_history,
            entropy_history=e._entropy_history,
            histogram=OrderedDict(sorted(e._histogram.items())),
            entropy=OrderedDict(sorted(e._entropy.items())),
        )
        # `schedule` and the 1/t-phase fields live directly on
        # `_last_state` rather than via `_update_last_state`, mirroring
        # icet's `write_data_container`. `_restart_ensemble` validates
        # `schedule` on resume, so it must always be present.
        if hasattr(e, "_schedule"):
            e._data_container._last_state["schedule"] = e._schedule
        if hasattr(e, "_in_one_over_t_phase"):
            e._data_container._last_state[
                "in_one_over_t_phase"
            ] = e._in_one_over_t_phase
            e._data_container._last_state[
                "window_entry_step"
            ] = e._window_entry_step
            e._data_container._last_state["switch_mode"] = e._switch_mode
        sites_by_species: list[dict[int, list[int]]] = [
            {
                int(species): [int(s) for s in sites]
                for species, sites in sublattice.items()
            }
            for sublattice in e.configuration._sites_by_species
        ]
        return {"sites_by_species": sites_by_species}

    def restore_state(
        self,
        container: BaseDataContainer,
        *,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> None:
        """Mutate this replica to match a saved checkpoint.

        Validates the proposed energy from the container's
        ``_last_state`` before touching any ensemble state, so a
        window violation leaves the replica untouched.

        ``container`` is a `BaseDataContainer` (not a
        `WangLandauDataContainer`) because `read_hdf5` deserialises
        containers generically. Only ``_last_state`` is needed for
        restoration; the existing `WangLandauDataContainer` on the
        ensemble is preserved so WL-specific post-processing methods
        remain available after resume.
        """
        # `BaseDataContainer.read` deserialises `_last_state` via JSON,
        # which coerces integer dict keys to strings.
        # `WangLandauDataContainer.read` overrides this and converts
        # them back. Containers reaching us through
        # `mchammer_pt.history.read_hdf5` are read as plain
        # `BaseDataContainer`s (history.py does not own WL knowledge),
        # so the conversion has to happen here for `_restart_ensemble`
        # to find its integer-keyed bin lookups.
        _coerce_wl_last_state_keys_to_int(container._last_state)

        # Validate the proposed configuration's energy before mutating
        # any ensemble state.
        proposed_occ = np.asarray(
            container._last_state["occupations"], dtype=int
        )
        proposed_potential = float(
            self._ensemble.calculator.calculate_total(occupations=proposed_occ)
        )
        new_bin = self._ensemble._get_bin_index(proposed_potential)
        if new_bin is None or not self._ensemble._inside_energy_window(new_bin):
            raise ValueError(
                f"restore_state would leave replica at energy "
                f"{proposed_potential} (bin {new_bin}), outside window "
                f"[{self._energy_limit_left}, {self._energy_limit_right}]."
            )

        # Copy the saved state into the existing WL-typed container
        # rather than replacing it wholesale. `read_hdf5` returns
        # `BaseDataContainer` instances; assigning one to the ensemble
        # would lose the `WangLandauDataContainer` subclass and break
        # WL-specific post-processing.
        self._ensemble._data_container._last_state = container._last_state
        caller_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble._restart_ensemble()
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)
        # After `_restart_ensemble`, configuration occupations match
        # the saved state; refresh the WL-cached potential and window
        # flag (reusing the already-computed potential).
        e = self._ensemble
        e._potential = proposed_potential
        e._reached_energy_window = True
        if sites_by_species is not None:
            self._ensemble.configuration._sites_by_species = sites_by_species

    @classmethod
    def restart_from(
        cls,
        container: BaseDataContainer,
        *,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        energy_spacing: float,
        energy_limit_left: float | None,
        energy_limit_right: float | None,
        random_seed: int,
        ensemble_cls: type[WangLandauEnsemble] = WangLandauEnsemble,
        ensemble_kwargs: Mapping[str, Any] | None = None,
        cluster_expansion_path: str | os.PathLike[str] | None = None,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> WangLandauReplica:
        """Construct a replica whose ensemble has been restored from `container`."""
        replica = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms,
            energy_spacing=energy_spacing,
            energy_limit_left=energy_limit_left,
            energy_limit_right=energy_limit_right,
            random_seed=random_seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            cluster_expansion_path=cluster_expansion_path,
        )
        replica.restore_state(container, sites_by_species=sites_by_species)
        return replica
