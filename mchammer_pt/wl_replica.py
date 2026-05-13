"""Per-window Wang-Landau replica handle.

Sibling of `mchammer_pt.replica.Replica`. Wraps a single
`icet.mchammer.ensembles.WangLandauEnsemble` (defaulting to
`OneOverTWangLandauEnsemble`) for use inside the REWL orchestrator.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion  # type: ignore[import-untyped]
from mchammer.calculators import (  # type: ignore[import-untyped]
    ClusterExpansionCalculator,
)
from mchammer.data_containers.wang_landau_data_container import (  # type: ignore[import-untyped]
    WangLandauDataContainer,
)
from mchammer.ensembles import WangLandauEnsemble  # type: ignore[import-untyped]
from mchammer.ensembles.one_over_t_wang_landau_ensemble import (  # type: ignore[import-untyped]
    OneOverTWangLandauEnsemble,
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
        converted: dict[int, Any] = {}
        for key, val in value.items():
            if isinstance(val, dict):
                val = {int(k): v for k, v in val.items()}
            converted[int(key)] = val
        last_state[tag] = converted


class WangLandauReplica:
    """One Wang-Landau ensemble at one energy window, wrapped for REWL use.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: starting structure. Its energy must lie inside the
            window (validated at construction).
        energy_spacing: bin size of the WL energy grid.
        energy_limit_left: lower window edge, or None for unbounded.
        energy_limit_right: upper window edge, or None for unbounded.
        random_seed: seed for this replica's MC random generator.
        ensemble_cls: WL ensemble class. Defaults to
            `OneOverTWangLandauEnsemble`; any `WangLandauEnsemble`
            subclass works.
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
        ensemble_cls: type[WangLandauEnsemble] = OneOverTWangLandauEnsemble,
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

        Reads the live `_entropy` dict on the WL ensemble. Unvisited
        in-window bins return 0.0 (icet's default for missing keys).
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
        """
        occ = np.asarray(occupations, dtype=int)
        e = self._ensemble
        e.update_occupations(sites=list(range(len(occ))), species=list(occ))
        e._potential = float(
            e.calculator.calculate_total(occupations=e.configuration.occupations)
        )
        e._reached_energy_window = e._inside_energy_window(
            e._get_bin_index(e._potential)
        )

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

    def attach_mchammer_observer(self, observer) -> None:
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
        ensemble is a `OneOverTWangLandauEnsemble`. Returns the
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
        # The 1/t schedule fields live directly on `_last_state`
        # rather than via `_update_last_state`, mirroring the inline
        # writes performed by
        # `OneOverTWangLandauEnsemble.write_data_container`.
        if isinstance(e, OneOverTWangLandauEnsemble):
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
        container: WangLandauDataContainer,
        *,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> None:
        """Mutate this replica to match a saved checkpoint."""
        self._ensemble._data_container = container
        # `BaseDataContainer.read` deserialises `_last_state` via JSON,
        # which coerces integer dict keys to strings.
        # `WangLandauDataContainer.read` overrides this and converts
        # them back. Containers reaching us through
        # `mchammer_pt.history.read_hdf5` are read as plain
        # `BaseDataContainer`s (history.py does not own WL knowledge),
        # so the conversion has to happen here for `_restart_ensemble`
        # to find its integer-keyed bin lookups.
        _coerce_wl_last_state_keys_to_int(container._last_state)
        caller_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble._restart_ensemble()
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)
        # After `_restart_ensemble`, configuration occupations match
        # the saved state; refresh the WL-cached potential and window
        # flag the same way `set_occupations` does.
        e = self._ensemble
        e._potential = float(
            e.calculator.calculate_total(occupations=e.configuration.occupations)
        )
        e._reached_energy_window = e._inside_energy_window(
            e._get_bin_index(e._potential)
        )
        if sites_by_species is not None:
            self._ensemble.configuration._sites_by_species = sites_by_species

    @classmethod
    def restart_from(
        cls,
        container: WangLandauDataContainer,
        *,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        energy_spacing: float,
        energy_limit_left: float | None,
        energy_limit_right: float | None,
        random_seed: int,
        ensemble_cls: type[WangLandauEnsemble] = OneOverTWangLandauEnsemble,
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
