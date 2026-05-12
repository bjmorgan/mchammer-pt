"""Per-window Wang-Landau replica handle.

Sibling of `mchammer_pt.replica.Replica`. Wraps a single
`icet.mchammer.ensembles.WangLandauEnsemble` (defaulting to
`OneOverTWangLandauEnsemble`) for use inside the REWL orchestrator.

The two replica classes duplicate the RNG-isolation pattern, the
`set_occupations`/`restore_state` machinery, and the checkpoint
snapshot. Duplication is accepted for clarity per CLAUDE.md's
clarity-over-DRY priority; a third ensemble type can drive a future
extraction of a shared base.
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
