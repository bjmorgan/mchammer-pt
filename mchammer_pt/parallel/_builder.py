"""Build-time inputs for persistent pool workers.

Each ``Builder`` is a frozen dataclass that carries the inputs needed
to construct a replica from serialisable values, with a ``build()``
method that does the construction. Builders cross the spawn boundary
via pickle; they are not used after the worker has constructed its
replica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase import Atoms
from icet import ClusterExpansion
from mchammer.ensembles import CanonicalEnsemble, WangLandauEnsemble

from ..replica import Replica
from ..wl_replica import WangLandauReplica


@dataclass(frozen=True, slots=True)
class CanonicalBuilder:
    """Build inputs for a canonical-ensemble :class:`Replica`."""

    ce_path: str
    atoms_dict: dict[str, Any]
    temperature: float
    seed: int
    ensemble_cls: type[CanonicalEnsemble]
    ensemble_kwargs: dict[str, Any]

    def build(self) -> Replica:
        """Construct the replica from the configured inputs."""
        atoms = Atoms(
            numbers=self.atoms_dict["numbers"],
            positions=self.atoms_dict["positions"],
            cell=self.atoms_dict["cell"],
            pbc=self.atoms_dict["pbc"],
        )
        ce = ClusterExpansion.read(self.ce_path)
        return Replica(
            cluster_expansion=ce,
            atoms=atoms,
            temperature=self.temperature,
            random_seed=self.seed,
            ensemble_cls=self.ensemble_cls,
            ensemble_kwargs=self.ensemble_kwargs,
            cluster_expansion_path=self.ce_path,
        )


@dataclass(frozen=True, slots=True)
class WLBuilder:
    """Build inputs for a Wang-Landau :class:`WangLandauReplica`."""

    ce_path: str
    atoms_dict: dict[str, Any]
    energy_spacing: float
    energy_limit_left: float | None
    energy_limit_right: float | None
    seed: int
    ensemble_cls: type[WangLandauEnsemble]
    ensemble_kwargs: dict[str, Any]

    def build(self) -> WangLandauReplica:
        """Construct the replica from the configured inputs."""
        atoms = Atoms(
            numbers=self.atoms_dict["numbers"],
            positions=self.atoms_dict["positions"],
            cell=self.atoms_dict["cell"],
            pbc=self.atoms_dict["pbc"],
        )
        ce = ClusterExpansion.read(self.ce_path)
        return WangLandauReplica(
            cluster_expansion=ce,
            atoms=atoms,
            energy_spacing=self.energy_spacing,
            energy_limit_left=self.energy_limit_left,
            energy_limit_right=self.energy_limit_right,
            random_seed=self.seed,
            ensemble_cls=self.ensemble_cls,
            ensemble_kwargs=self.ensemble_kwargs,
            cluster_expansion_path=self.ce_path,
        )
