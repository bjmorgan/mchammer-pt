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

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.ensembles import CanonicalEnsemble

from ..replica import Replica
from ..wl_ensemble import CoordinatedWangLandauEnsemble
from ..wl_replica import WangLandauReplica


@dataclass(frozen=True, slots=True)
class AtomsSpec:
    """Pickle-safe spec for an ``ase.Atoms`` instance.

    ``ase.Atoms`` is not reliably picklable (its ``Cell`` carries
    C-extension state); ``AtomsSpec`` captures the four fields the
    workers need as plain numpy arrays so a worker's builder can
    reconstruct the ``Atoms`` inside the spawned subprocess.

    Attributes:
        numbers: ``(N,)`` ``int64`` array of atomic numbers, one per
            site.
        positions: ``(N, 3)`` ``float64`` array of Cartesian
            coordinates (angstrom).
        cell: ``(3, 3)`` ``float64`` array; rows are the cell vectors
            (angstrom).
        pbc: ``(3,)`` ``bool`` array; per-axis periodic boundary flags.
    """

    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> AtomsSpec:
        """Capture an ``Atoms`` instance as a serialisable spec.

        Copies each array (breaking aliasing with the input ``atoms``)
        and marks the copies non-writeable so the spec is deeply
        immutable, matching the ``frozen=True`` declaration.
        """
        def _frozen(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
            out = np.array(arr, dtype=dtype)
            out.setflags(write=False)
            return out

        return cls(
            numbers=_frozen(atoms.numbers, np.dtype(np.int64)),
            positions=_frozen(atoms.positions, np.dtype(np.float64)),
            cell=_frozen(atoms.cell.array, np.dtype(np.float64)),
            pbc=_frozen(atoms.pbc, np.dtype(bool)),
        )

    def to_atoms(self) -> Atoms:
        """Reconstruct an ``Atoms`` instance from this spec."""
        return Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
        )


@dataclass(frozen=True, slots=True)
class CanonicalBuilder:
    """Build inputs for a canonical-ensemble :class:`Replica`."""

    ce_path: str
    atoms: AtomsSpec
    temperature: float
    seed: int
    ensemble_cls: type[CanonicalEnsemble]
    ensemble_kwargs: dict[str, Any]

    def build(self) -> Replica:
        """Construct the replica from the configured inputs.

        Reads ``ce_path`` from disk (typically inside the spawned
        subprocess) and forwards everything to the :class:`Replica`
        constructor.
        """
        ce = ClusterExpansion.read(self.ce_path)
        return Replica(
            cluster_expansion=ce,
            atoms=self.atoms.to_atoms(),
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
    atoms: AtomsSpec
    energy_spacing: float
    energy_limit_left: float | None
    energy_limit_right: float | None
    seed: int
    ensemble_cls: type[CoordinatedWangLandauEnsemble]
    ensemble_kwargs: dict[str, Any]

    def build(self) -> WangLandauReplica:
        """Construct the replica from the configured inputs.

        Reads ``ce_path`` from disk (typically inside the spawned
        subprocess) and forwards everything to the
        :class:`WangLandauReplica` constructor.
        """
        ce = ClusterExpansion.read(self.ce_path)
        return WangLandauReplica(
            cluster_expansion=ce,
            atoms=self.atoms.to_atoms(),
            energy_spacing=self.energy_spacing,
            energy_limit_left=self.energy_limit_left,
            energy_limit_right=self.energy_limit_right,
            random_seed=self.seed,
            ensemble_cls=self.ensemble_cls,
            ensemble_kwargs=self.ensemble_kwargs,
            cluster_expansion_path=self.ce_path,
        )
