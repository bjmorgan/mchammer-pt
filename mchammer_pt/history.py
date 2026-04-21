"""Persistence layer for parallel-tempering runs.

`ExchangeHistory` is a dataclass holding the PT-level observations
produced during a run: per-cycle energies per replica, per-cycle
replica-label positions on the ladder, and per-pair swap attempt and
acceptance counts. It is cheap to construct in memory and maps
one-to-one onto a single HDF5 group.

`write_hdf5` bundles an `ExchangeHistory` together with one
`mchammer.BaseDataContainer` per replica into a single HDF5 file.
`read_hdf5` is the inverse. Users who want memory-only runs pass
`data_container_file=None` to `CanonicalParallelTempering` and
nothing is written.

Layout of the HDF5 file:

    /
    ├── meta/                         # run metadata
    ├── replicas/
    │   ├── 0                         # opaque bytes: mchammer tarball
    │   ├── 1
    │   └── ...
    └── exchanges/
        ├── energies_per_cycle        # (n_cycles+1, n_replicas) float64
        ├── replica_labels_per_cycle  # (n_cycles+1, n_replicas) int64
        ├── swap_attempted            # (n_replicas-1,) int64
        └── swap_accepted             # (n_replicas-1,) int64

Each replica's container is stored as an opaque byte dataset — the
`mchammer.BaseDataContainer` on-disk format (a tarball) is owned by
mchammer and treated as a black box here. `read_hdf5` reverses the
embedding via a temp file.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)

# Types allowed in the `meta` dict. h5py group attrs accept scalars
# (int, float, str, bool) and numpy arrays; nested dicts, None, and
# lists are not round-trippable. Narrow type here rather than
# `dict[str, Any]` to document the contract at the call site.
MetaValue = int | float | str | bool | np.ndarray


@dataclass
class ExchangeHistory:
    """Per-cycle PT observations.

    Attributes:
        energies_per_cycle: total CE energy (eV) of each replica at the
            end of each cycle, shape ``(n_cycles+1, n_replicas)``. Row 0
            is pre-run energies.
        replica_labels_per_cycle: which original replica *label* is
            currently at each temperature index, shape
            ``(n_cycles+1, n_replicas)``. Labels permute on accepted
            exchanges.
        swap_attempted: per-pair attempt counts, shape
            ``(n_replicas-1,)``.
        swap_accepted: per-pair accepted counts, same shape.
    """

    energies_per_cycle: np.ndarray
    replica_labels_per_cycle: np.ndarray
    swap_attempted: np.ndarray
    swap_accepted: np.ndarray

    @classmethod
    def empty(cls, n_cycles: int, n_replicas: int) -> ExchangeHistory:
        """Allocate a zero-filled history of the given shape."""
        return cls(
            energies_per_cycle=np.zeros((n_cycles + 1, n_replicas), dtype=np.float64),
            replica_labels_per_cycle=np.zeros(
                (n_cycles + 1, n_replicas), dtype=np.int64
            ),
            swap_attempted=np.zeros(n_replicas - 1, dtype=np.int64),
            swap_accepted=np.zeros(n_replicas - 1, dtype=np.int64),
        )


def write_hdf5(
    path: Path | str,
    history: ExchangeHistory,
    replica_containers: list[BaseDataContainer],
    meta: dict[str, MetaValue],
) -> None:
    """Write an `ExchangeHistory`, replica containers, and metadata.

    Each container is serialised via its `write` method (which produces
    an mchammer tarball) and the resulting bytes are embedded as a
    single opaque ``uint8`` dataset at ``/replicas/<i>``.
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        exchanges = f.create_group("exchanges")
        exchanges.create_dataset("energies_per_cycle", data=history.energies_per_cycle)
        exchanges.create_dataset(
            "replica_labels_per_cycle", data=history.replica_labels_per_cycle
        )
        exchanges.create_dataset("swap_attempted", data=history.swap_attempted)
        exchanges.create_dataset("swap_accepted", data=history.swap_accepted)

        meta_group = f.create_group("meta")
        for key, value in meta.items():
            meta_group.attrs[key] = value

        replicas = f.create_group("replicas")
        for i, container in enumerate(replica_containers):
            with tempfile.NamedTemporaryFile(suffix=".dc", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                container.write(str(tmp_path))
                payload = tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)
            replicas.create_dataset(str(i), data=np.frombuffer(payload, dtype=np.uint8))


def read_hdf5(
    path: Path | str,
) -> tuple[ExchangeHistory, list[BaseDataContainer], dict[str, MetaValue]]:
    """Read a file written by `write_hdf5`.

    Returns the `ExchangeHistory`, a list of `BaseDataContainer`s (one
    per replica group, in integer-ID order), and the metadata dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with h5py.File(path, "r") as f:
        exchanges = f["exchanges"]
        history = ExchangeHistory(
            energies_per_cycle=np.array(exchanges["energies_per_cycle"]),
            replica_labels_per_cycle=np.array(exchanges["replica_labels_per_cycle"]),
            swap_attempted=np.array(exchanges["swap_attempted"]),
            swap_accepted=np.array(exchanges["swap_accepted"]),
        )
        meta: dict[str, MetaValue] = {}
        for key, value in f["meta"].attrs.items():
            meta[key] = np.array(value) if isinstance(value, np.ndarray) else value
        containers: list[BaseDataContainer] = []
        if "replicas" in f:
            replica_keys = sorted(f["replicas"].keys(), key=int)
            for key in replica_keys:
                payload = f[f"replicas/{key}"][()].tobytes()
                with tempfile.NamedTemporaryFile(suffix=".dc", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    tmp_path.write_bytes(payload)
                    containers.append(BaseDataContainer.read(str(tmp_path)))
                finally:
                    tmp_path.unlink(missing_ok=True)
    return history, containers, meta
