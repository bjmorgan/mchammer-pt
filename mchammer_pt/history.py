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

import os
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


def _normalise_meta_value(value: object) -> MetaValue:
    """Cast h5py-returned attrs to the declared ``MetaValue`` union.

    h5py returns numpy scalar types (``np.int64``, ``np.float64``,
    ``np.bool_``, ``np.bytes_``) and plain ``bytes`` for attrs, not
    the Python ``int`` / ``float`` / ``bool`` / ``str`` declared in
    ``MetaValue``. Normalise on the read path so callers see the
    contract types without having to cast.

    The ``bytes`` check runs before ``np.generic`` because ``np.bytes_``
    is a subclass of both, and ``np.bytes_.item()`` returns plain
    ``bytes`` rather than ``str`` — so the ``np.generic`` branch
    would leak ``bytes`` past the decode step if ordered first.
    """
    if isinstance(value, np.ndarray):
        return np.array(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value  # type: ignore[return-value]


@dataclass(eq=False)
class ExchangeHistory:
    """Per-cycle PT observations.

    ``eq=False`` is set because the four dataclass fields are numpy
    arrays, for which ``==`` returns an element-wise array rather than
    a bool. The auto-generated ``__eq__`` would be broken: ``h1 == h2``
    would raise ``ValueError: The truth value of an array with more
    than one element is ambiguous``. Callers that want structural
    equality should compare the arrays field-by-field (or
    ``numpy.array_equal``).

    Attributes:
        energies_per_cycle: total CE energy (eV) at the end of each
            cycle, shape ``(n_cycles+1, n_replicas)``. Column ``k`` is
            the sample stream at temperature position ``k`` on the
            ladder; the configuration at that temperature position may
            change on accepted exchanges. Row 0 is the pre-run
            snapshot.
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

    @classmethod
    def concatenate(cls, *histories: ExchangeHistory) -> ExchangeHistory:
        """Concatenate sequential histories from successive runs.

        Stacks ``energies_per_cycle`` and ``replica_labels_per_cycle``
        along the cycle axis, dropping the pre-run snapshot (row 0)
        from every history after the first. Sums ``swap_attempted``
        and ``swap_accepted`` element-wise.

        All histories must come from runs on the same temperature
        ladder. This method validates replica count but cannot check
        temperature agreement (temperatures are not stored on
        ``ExchangeHistory``).

        Raises:
            ValueError: if no histories are provided, or if replica
                counts differ across histories.
        """
        if not histories:
            raise ValueError("concatenate requires at least one history")
        n_replicas = histories[0].energies_per_cycle.shape[1]
        for i, h in enumerate(histories):
            if h.energies_per_cycle.shape[1] != n_replicas:
                raise ValueError(
                    f"history {i} has {h.energies_per_cycle.shape[1]} "
                    f"replicas but history 0 has {n_replicas}"
                )
        energy_parts = [histories[0].energies_per_cycle] + [
            h.energies_per_cycle[1:] for h in histories[1:]
        ]
        label_parts = [histories[0].replica_labels_per_cycle] + [
            h.replica_labels_per_cycle[1:] for h in histories[1:]
        ]
        swap_attempted = np.zeros_like(histories[0].swap_attempted)
        swap_accepted = np.zeros_like(histories[0].swap_accepted)
        for h in histories:
            swap_attempted = swap_attempted + h.swap_attempted
            swap_accepted = swap_accepted + h.swap_accepted
        return cls(
            energies_per_cycle=np.concatenate(energy_parts, axis=0),
            replica_labels_per_cycle=np.concatenate(label_parts, axis=0),
            swap_attempted=swap_attempted,
            swap_accepted=swap_accepted,
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

    Writes are atomic: the file is first written to a sibling ``.tmp``
    path and renamed on success via ``os.replace``. A partial or failed
    write leaves the target path untouched.
    """
    path = Path(path)
    tmp_target = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(tmp_target, "w") as f:
            exchanges = f.create_group("exchanges")
            exchanges.create_dataset(
                "energies_per_cycle", data=history.energies_per_cycle
            )
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
                replicas.create_dataset(
                    str(i), data=np.frombuffer(payload, dtype=np.uint8)
                )
        os.replace(tmp_target, path)
    except BaseException:
        # Clean the partial .tmp on any failure; leave the target path
        # untouched so read_hdf5 never sees a half-written file.
        Path(tmp_target).unlink(missing_ok=True)
        raise


_REQUIRED_GROUPS = ("exchanges", "meta", "replicas")
_REQUIRED_EXCHANGE_DATASETS = (
    "energies_per_cycle",
    "replica_labels_per_cycle",
    "swap_attempted",
    "swap_accepted",
)


def read_hdf5(
    path: Path | str,
) -> tuple[ExchangeHistory, list[BaseDataContainer], dict[str, MetaValue]]:
    """Read a file written by `write_hdf5`.

    Returns the `ExchangeHistory`, a list of `BaseDataContainer`s (one
    per replica group, in integer-ID order), and the metadata dict.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if the file is missing one of the required
            top-level groups (``exchanges``, ``meta``, ``replicas``)
            or one of the required ``exchanges/`` datasets.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with h5py.File(path, "r") as f:
        for group_name in _REQUIRED_GROUPS:
            if group_name not in f:
                raise KeyError(
                    f"{path}: missing required top-level group '{group_name}'. "
                    f"File does not look like mchammer-pt HDF5 output."
                )
        exchanges = f["exchanges"]
        for dataset_name in _REQUIRED_EXCHANGE_DATASETS:
            if dataset_name not in exchanges:
                raise KeyError(
                    f"{path}: missing required dataset "
                    f"'exchanges/{dataset_name}'. "
                    f"File may be from an incompatible mchammer-pt version."
                )
        history = ExchangeHistory(
            energies_per_cycle=np.array(exchanges["energies_per_cycle"]),
            replica_labels_per_cycle=np.array(exchanges["replica_labels_per_cycle"]),
            swap_attempted=np.array(exchanges["swap_attempted"]),
            swap_accepted=np.array(exchanges["swap_accepted"]),
        )
        meta: dict[str, MetaValue] = {}
        for key, value in f["meta"].attrs.items():
            meta[key] = _normalise_meta_value(value)
        containers: list[BaseDataContainer] = []
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
