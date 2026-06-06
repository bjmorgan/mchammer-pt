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

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
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
        replica_labels_per_cycle: position-indexed carrier labels.
            ``replica_labels_per_cycle[cycle][position]`` is the carrier
            id at each ``(window, slot)`` position, shape
            ``(n_cycles+1, N_w)`` where ``N_w = sum(W_i)`` equals
            ``n_replicas`` when every window has one walker. Labels
            permute on accepted exchanges.
        swap_attempted: per-pair attempt counts, shape
            ``(n_replicas-1,)``.
        swap_accepted: per-pair accepted counts, same shape.
        window_of_position: shape ``(N_w,)``, the window rung index of
            each label position. Runs always record it (the identity
            mapping for single-walker pools); it is ``None`` only when
            reading an older file written before this dataset existed.
            Lets `round_trip_counts` interpret a multi-walker label array
            read back from disk, when the live pool is gone.
    """

    energies_per_cycle: np.ndarray
    replica_labels_per_cycle: np.ndarray
    swap_attempted: np.ndarray
    swap_accepted: np.ndarray
    window_of_position: np.ndarray | None = None

    @classmethod
    def empty(
        cls,
        n_cycles: int,
        n_replicas: int,
        n_carriers: int | None = None,
        window_of_position: np.ndarray | None = None,
    ) -> ExchangeHistory:
        """Allocate a zero-filled history of the given shape.

        ``n_carriers`` is the number of position-indexed carrier labels
        (the total number of walkers across all windows). It defaults to
        ``n_replicas`` (the single-walker / canonical case).
        ``window_of_position`` maps each label position to its window
        rung; pass it for multi-walker runs so the history is
        self-describing, or leave it ``None`` for the single-walker case.
        """
        if n_carriers is None:
            n_carriers = n_replicas
        return cls(
            energies_per_cycle=np.zeros((n_cycles + 1, n_replicas), dtype=np.float64),
            replica_labels_per_cycle=np.zeros(
                (n_cycles + 1, n_carriers), dtype=np.int64
            ),
            swap_attempted=np.zeros(n_replicas - 1, dtype=np.int64),
            swap_accepted=np.zeros(n_replicas - 1, dtype=np.int64),
            window_of_position=(
                None
                if window_of_position is None
                else np.asarray(window_of_position, dtype=np.int64)
            ),
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
            # Static across segments of the same ladder; carry the first.
            window_of_position=histories[0].window_of_position,
        )


def write_hdf5(
    path: Path | str,
    history: ExchangeHistory,
    replica_containers: list[BaseDataContainer],
    meta: dict[str, MetaValue],
    orchestrator_state: dict[str, np.ndarray | str] | None = None,
    replica_extra: list[dict[str, Any]] | None = None,
    window_groups: list[dict[str, Any] | None] | None = None,
) -> None:
    """Write an `ExchangeHistory`, replica containers, and metadata.

    Each container is serialised via its `write` method (which produces
    an mchammer tarball) and the resulting bytes are embedded as a
    single opaque ``uint8`` dataset at ``/replicas/<i>``.

    When ``orchestrator_state`` is supplied, an ``/orchestrator/`` group
    is added carrying the orchestrator-level runtime state needed for
    resume: ``replica_labels`` (int64 array of the current replica
    permutation) and ``rng_state`` (a JSON string round-tripping the
    orchestrator's exchange-proposal RNG ``bit_generator.state``).
    Files written without ``orchestrator_state`` are still readable
    via `read_hdf5`; ``CanonicalParallelTempering.resume`` requires
    files that include the group.

    When ``replica_extra`` is supplied, its length must match
    ``replica_containers``; each element's ``"sites_by_species"`` field
    is JSON-encoded and stored at ``/sites_by_species/<i>``. This
    carries `ConfigurationManager._sites_by_species` — the
    path-dependent per-sublattice species → site-list cache that
    bit-identical canonical-ensemble resume requires alongside the
    container's `_last_state`.

    When ``window_groups`` is supplied, ``orchestrator_state`` must also
    be supplied (the ``/orchestrator`` group must exist). Each non-``None``
    element at index ``g`` is stored under
    ``/orchestrator/window_groups/<g>/`` with datasets ``rng_state``
    (JSON string) and ``phase`` (string). ``None`` entries are
    skipped, leaving no subgroup for that index.

    Writes are atomic: the file is first written to a sibling ``.tmp``
    path and renamed on success via ``os.replace``. A partial or failed
    write leaves the target path untouched.
    """
    if replica_extra is not None and len(replica_extra) != len(replica_containers):
        raise ValueError(
            f"replica_extra has {len(replica_extra)} entries but "
            f"replica_containers has {len(replica_containers)}; the "
            f"two must match one-to-one."
        )
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
            if history.window_of_position is not None:
                exchanges.create_dataset(
                    "window_of_position",
                    data=np.asarray(history.window_of_position, dtype=np.int64),
                )

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

            if orchestrator_state is not None:
                orchestrator_group = f.create_group("orchestrator")
                orchestrator_group.create_dataset(
                    "replica_labels",
                    data=np.asarray(
                        orchestrator_state["replica_labels"], dtype=np.int64
                    ),
                )
                orchestrator_group.create_dataset(
                    "rng_state",
                    data=str(orchestrator_state["rng_state"]),
                )

            if replica_extra is not None:
                sites_group = f.create_group("sites_by_species")
                for i, extra in enumerate(replica_extra):
                    sites_group.create_dataset(
                        str(i), data=json.dumps(extra["sites_by_species"])
                    )

            if window_groups is not None:
                if orchestrator_state is None:
                    raise ValueError(
                        "window_groups requires orchestrator_state to be supplied; "
                        "the /orchestrator group must exist to host the subgroup."
                    )
                # Omit /orchestrator/window_groups/ entirely when every
                # entry is None (the all-W=1 case). The reader relies on
                # subgroup absence to mean "this is a single-walker
                # window"; an empty-but-present parent group leaks an
                # internal detail of the writer.
                if any(entry is not None for entry in window_groups):
                    wg_parent = f["orchestrator"].create_group("window_groups")
                    for g, entry in enumerate(window_groups):
                        if entry is None:
                            continue
                        sub = wg_parent.create_group(str(g))
                        sub.create_dataset(
                            "rng_state", data=str(entry["rng_state"])
                        )
                        sub.create_dataset("phase", data=str(entry["phase"]))
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
            window_of_position=(
                np.array(exchanges["window_of_position"])
                if "window_of_position" in exchanges
                else None
            ),
        )
        meta: dict[str, MetaValue] = {}
        for key, value in f["meta"].attrs.items():
            meta[key] = _normalise_meta_value(value)
        # WL containers carry WL-specific state (int-keyed bin dicts in
        # `_last_state`, a re-tupled `_random_state`) that
        # `BaseDataContainer.read` would not restore -- it deserialises
        # via JSON and leaves bin keys as strings. Dispatch on the
        # recorded ensemble class so each consumer sees a usable
        # container without having to re-run the WL-specific coercion.
        ensemble_fqn = str(meta.get("ensemble_cls_fqn", ""))
        reader_cls: type[BaseDataContainer] = (
            WangLandauDataContainer if "WangLandau" in ensemble_fqn
            else BaseDataContainer
        )
        containers: list[BaseDataContainer] = []
        replica_keys = sorted(f["replicas"].keys(), key=int)
        for key in replica_keys:
            payload = f[f"replicas/{key}"][()].tobytes()
            with tempfile.NamedTemporaryFile(suffix=".dc", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                tmp_path.write_bytes(payload)
                containers.append(reader_cls.read(str(tmp_path)))
            finally:
                tmp_path.unlink(missing_ok=True)
    return history, containers, meta
