"""Checkpoint and resume machinery.

This module owns the serialisation side of `mchammer-pt`'s
checkpoint format: identity hashes for the CE and ensemble kwargs,
the `CheckpointWriter` `CycleCallback`, and the writer/reader
helpers used by `CanonicalParallelTempering.save_checkpoint`,
`CanonicalParallelTempering.resume`, and the existing
`data_container_file=` write path.

Schema design lives in
`docs/superpowers/specs/2026-05-08-checkpoint-and-resume-design.md`.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from icet import ClusterExpansion  # type: ignore[import-untyped]


def _compute_ce_identity(cluster_expansion: ClusterExpansion) -> str:
    """SHA-256 hex digest of an icet `ClusterExpansion`.

    Builds a stable canonical byte representation from public
    `ClusterExpansion` attributes — orbit topology and parameters
    via `to_dataframe().to_csv()`, plus chemistry, cutoffs, and the
    primitive structure (cell, atomic numbers, scaled positions,
    PBC) — and hashes it. Computed once at orchestrator construction
    time and stored on the orchestrator; every checkpoint write
    reuses the cached value.

    Note: `ClusterExpansion.write` would have been a more direct
    serialiser to hash, but its `.ce` tar archive embeds an `mtime`
    field that varies between calls, so its byte stream is not
    deterministic. The canonical form here uses only stable
    attributes.
    """
    df_bytes = cluster_expansion.to_dataframe().to_csv().encode("utf-8")
    chemical_symbols = repr(cluster_expansion.chemical_symbols).encode("utf-8")
    cutoffs = repr(list(cluster_expansion.cutoffs)).encode("utf-8")

    primitive = cluster_expansion.primitive_structure
    structure_bytes = b"|".join(
        [
            primitive.cell.array.tobytes(),
            primitive.numbers.tobytes(),
            primitive.get_scaled_positions().tobytes(),
            primitive.pbc.tobytes(),
        ]
    )

    digest = hashlib.sha256()
    for chunk in (df_bytes, chemical_symbols, cutoffs, structure_bytes):
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def _compute_ensemble_kwargs_hash(
    ensemble_kwargs: Mapping[str, Any] | None,
) -> str:
    """Best-effort stable hash of ensemble kwargs.

    Returns the SHA-256 hex digest of `pickle.dumps(sorted(items))`
    when the kwargs serialise cleanly, or the sentinel ``""``
    otherwise. Any pickling failure (non-picklable values like icet
    `ClusterSpace`, instances of locally-defined classes, etc.) falls
    back to the sentinel; the resume-time identity check treats a
    sentinel value on either side as "kwargs identity unknown, skip
    the check".

    Pickle is used purely as a stable serialiser for hashing — the
    bytes are not stored anywhere and never unpickled.

    The empty dict and `None` produce the same hash so that
    `ensemble_kwargs=None` and `ensemble_kwargs={}` are
    interchangeable from the user's perspective.
    """
    canonical = dict(ensemble_kwargs) if ensemble_kwargs else {}
    try:
        # `sorted(canonical.items())` makes the bytes order-independent
        # so callers passing the same kwargs in different insertion
        # orders get matching hashes.
        payload = pickle.dumps(sorted(canonical.items()))
    except Exception:
        return ""
    return hashlib.sha256(payload).hexdigest()


def _read_orchestrator_state(path: Path | str) -> dict[str, np.ndarray | str]:
    """Read the orchestrator-level state from a checkpoint file.

    Reads the ``/orchestrator/replica_labels`` and
    ``/orchestrator/rng_state`` datasets from a file written by
    `write_hdf5` with ``orchestrator_state=`` populated.

    Returns:
        Dict with keys ``replica_labels`` (int64 numpy array) and
        ``rng_state`` (JSON string).

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if the file does not have an ``/orchestrator/``
            group with both required datasets — typically a sign
            the file was written by a pre-checkpoint version of
            ``write_hdf5`` and is therefore not a valid resume
            source.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with h5py.File(path, "r") as f:
        if "orchestrator" not in f:
            raise KeyError(
                f"{path}: missing required '/orchestrator/' group. "
                f"File was not written as a checkpoint and cannot be "
                f"used for resume."
            )
        group = f["orchestrator"]
        for name in ("replica_labels", "rng_state"):
            if name not in group:
                raise KeyError(
                    f"{path}: missing 'orchestrator/{name}' dataset."
                )
        replica_labels = np.array(group["replica_labels"])
        # h5py stores Python str datasets as variable-length bytes;
        # decode on read.
        rng_state_raw = group["rng_state"][()]
        rng_state = (
            rng_state_raw.decode("utf-8")
            if isinstance(rng_state_raw, bytes)
            else str(rng_state_raw)
        )
    return {"replica_labels": replica_labels, "rng_state": rng_state}


def _serialise_rng_state(rng: np.random.Generator) -> str:
    """JSON-encode `rng.bit_generator.state` for HDF5 round-trip.

    `np.random.default_rng().bit_generator.state` is a small nested
    dict of int64s that JSON round-trips cleanly. Using JSON keeps
    the checkpoint schema free of pickle on the read path.
    """
    return json.dumps(rng.bit_generator.state)


def _write_checkpoint(pt: object, path: Path | str) -> None:
    """Write a full checkpoint of `pt` to `path` atomically.

    Used by `CanonicalParallelTempering.save_checkpoint`,
    `CheckpointWriter`, and the `data_container_file=` write path.
    `pt` must be a `CanonicalParallelTempering` instance whose
    `run()` has been called at least once.

    The function reads the identity hashes the orchestrator cached
    at construction (`_ce_identity`, `_ensemble_cls_fqn`,
    `_ensemble_kwargs_hash`, `_random_seed`) and the live state
    (`_history`, `_replica_labels`, `_rng`) and packs them into the
    HDF5 schema documented in
    ``docs/superpowers/specs/2026-05-08-checkpoint-and-resume-design.md``.
    """
    from .history import write_hdf5

    if pt._history is None:  # type: ignore[attr-defined]
        raise RuntimeError(
            "save_checkpoint requires run() to have been called at "
            "least once; the per-replica data containers do not have "
            "a populated `_last_state` until a run completes."
        )
    meta: dict[str, Any] = {
        "schema_version": "1",
        "temperatures": pt._temperatures,  # type: ignore[attr-defined]
        "block_size": int(pt._block_size),  # type: ignore[attr-defined]
        "random_seed": int(pt._random_seed),  # type: ignore[attr-defined]
        "ce_identity": pt._ce_identity,  # type: ignore[attr-defined]
        "ensemble_cls_fqn": pt._ensemble_cls_fqn,  # type: ignore[attr-defined]
        "ensemble_kwargs_hash": pt._ensemble_kwargs_hash,  # type: ignore[attr-defined]
    }
    orchestrator_state: dict[str, np.ndarray | str] = {
        "replica_labels": pt._replica_labels.copy(),  # type: ignore[attr-defined]
        "rng_state": _serialise_rng_state(pt._rng),  # type: ignore[attr-defined]
    }
    write_hdf5(
        Path(path),
        history=pt._history,  # type: ignore[attr-defined]
        replica_containers=pt._pool.data_containers(),  # type: ignore[attr-defined]
        meta=meta,
        orchestrator_state=orchestrator_state,
    )
