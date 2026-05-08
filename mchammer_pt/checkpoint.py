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

from .history import ExchangeHistory


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


def _read_replica_extra(path: Path | str) -> list[dict[str, Any]]:
    """Read per-replica extra state (e.g. ``_sites_by_species``).

    Reads the ``/sites_by_species/<i>`` JSON datasets written by
    `write_hdf5` with ``replica_extra=`` populated. The returned list
    is in integer-id order, with one dict per replica carrying
    ``"sites_by_species"`` reconstructed (JSON's string dict keys
    converted back to ``int`` so the structure round-trips to its
    original ``list[dict[int, list[int]]]`` shape).

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if the file does not have a ``/sites_by_species/``
            group — typically a sign the file was written by a
            pre-checkpoint version of `write_hdf5` and is therefore
            not a valid resume source for bit-identical continuation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with h5py.File(path, "r") as f:
        if "sites_by_species" not in f:
            raise KeyError(
                f"{path}: missing required '/sites_by_species/' group. "
                f"File was not written as a checkpoint and cannot be "
                f"used for resume."
            )
        group = f["sites_by_species"]
        keys = sorted(group.keys(), key=int)
        extras: list[dict[str, Any]] = []
        for key in keys:
            payload_raw = group[key][()]
            payload = (
                payload_raw.decode("utf-8")
                if isinstance(payload_raw, bytes)
                else str(payload_raw)
            )
            decoded = json.loads(payload)
            sites_by_species = [
                {int(species): list(sites) for species, sites in sublattice.items()}
                for sublattice in decoded
            ]
            extras.append({"sites_by_species": sites_by_species})
    return extras


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
        "schema_version": "2",
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
    # Refresh per-replica `_last_state` (populating the four fields
    # `_restart_ensemble` reads on resume) and capture the additional
    # state required for bit-identical continuation. The pool's
    # `snapshot_for_checkpoint()` is the cross-pool entry point — it
    # works for both `SerialPool` (in-process) and `ProcessPool`
    # (round-trips to each worker). Snapshot runs before
    # `data_containers()` because the snapshot side-effect populates
    # each container's `_last_state`, which mchammer's
    # `_restart_ensemble` reads on resume.
    replica_extra = pt._pool.snapshot_for_checkpoint()  # type: ignore[attr-defined]
    write_hdf5(
        Path(path),
        history=pt._history,  # type: ignore[attr-defined]
        replica_containers=pt._pool.data_containers(),  # type: ignore[attr-defined]
        meta=meta,
        orchestrator_state=orchestrator_state,
        replica_extra=replica_extra,
    )


class CheckpointWriter:
    """Periodic full-checkpoint writer.

    A built-in `CycleCallback` for crash-safe long PT runs. Writes
    the same payload `pt.save_checkpoint(path)` produces, atomically
    every ``interval`` cycles plus the final cycle. The file at
    ``path`` is overwritten on each emission, so on resume the user
    picks up from the most recent successful write.

    A failed write raises out of `on_cycle_end`. The orchestrator's
    cycle-callback fan-out propagates the exception, which aborts
    the run with the partial history preserved on `pt.history`.
    A full disk fails loud rather than silently losing checkpoints.

    Prefer `pt.attach_checkpoint_writer(path, interval=...)` over
    constructing this directly — the convenience method binds
    ``pt=self`` so the user does not have to repeat the orchestrator
    reference.

    Args:
        path: target file. Overwritten atomically on each emission.
        interval: emit every ``interval`` completed cycles. Must be
            ``>= 1``. The final cycle of every `run()` always emits.
        pt: the orchestrator this writer checkpoints. Required —
            `CheckpointWriter` reads identity hashes and live state
            off it on every emission.
    """

    def __init__(
        self,
        path: Path | str,
        interval: int = 1000,
        *,
        pt: object,
    ) -> None:
        if int(interval) < 1:
            raise ValueError(f"interval must be >= 1, got {interval!r}")
        self._path = Path(path)
        self._interval = int(interval)
        self._pt = pt

    def on_cycle_end(
        self,
        cycle: int,
        n_cycles: int,
        history: ExchangeHistory,
    ) -> None:
        is_interval_emission = (cycle + 1) % self._interval == 0
        is_final_emission = cycle == n_cycles - 1
        if not (is_interval_emission or is_final_emission):
            return
        _write_checkpoint(self._pt, self._path)
