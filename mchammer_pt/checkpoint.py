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
import pickle
from collections.abc import Mapping
from typing import Any

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

    Returns the SHA-256 hex digest of `pickle.dumps(kwargs)` when
    the kwargs pickle cleanly, or the sentinel ``""`` otherwise
    (kwargs containing icet `ClusterSpace`, `ClusterExpansion`, or
    other non-picklable objects fall back to the sentinel).

    Pickle is used purely as a stable serialiser for hashing — the
    bytes are not stored anywhere and never unpickled. The hash
    feeds into the resume-time identity check, where a sentinel
    value on either side is treated as "kwargs identity unknown,
    skip the check".

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
    except (TypeError, pickle.PickleError):
        return ""
    return hashlib.sha256(payload).hexdigest()
