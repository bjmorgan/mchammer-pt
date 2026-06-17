"""Checkpoint and resume machinery.

This module owns the serialisation side of `mchammer-pt`'s
checkpoint format: identity hashes for the CE and ensemble kwargs,
the `CheckpointWriter` `CycleCallback`, and the writer/reader
helpers used by every `BaseParallelTempering` subclass that
supports checkpoint and resume (`CanonicalParallelTempering` and
`WangLandauParallelTempering` as of schema version ``"5"``).

The on-disk schema (version ``"5"``) is HDF5 with these top-level
groups: ``meta`` (run metadata as attrs; six shared keys —
``schema_version``, ``block_size``, ``random_seed``,
``ce_identity``, ``ensemble_cls_fqn``, ``ensemble_kwargs_hash`` —
plus ladder-specific keys contributed by each orchestrator
subclass via ``_checkpoint_meta()``: ``temperatures`` for canonical
PT, ``windows`` + ``energy_spacing`` + ``flatness_mode`` +
``merge_cadence`` + ``walkers_per_window`` for REWL);
``exchanges`` (per-cycle history arrays); ``replicas`` (one opaque
tarball per walker, the native mchammer ``BaseDataContainer``
format — flat in window-major / walker-minor order, length
``sum(walkers_per_window)``); ``orchestrator`` (the exchange-proposal
RNG state, the replica-label permutation, and — for REWL with
``walkers_per_window[g] > 1`` — one ``/orchestrator/window_groups/<g>/``
subgroup per multi-walker window carrying ``rng_state`` (group
exchange RNG) and ``phase`` (collective WL phase); W=1 windows omit
the subgroup entirely); and ``sites_by_species`` (one JSON dataset
per walker carrying the path-dependent
``ConfigurationManager._sites_by_species`` cache that bit-identical
resume requires alongside ``_last_state``).

Schema v4 is a hard break from v3 — v3 checkpoints are refused by
v4 readers with a message pointing at the last v3-capable release
(0.9.0). Schema v5 drops the per-window ``exchange_idx``. Single-
walker v4 checkpoints still load under a v5 reader; v4 multi-walker
checkpoints are refused, since their window-indexed replica labels
are incompatible with v5's walker-indexed labels.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from icet import ClusterExpansion
from mchammer.data_containers.base_data_container import BaseDataContainer

from .history import ExchangeHistory, MetaValue


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
    except (pickle.PicklingError, TypeError, AttributeError):
        # Common non-picklable cases: icet ClusterSpace/ClusterExpansion
        # (TypeError on copy), instances of locally-defined classes
        # (AttributeError on lookup), or unsupported objects
        # (PicklingError). Let MemoryError, RecursionError, and
        # KeyboardInterrupt propagate — they're real failures, not
        # "kwargs aren't stably hashable".
        return ""
    return hashlib.sha256(payload).hexdigest()


def _validate_kwargs_hash(
    path: Path | str,
    meta: dict[str, MetaValue],
    ensemble_kwargs: Mapping[str, Any] | None,
    caller: str,
    allow_mismatch: bool = False,
) -> None:
    """Resume-side guard for the ensemble-kwargs hash.

    Hard-error on a real mismatch (both sides hashed cleanly and
    differ). When either side returned the unpicklable-kwargs
    sentinel ``""``, the hash carries no information and the guard
    cannot enforce identity — emit a `UserWarning` rather than
    silently skipping, so a user resuming with materially different
    kwargs sees a signal that bit-identical resume isn't guaranteed.

    When ``allow_mismatch`` is True, a real mismatch is downgraded from a
    hard error to a `UserWarning` and execution proceeds; the sentinel
    (unhashable) case is unchanged. This backs the ``allow_kwargs_mismatch``
    flag on the resume/measure entry points, for continuing a run across
    software environments where the pickle of identical move objects
    differs.
    """
    expected = _compute_ensemble_kwargs_hash(ensemble_kwargs)
    saved = meta.get("ensemble_kwargs_hash", "")
    if expected and saved and expected != saved:
        if not allow_mismatch:
            raise ValueError(
                f"{path}: ensemble_kwargs hash mismatch. {caller} was "
                f"called with kwargs that hash differently from the "
                f"checkpoint."
            )
        warnings.warn(
            f"{path}: ensemble_kwargs hash mismatch, but {caller} was "
            f"called with allow_kwargs_mismatch=True, so the "
            f"kwargs-identity guard is bypassed. Only this check is "
            f"relaxed; CE identity and ensemble_cls are still enforced. "
            f"This is intended for resuming across software environments "
            f"(differing Python, numpy, or platform) where the pickle of "
            f"identical move objects differs. Bit-identical continuation "
            f"is not guaranteed.",
            UserWarning,
            stacklevel=3,
        )
    elif not expected or not saved:
        side = "the supplied" if not expected else "the checkpoint's"
        warnings.warn(
            f"{path}: {side} ensemble_kwargs are not stably "
            f"hashable (typically because they contain icet "
            f"ClusterSpace, ClusterExpansion, or similar "
            f"non-picklable objects). The kwargs-identity guard "
            f"is being skipped; if {caller} was called with "
            f"materially different kwargs from the original run, "
            f"the resumed trajectory will diverge silently.",
            UserWarning,
            stacklevel=3,
        )


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


def _has_window_groups(path: Path | str) -> bool:
    """Whether the checkpoint at `path` carries a ``window_groups`` subgroup.

    A present ``/orchestrator/window_groups/`` group marks a multi-walker
    REWL checkpoint. The writer omits the group entirely for all-W=1 runs,
    so absence means single-walker.
    """
    with h5py.File(Path(path), "r") as f:
        return (
            "orchestrator" in f and "window_groups" in f["orchestrator"]
        )


def _validate_wl_schema_version(
    path: Path | str, schema_version: object
) -> None:
    """Gate a REWL checkpoint by its on-disk ``schema_version``.

    Schema ``"5"`` is the current format. Schema ``"4"`` is accepted only
    for single-walker runs: a ``"4"`` checkpoint that carries a
    ``window_groups`` subgroup is an old multi-walker file whose
    window-indexed replica labels are incompatible with the walker-indexed
    labels schema 5 writes, and is refused. Any other version is
    refused with a pointer at the last v3-capable release.

    Args:
        path: checkpoint path, for error messages.
        schema_version: the ``schema_version`` attribute read from ``/meta``.

    Raises:
        ValueError: the version is unsupported, or a ``"4"`` multi-walker
            checkpoint is presented.
    """
    if schema_version == "5":
        return
    if schema_version == "4":
        if _has_window_groups(path):
            raise ValueError(
                f"{path}: checkpoint is schema 4 with multi-walker windows, "
                f"whose label layout is incompatible with this version; "
                f"regenerate the run with the current version."
            )
        return
    raise ValueError(
        f"{path}: unsupported schema_version {schema_version!r}; this "
        f"mchammer-pt understands '4' and '5' only. For v3 checkpoints, "
        f"resume with mchammer-pt 0.9.0 or earlier."
    )


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


def _read_window_groups(
    path: Path | str,
    containers: Sequence[Any] | None = None,
) -> list[dict[str, Any] | None]:
    """Read per-window group-level state from a v5 checkpoint.

    Args:
        path: HDF5 file written by `_write_checkpoint`.
        containers: optional flat list of `BaseDataContainer` instances
            already deserialised via `history.read_hdf5`, in window-major /
            walker-minor order (length M = sum of walkers_per_window).
            When supplied, the phase consistency check reads
            ``_last_state["phase"]`` directly from these in-memory
            containers, avoiding the cost of re-deserialising the
            replica tarballs. When omitted, the phase check is skipped
            (used by hand-crafted unit fixtures that have no
            ``/replicas/`` payloads to validate against).

    Returns:
        One entry per window. Each entry is a dict with keys
        ``rng_state`` (JSON str) and ``phase`` (str) when
        ``/orchestrator/window_groups/<g>/`` exists; ``None`` when the
        subgroup is absent (the W=1 convention).

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if ``/meta``, ``/meta/walkers_per_window``, or
            ``/orchestrator`` is missing — sign of a pre-v4 file or
            one that was not written as a checkpoint.
        ValueError: if `containers` is supplied and the on-disk group
            phase for any window disagrees with the
            ``_last_state["phase"]`` of any of its walkers, indicating
            a corrupted checkpoint file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with h5py.File(path, "r") as f:
        if (
            "meta" not in f
            or "walkers_per_window" not in f["meta"].attrs
            or "orchestrator" not in f
        ):
            raise KeyError(
                f"{path}: missing /meta/walkers_per_window or "
                f"/orchestrator; not a WL checkpoint (canonical PT "
                f"v4 checkpoints legitimately lack walkers_per_window)."
            )
        wpw = np.asarray(f["meta"].attrs["walkers_per_window"])
        n_windows = len(wpw)
        wg_parent = f["orchestrator"].get("window_groups")
        out: list[dict[str, Any] | None] = []
        for g in range(n_windows):
            if wg_parent is None or str(g) not in wg_parent:
                out.append(None)
                continue
            sub = wg_parent[str(g)]
            rng_raw = sub["rng_state"][()]
            phase_raw = sub["phase"][()]
            out.append({
                "rng_state": (
                    rng_raw.decode("utf-8")
                    if isinstance(rng_raw, bytes)
                    else str(rng_raw)
                ),
                "phase": (
                    phase_raw.decode("utf-8")
                    if isinstance(phase_raw, bytes)
                    else str(phase_raw)
                ),
            })

    if containers is None:
        return out
    expected_m = int(sum(wpw))
    if len(containers) != expected_m:
        raise ValueError(
            f"{path}: container-count mismatch — "
            f"sum(walkers_per_window) = {expected_m} but caller passed "
            f"{len(containers)} containers; cannot validate phase "
            f"consistency."
        )
    flat = 0
    for g, entry in enumerate(out):
        nw = int(wpw[g])
        if entry is not None:
            group_phase = entry["phase"]
            for w in range(nw):
                walker_phase = containers[flat + w]._last_state.get("phase")
                if walker_phase != group_phase:
                    raise ValueError(
                        f"{path}: phase consistency failure at "
                        f"window {g} walker {w}: group phase "
                        f"{group_phase!r} vs walker _last_state "
                        f"phase {walker_phase!r}; checkpoint is "
                        f"corrupted."
                    )
        flat += nw
    return out


def _serialise_rng_state(rng: np.random.Generator) -> str:
    """JSON-encode `rng.bit_generator.state` for HDF5 round-trip.

    `np.random.default_rng().bit_generator.state` is a small nested
    dict of int64s that JSON round-trips cleanly. Using JSON keeps
    the checkpoint schema free of pickle on the read path.
    """
    return json.dumps(rng.bit_generator.state)


def _write_checkpoint(pt: object, path: Path | str) -> None:
    """Write a full checkpoint of `pt` to `path` atomically.

    Used by every orchestrator's ``save_checkpoint`` method, the
    ``data_container_file=`` write path inside ``run()``, and
    `CheckpointWriter`. `pt` must be a `BaseParallelTempering`
    subclass that:

    - carries the identity attributes set during construction
      (`_ce_identity`, `_ensemble_cls_fqn`, `_ensemble_kwargs_hash`,
      `_random_seed`, `_block_size`);
    - carries the live orchestrator state (`_history`,
      `_replica_labels`, `_rng`);
    - implements `_checkpoint_meta()` returning any ladder-specific
      keys (canonical PT contributes ``temperatures``; REWL
      contributes ``windows``, ``energy_spacing``, ``flatness_mode``,
      ``merge_cadence``, and ``walkers_per_window``).

    Requires `run()` to have been called at least once, so that
    each replica's `_last_state` is populated and the on-disk
    container round-trips through ``_restart_ensemble``.

    The function packs these into the schema-``"5"`` HDF5 layout
    described in this module's docstring.
    """
    from .history import write_hdf5

    if pt._history is None:  # type: ignore[attr-defined]
        raise RuntimeError(
            "save_checkpoint requires run() to have been called at "
            "least once; the per-replica data containers do not have "
            "a populated `_last_state` until a run completes."
        )
    meta: dict[str, MetaValue] = {
        "schema_version": "5",
        "block_size": int(pt._block_size),  # type: ignore[attr-defined]
        "random_seed": int(pt._random_seed),  # type: ignore[attr-defined]
        "ce_identity": pt._ce_identity,  # type: ignore[attr-defined]
        "ensemble_cls_fqn": pt._ensemble_cls_fqn,  # type: ignore[attr-defined]
        "ensemble_kwargs_hash": pt._ensemble_kwargs_hash,  # type: ignore[attr-defined]
    }
    meta.update(pt._checkpoint_meta())  # type: ignore[attr-defined]
    orchestrator_state: dict[str, np.ndarray | str] = {
        "replica_labels": pt._replica_labels.copy(),  # type: ignore[attr-defined]
        "rng_state": _serialise_rng_state(pt._rng),  # type: ignore[attr-defined]
    }
    # Refresh per-replica `_last_state` (populating the fields
    # `_restart_ensemble` reads on resume) and capture the additional
    # state required for bit-identical continuation. The pool's
    # `snapshot_for_checkpoint()` is the cross-pool entry point — it
    # works for both `SerialPool` (in-process) and `ProcessPool`
    # (round-trips to each worker). Snapshot runs before
    # `data_containers()` because the snapshot side-effect populates
    # each container's `_last_state`, which mchammer's
    # `_restart_ensemble` reads on resume.
    snapshot = pt._pool.snapshot_for_checkpoint()  # type: ignore[attr-defined]
    if isinstance(snapshot, dict):
        # WL pools (v4 shape): per_walker + group_state.
        replica_extra = snapshot["per_walker"]
        window_groups = snapshot["group_state"]
    else:
        # Canonical PT pools still return a flat list.
        replica_extra = snapshot
        window_groups = None
    write_hdf5(
        Path(path),
        history=pt._history.truncated_to(pt.cycles_in_segment),  # type: ignore[attr-defined]
        replica_containers=pt._pool.data_containers(),  # type: ignore[attr-defined]
        meta=meta,
        orchestrator_state=orchestrator_state,
        replica_extra=replica_extra,
        window_groups=window_groups,
    )


def completed_cycles(
    containers: Sequence[BaseDataContainer],
    block_size: int,
) -> int:
    """Completed REWL cycles inferred from the restored walker MC steps.

    In REWL every active walker advances one block of ``block_size`` MC
    trial steps per cycle, so a walker's restored ``step`` equals
    ``cycles_run * block_size``. Walkers that converge early stop
    advancing (icet's ``_terminate_sampling`` short-circuit) and freeze
    at a lower step, so the orchestrator's completed-cycle count is the
    ``max`` across walkers, not any single walker's value. A walker that
    converges mid-block in the 1/t phase (when ``observer_interval <
    block_size``) freezes at an ``observer_interval`` boundary that is
    not a ``block_size`` boundary; floor division over the ``max``
    tolerates this legitimate off-block frozen step without requiring
    each walker's step to be an exact multiple.

    The step is read from each container's ``_last_state["last_step"]``
    -- the field ``WangLandauReplica`` restores on resume -- so the count
    is correct for padded legacy checkpoints and cumulative across
    chained resumes, unlike the per-cycle history length.

    Args:
        containers: per-walker data containers, e.g. as returned by
            :func:`mchammer_pt.read_hdf5`. Must be non-empty.
        block_size: MC trial steps per walker per cycle (the checkpoint's
            ``meta["block_size"]``).

    Returns:
        ``max(step) // block_size`` across all walkers.

    Raises:
        ValueError: if ``block_size < 1`` or ``containers`` is empty.
    """
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    if not containers:
        raise ValueError("completed_cycles requires at least one container")
    steps = [int(c._last_state["last_step"]) for c in containers]
    return max(steps) // block_size


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
