"""Reassemble the window-subset pieces of one split REWL run.

A large REWL run can be split across several jobs/nodes, each running a
contiguous, complementary subset of the windows (overlapping its
neighbour at the boundary exactly as the windows of a single run do).
``reassemble`` takes the resulting checkpoint pieces of **one** run (one
seed), unions their per-window Wang-Landau data containers by window, and
writes a single stitch-ready artefact representing the complete run.

The artefact is analysis-only: it carries the per-window containers plus
identity and self-description metadata, but omits the run-execution
metadata and the ``/orchestrator/`` + ``/sites_by_species/`` groups a
resumable checkpoint has, so ``WangLandauParallelTempering.resume``
refuses it. The existing ``mchammer-pt-stitch`` reads it unchanged.

Reassembly is per-seed and runs once per seed, before stitching and never
inside a bootstrap loop. It unions *complementary* windows only and errors
on a window-key collision (the same window appearing in two inputs),
pointing the user at ``mchammer-pt-stitch --multi-run`` for the distinct
operation of averaging identical windows across seeds.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from mchammer.data_containers.base_data_container import BaseDataContainer

from mchammer_pt.cli.stitch import WindowKey, _get_window_params
from mchammer_pt.history import (
    ExchangeHistory,
    MetaValue,
    read_hdf5,
    write_hdf5,
)
from mchammer_pt.wl import _windows_to_array

# One loaded checkpoint piece: (source label, meta dict, containers).
Piece = tuple[str, dict[str, MetaValue], list[BaseDataContainer]]


def _window_sort_key(key: WindowKey) -> tuple[float, float]:
    """Energy-ascending sort key for a window, treating open edges as
    infinities (``None`` left -> ``-inf``, ``None`` right -> ``+inf``)."""
    lo, hi = key
    lo_val = float("-inf") if lo is None else float(lo)
    hi_val = float("inf") if hi is None else float(hi)
    return lo_val, hi_val


def reassemble_pieces(
    pieces: list[Piece],
) -> tuple[list[BaseDataContainer], dict[str, MetaValue]]:
    """Union the per-window containers of one run's checkpoint pieces.

    Args:
        pieces: loaded checkpoint pieces of a single run, each a
            ``(source_label, meta, containers)`` tuple. ``source_label``
            names the file in error messages; ``meta`` is the checkpoint
            metadata; ``containers`` are the per-walker
            ``WangLandauDataContainer`` instances.

    Returns:
        ``(union_containers, out_meta)``. ``union_containers`` are all
        pieces' containers in window-ascending order. ``out_meta`` carries
        the verified ``ensemble_cls_fqn``, ``ce_identity``, and
        ``energy_spacing``, the reconstructed ``windows`` and
        ``walkers_per_window`` for the union, ``schema_version``, and a
        ``reassembled = True`` marker.
    """
    sorted_keys, by_window, out_meta = _union(pieces)
    union_containers: list[BaseDataContainer] = []
    for key in sorted_keys:
        union_containers.extend(by_window[key])
    return union_containers, out_meta


def _validate_pieces(pieces: list[Piece]) -> None:
    """Raise ``ValueError`` if the pieces cannot form one complete run.

    Owns the guards that fire where heterogeneous files first meet: at
    least two pieces; every piece a Wang-Landau checkpoint; a shared
    cluster-expansion identity; a shared system size (supercell); a shared
    energy spacing; and no window-key collision across pieces (the same
    window in two inputs). Grid/overlap/gap checks are left to
    ``stitch_entropy`` on the unioned result.
    """
    if len(pieces) < 2:
        raise ValueError(
            f"reassembly needs at least two checkpoint pieces; "
            f"got {len(pieces)}"
        )

    # Every piece is a Wang-Landau checkpoint.
    for label, meta, _ in pieces:
        fqn = str(meta.get("ensemble_cls_fqn", ""))
        if "WangLandau" not in fqn:
            raise ValueError(
                f"{label} is not a Wang-Landau checkpoint "
                f"(ensemble_cls_fqn={fqn!r}); reassembly applies to "
                f"REWL runs only"
            )

    # Identical cluster-expansion identity. The CE identity hash covers
    # only the primitive structure, so it is necessary but not sufficient;
    # the system-size guard below catches a shared CE on different cells.
    ref_label, ref_meta, _ = pieces[0]
    ref_ce = str(ref_meta.get("ce_identity", ""))
    for label, meta, _ in pieces[1:]:
        ce = str(meta.get("ce_identity", ""))
        if ce != ref_ce:
            raise ValueError(
                f"inputs disagree on cluster-expansion identity: "
                f"{ref_label} has {ref_ce[:12]}..., "
                f"{label} has {ce[:12]}...; all pieces must come from "
                f"the same cluster expansion"
            )

    # Identical system size (n_sc) across every container.
    size_label: dict[int, str] = {}
    for label, _, containers in pieces:
        for dc in containers:
            size_label.setdefault(len(dc.structure), label)
    if len(size_label) > 1:
        (n_a, label_a), (n_b, label_b) = sorted(size_label.items())[:2]
        raise ValueError(
            f"inputs disagree on system size: {label_a} has {n_a} sites, "
            f"{label_b} has {n_b} sites; all pieces must come from the "
            f"same supercell"
        )

    # Identical energy spacing.
    ref_spacing = float(ref_meta["energy_spacing"])
    for label, meta, _ in pieces[1:]:
        spacing = float(meta["energy_spacing"])
        if spacing != ref_spacing:
            raise ValueError(
                f"inputs disagree on energy_spacing: {ref_label} has "
                f"{ref_spacing}, {label} has {spacing}; reassembly requires "
                f"a common grid"
            )

    # No window-key collision across pieces. Within a piece, repeats of a
    # key are legitimate multi-walker data; the same key in two different
    # pieces is the collision reassembly refuses.
    seen: dict[WindowKey, str] = {}
    for label, _, containers in pieces:
        piece_keys: set[WindowKey] = set()
        for dc in containers:
            params, err = _get_window_params(dc, label)
            if err is not None:
                raise ValueError(err)
            piece_keys.add(
                (params["energy_limit_left"], params["energy_limit_right"])
            )
        for key in piece_keys:
            if key in seen:
                lo, hi = key
                raise ValueError(
                    f"window ({lo}, {hi}) appears in more than one input "
                    f"({seen[key]} and {label}); reassembly unions "
                    f"complementary windows and refuses to merge identical "
                    f"ones. To average independent runs of the same "
                    f"windows, use 'mchammer-pt-stitch --multi-run' instead."
                )
            seen[key] = label


def _union(
    pieces: list[Piece],
) -> tuple[
    list[WindowKey],
    dict[WindowKey, list[BaseDataContainer]],
    dict[str, MetaValue],
]:
    """Group containers by window across pieces and build the output meta."""
    _validate_pieces(pieces)
    by_window: dict[WindowKey, list[BaseDataContainer]] = defaultdict(list)
    for label, _, containers in pieces:
        for dc in containers:
            params, err = _get_window_params(dc, label)
            if err is not None:
                raise ValueError(err)
            key = (params["energy_limit_left"], params["energy_limit_right"])
            by_window[key].append(dc)

    sorted_keys = sorted(by_window, key=_window_sort_key)
    walkers_per_window = [len(by_window[k]) for k in sorted_keys]
    first_meta = pieces[0][1]
    out_meta: dict[str, MetaValue] = {
        "schema_version": "5",
        "ensemble_cls_fqn": str(first_meta["ensemble_cls_fqn"]),
        "ce_identity": str(first_meta["ce_identity"]),
        "energy_spacing": float(first_meta["energy_spacing"]),
        "windows": _windows_to_array(sorted_keys),
        "walkers_per_window": np.asarray(walkers_per_window, dtype=np.int32),
        "reassembled": True,
    }
    return sorted_keys, by_window, out_meta


def _load_piece(path: Path) -> tuple[Piece | None, str | None]:
    """Read one checkpoint piece. Returns ``(piece, error_message)`` with
    exactly one side non-``None``."""
    try:
        _, containers, meta = read_hdf5(path)
    except (OSError, KeyError, ValueError, TypeError) as e:
        return None, f"could not read checkpoint {path}: {e}"
    if not containers:
        return None, f"checkpoint {path} contains no replica data containers"
    return (str(path), meta, containers), None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mchammer-pt-reassemble",
        description=(
            "Reassemble the complementary window-subset pieces of one "
            "split REWL run (one seed) into a single stitch-ready, "
            "analysis-only checkpoint that mchammer-pt-stitch reads "
            "unchanged. Unions complementary windows and errors on a "
            "window-key collision (use 'mchammer-pt-stitch --multi-run' "
            "to average identical windows across seeds instead). The "
            "output is not resumable."
        ),
    )
    p.add_argument(
        "inputs", type=Path, nargs="+",
        help="The checkpoint pieces of one run (>= 2).",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("reassembled.h5"),
        help="Output HDF5 path. Default: reassembled.h5 (CWD).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if len(args.inputs) < 2:
        print(
            f"error: reassembly needs at least two checkpoint pieces; "
            f"got {len(args.inputs)}",
            file=sys.stderr,
        )
        return 2

    pieces: list[Piece] = []
    for path in args.inputs:
        piece, err = _load_piece(path)
        if err is not None:
            print(f"error: {err}", file=sys.stderr)
            return 2
        assert piece is not None  # err is None implies piece is set
        pieces.append(piece)

    try:
        containers, meta = reassemble_pieces(pieces)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    history = ExchangeHistory.empty(n_cycles=0, n_replicas=len(containers))
    try:
        write_hdf5(
            args.output,
            history=history,
            replica_containers=containers,
            meta=meta,
            orchestrator_state=None,
        )
    except OSError as e:
        print(f"error: could not write {args.output}: {e}", file=sys.stderr)
        return 2

    n_windows = int(len(np.asarray(meta["walkers_per_window"])))
    print(
        f"wrote {args.output} (reassembled {len(pieces)} pieces into "
        f"{n_windows} windows, {len(containers)} walkers)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
