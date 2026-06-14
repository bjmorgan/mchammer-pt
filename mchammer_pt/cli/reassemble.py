"""Reassemble the window-subset pieces of one split REWL run.

A large REWL run can be split across several jobs/nodes, each running a
contiguous, complementary subset of the windows (overlapping its
neighbour at the boundary exactly as the windows of a single run do).
``reassemble`` takes the resulting checkpoint pieces of **one** run (one
seed), unions their per-window Wang-Landau data containers by window, and
writes a single stitch-ready artifact representing the complete run.

The artifact is analysis-only: it carries the per-window containers plus
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


def _union(
    pieces: list[Piece],
) -> tuple[
    list[WindowKey],
    dict[WindowKey, list[BaseDataContainer]],
    dict[str, MetaValue],
]:
    """Group containers by window across pieces and build the output meta.

    (Guards are added in a later task.)
    """
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
