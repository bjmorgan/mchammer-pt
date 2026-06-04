"""Multi-run (multi-seed) merging for ``mchammer-pt-stitch``.

Combines N independent run checkpoints into one consensus density of
states: each window is merged across runs (via ``merge_entropies``, so a
checkpoint passed more than once is weighted by its multiplicity), then
the merged per-window curves are stitched once by the unchanged
``stitch_entropy``. HDF5 reading lives in the CLI; the
merge helper and ``stitch_entropy`` stay HDF5-agnostic.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from mchammer.data_containers.base_data_container import BaseDataContainer

from mchammer_pt.analysis.dos import stitch_entropy
from mchammer_pt.cli.stitch import (
    WindowKey,
    _format_window_summary,
    _get_window_params,
    _load_from_checkpoint,
    _select_window_keys,
    _trim_entropy_bins,
)
from mchammer_pt.wl_coordinator import merge_entropies
from mchammer_pt.wl_result import WindowResult


def merge_runs_per_window(
    per_run_windows: list[dict[WindowKey, dict[int, float]]],
) -> dict[WindowKey, dict[int, float]]:
    """Merge per-window entropy curves across independent runs.

    Each element of ``per_run_windows`` is one run's mapping from window
    key to that window's entropy curve (``{bin_index: ln g}``). All runs
    must expose the same set of window keys. For each window the per-run
    curves are combined with
    :func:`mchammer_pt.wl_coordinator.merge_entropies`, which aligns
    independent estimates on their common bins and averages bin-wise over
    the runs that cover each bin; a run passed more than once therefore
    contributes with its multiplicity.

    Args:
        per_run_windows: per-run ``window key -> entropy curve`` mappings.
            May contain repeated runs; repeats are weighted accordingly.

    Returns:
        ``window key -> merged entropy curve``, one entry per window key.

    Raises:
        ValueError: if ``per_run_windows`` is empty or the runs do not all
            expose the same set of window keys.
        RuntimeError: if, within a window, the runs share no common bin
            (propagated from :func:`mchammer_pt.wl_coordinator.merge_entropies`,
            re-raised with the offending window key so the caller can name the
            energy band).
    """
    if not per_run_windows:
        raise ValueError(
            "per_run_windows is empty; need at least one run"
        )
    keys = set(per_run_windows[0])
    for i, run in enumerate(per_run_windows[1:], start=1):
        if set(run) != keys:
            raise ValueError(
                f"run {i} window keys differ from run 0: "
                f"run 0 has {keys}, run {i} has {set(run)}"
            )
    merged: dict[WindowKey, dict[int, float]] = {}
    for key in keys:
        try:
            merged[key] = merge_entropies([run[key] for run in per_run_windows])
        except RuntimeError as e:
            raise RuntimeError(f"window {key}: {e}") from e
    return merged


def _load_runs(
    paths: list[Path],
) -> tuple[list[list[BaseDataContainer]], str | None]:
    """Read one checkpoint per run. Returns ``(runs, error_message)``.

    Each path is read as an independent run via the same checkpoint
    reader the single-run path uses. On the first read error the runs
    list is empty and the message is non-None.
    """
    runs: list[list[BaseDataContainer]] = []
    for path in paths:
        raw, err = _load_from_checkpoint(path)
        if err is not None:
            return [], err
        runs.append(raw)
    return runs, None


def run_multi_run(args: argparse.Namespace) -> int:
    """Stitch N run checkpoints into one consensus DOS.

    Loads each checkpoint as an independent run, validates that every run
    covers the same windows on the same grid, merges each window across
    runs (multiplicity-weighted), then stitches once. Writes the
    consensus ``energy``/``entropy`` CSV and prints per-pair overlap
    standard deviations to stderr as a join diagnostic.

    A run that produces no data for a kept window (its walker never entered)
    is excluded from that window's merge and a note is printed to stderr; the
    window's consensus is then formed from the remaining runs.
    """
    # This mirrors the validate/group/select/trim/stitch/write spine of the
    # single-run path in ``stitch.main``. The two are intentionally not
    # unified: the multi-run path additionally validates window-key equality
    # across runs, merges each window across runs, tolerates a run missing a
    # window, and reports per-pair overlap to stderr -- threading those
    # differences through a shared helper would obscure both flows.
    if len(args.inputs) < 2:
        print(
            "error: --multi-run needs at least two run checkpoints",
            file=sys.stderr,
        )
        return 2

    runs, err = _load_runs(args.inputs)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2

    spacings: set[float | None] = set()
    per_run_by_window: list[dict[WindowKey, list[BaseDataContainer]]] = []
    for run_idx, dcs in enumerate(runs):
        by_window: dict[WindowKey, list[BaseDataContainer]] = defaultdict(list)
        for dc_idx, dc in enumerate(dcs):
            params, perr = _get_window_params(
                dc, f"run {run_idx} replica {dc_idx}"
            )
            if perr is not None:
                print(f"error: {perr}", file=sys.stderr)
                return 2
            spacings.add(params["energy_spacing"])
            key = (params["energy_limit_left"], params["energy_limit_right"])
            by_window[key].append(dc)
        per_run_by_window.append(by_window)

    if len(spacings) != 1:
        print(
            f"error: runs disagree on energy_spacing: "
            f"{sorted(spacings, key=lambda v: (v is None, v))}",
            file=sys.stderr,
        )
        return 2
    energy_spacing = float(next(iter(spacings)))  # type: ignore[arg-type]

    key_sets = [set(bw) for bw in per_run_by_window]
    if any(ks != key_sets[0] for ks in key_sets[1:]):
        print(
            "error: runs have different window keys; every run must cover "
            "the same windows on the same grid",
            file=sys.stderr,
        )
        return 2

    by_window0 = per_run_by_window[0]
    if len(by_window0) < 2:
        print(
            f"error: stitching needs at least two distinct windows; got "
            f"{len(by_window0)}",
            file=sys.stderr,
        )
        return 2

    if (
        args.emin is not None
        and args.emax is not None
        and args.emin >= args.emax
    ):
        print(
            f"error: --emin ({args.emin}) must be < --emax ({args.emax})",
            file=sys.stderr,
        )
        return 2

    kept_keys, err = _select_window_keys(by_window0, args.windows)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2

    per_run_windows: list[dict[WindowKey, dict[int, float]]] = []
    for run_idx, by_window in enumerate(per_run_by_window):
        curves: dict[WindowKey, dict[int, float]] = {}
        for lo, hi in kept_keys:
            result = WindowResult(
                energy_limit_left=(
                    float(lo) if lo is not None else float("-inf")
                ),
                energy_limit_right=(
                    float(hi) if hi is not None else float("inf")
                ),
                energy_spacing=energy_spacing,
                containers=tuple(by_window[(lo, hi)]),
            )
            df = result.get_entropy(fill_factor_limit=args.fill_factor_limit)
            if df is None or df.empty:
                curves[(lo, hi)] = {}
                print(
                    f"note: run {run_idx} contributed no data to window "
                    f"({lo}, {hi}); excluded from its merge",
                    file=sys.stderr,
                )
            else:
                curves[(lo, hi)] = {
                    int(b): float(v)
                    for b, v in zip(df.index, df["entropy"], strict=True)
                }
        per_run_windows.append(curves)

    try:
        merged = merge_runs_per_window(per_run_windows)
    except (ValueError, RuntimeError) as e:
        print(f"error: merging runs failed: {e}", file=sys.stderr)
        return 2

    per_window: list[pd.DataFrame] = []
    surviving_keys: list[WindowKey] = []
    for lo, hi in kept_keys:
        curve = merged[(lo, hi)]
        if not curve:
            print(
                f"error: window ({lo}, {hi}) produced no entropy data "
                f"across {len(runs)} run(s)",
                file=sys.stderr,
            )
            return 2
        bins = sorted(curve)
        df = pd.DataFrame({
            "energy": [b * energy_spacing for b in bins],
            "entropy": [curve[b] for b in bins],
        })
        df = _trim_entropy_bins(df, args.emin, args.emax)
        if df.empty:
            continue
        per_window.append(df)
        surviving_keys.append((lo, hi))

    filters_active = (
        args.windows is not None
        or args.emin is not None
        or args.emax is not None
    )
    if len(per_window) < 2:
        print(
            f"error: filters left fewer than 2 windows for stitching "
            f"({_format_window_summary(surviving_keys, len(by_window0))})",
            file=sys.stderr,
        )
        return 2

    try:
        stitched, errors = stitch_entropy(per_window, energy_spacing)
    except ValueError as e:
        print(f"error: stitching failed: {e}", file=sys.stderr)
        return 2

    try:
        stitched.to_csv(args.output, index=False)
    except OSError as e:
        print(f"error: could not write {args.output}: {e}", file=sys.stderr)
        return 2

    for pair, std in errors.items():
        print(f"overlap std window {pair}: {std:.3g}", file=sys.stderr)

    # A stitched window may have been merged from fewer than all runs (a seed
    # whose walker never entered it, excluded above). Reflect that on stdout so
    # a reader of "merged N runs" is not misled into assuming every window used
    # every run; the per-window detail is in the stderr notes above.
    thin_windows = sum(
        1
        for key in surviving_keys
        if sum(1 for rc in per_run_windows if rc[key]) < len(runs)
    )

    msg = (
        f"wrote {args.output} ({len(stitched)} rows; "
        f"merged {len(runs)} runs)"
    )
    if thin_windows:
        msg += (
            f"; {thin_windows} of {len(surviving_keys)} windows merged "
            f"fewer than {len(runs)} runs (see stderr notes)"
        )
    if filters_active:
        msg += "; " + _format_window_summary(surviving_keys, len(by_window0))
    print(msg)
    return 0
