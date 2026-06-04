"""Multi-run (multi-seed) merging for ``mchammer-pt-stitch``.

Combines N independent run checkpoints into one consensus density of
states: each window is merged across runs (via ``merge_entropies``, so a
checkpoint passed more than once is weighted by its multiplicity -- a
bootstrap draw), then the merged per-window curves are stitched once by
the unchanged ``stitch_entropy``. HDF5 reading lives in the CLI; the
merge helper and ``stitch_entropy`` stay HDF5-agnostic.
"""
from __future__ import annotations

from pathlib import Path

from mchammer.data_containers.base_data_container import BaseDataContainer

from mchammer_pt.cli.stitch import WindowKey, _load_from_checkpoint
from mchammer_pt.wl_coordinator import merge_entropies


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
            (propagated from :func:`mchammer_pt.wl_coordinator.merge_entropies`).
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
    return {
        key: merge_entropies([run[key] for run in per_run_windows])
        for key in keys
    }


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
