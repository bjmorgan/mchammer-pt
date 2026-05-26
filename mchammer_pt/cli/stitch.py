"""Standalone Wang-Landau DOS stitcher CLI: ``mchammer-pt-stitch``.

Default input is a single mchammer-pt checkpoint HDF5 (the artefact
written by ``data_container_file=``, ``save_checkpoint``, or
``CheckpointWriter``); the CLI extracts per-walker
``WangLandauDataContainer`` instances via ``mchammer_pt.read_hdf5``,
which dispatches on the recorded ensemble class so WL payloads come
back as ``WangLandauDataContainer`` (with int bin keys restored).

With ``--containers``, the CLI instead reads two or more
``WangLandauDataContainer`` files passed positionally (each typically
the on-disk form written by ``WangLandauDataContainer.write``).

Containers are grouped by window using each one's
``energy_limit_left`` / ``energy_limit_right`` ensemble parameters,
walker-merged within each window via ``WindowResult.get_entropy()``,
and stitched via ``mchammer_pt.analysis.dos.stitch_entropy``. Handles
single- and multi-walker REWL output through one code path.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from mchammer.data_containers.base_data_container import BaseDataContainer
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt import read_hdf5
from mchammer_pt.analysis.dos import stitch_entropy
from mchammer_pt.wl_result import WindowResult

_REQUIRED_PARAMS = ("energy_spacing", "energy_limit_left", "energy_limit_right")


def _parse_window_indices(s: str) -> list[int]:
    """argparse type for ``--windows``: comma-separated 0-based ints."""
    try:
        return [int(x) for x in s.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--windows: expected comma-separated integers; got {s!r}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mchammer-pt-stitch",
        description=(
            "Stitch a Wang-Landau parallel-tempering run into a single "
            "density of states. By default reads one mchammer-pt "
            "checkpoint HDF5 file; with --containers reads two or more "
            "individual WangLandauDataContainer files. Containers that "
            "share the same (energy_limit_left, energy_limit_right) "
            "are walker-merged before stitching."
        ),
    )
    p.add_argument(
        "inputs", type=Path, nargs="+",
        help=(
            "Default: a single mchammer-pt checkpoint HDF5. With "
            "--containers: two or more WangLandauDataContainer files."
        ),
    )
    p.add_argument(
        "--containers", action="store_true",
        help=(
            "Treat positional inputs as individual "
            "WangLandauDataContainer files rather than a single "
            "mchammer-pt checkpoint."
        ),
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("dos.csv"),
        help="Output CSV path. Default: dos.csv (CWD).",
    )
    p.add_argument(
        "--fill-factor-limit", type=float, default=None,
        help=(
            "If given, each walker contributes the entropy recorded at "
            "the step when its fill factor first reached this limit."
        ),
    )
    p.add_argument(
        "--windows", type=_parse_window_indices, default=None,
        metavar="IDX[,IDX...]",
        help=(
            "Comma-separated 0-based window indices to keep "
            "(energy-sorted ascending). If omitted, keep all."
        ),
    )
    p.add_argument(
        "--emin", type=float, default=None, metavar="E_MIN",
        help=(
            "Drop bins at or below E_MIN from each surviving "
            "window before stitching (kept interval is open above E_MIN)."
        ),
    )
    p.add_argument(
        "--emax", type=float, default=None, metavar="E_MAX",
        help=(
            "Drop bins at or above E_MAX from each surviving "
            "window before stitching (kept interval is open below E_MAX)."
        ),
    )
    return p


def _load_from_checkpoint(
    path: Path,
) -> tuple[list[BaseDataContainer], str | None]:
    """Return (containers, error_message) -- exactly one is non-empty.

    `read_hdf5` dispatches on the recorded ensemble class, so WL
    checkpoints come back as `WangLandauDataContainer` instances
    with int-keyed `_last_state` already restored. Non-WL
    checkpoints are caught by the missing-parameter check in the
    caller.
    """
    try:
        _, raw, _ = read_hdf5(path)
    except (OSError, KeyError, ValueError, TypeError) as e:
        return [], f"could not read checkpoint {path}: {e}"
    if not raw:
        return [], f"checkpoint {path} contains no replica data containers"
    return raw, None


def _load_loose(
    paths: list[Path],
) -> tuple[list[BaseDataContainer], str | None]:
    """Return (containers, error_message) -- exactly one is non-empty."""
    dcs: list[BaseDataContainer] = []
    for path in paths:
        try:
            dcs.append(WangLandauDataContainer.read(path))
        except (OSError, ValueError, TypeError, KeyError) as e:
            return [], f"could not read {path}: {e}"
    return dcs, None


def _get_window_params(
    dc: BaseDataContainer, source: str,
) -> tuple[dict[str, float | None], str | None]:
    """Pull the required window parameters off a container.

    `source` is a human-readable identifier (file path, or "replica N"
    inside a checkpoint) used in the error message.
    """
    params: dict[str, float | None] = {}
    for key in _REQUIRED_PARAMS:
        try:
            params[key] = dc.ensemble_parameters[key]
        except KeyError:
            return {}, (
                f"{source} is missing required ensemble parameter "
                f"'{key}'; is this a Wang-Landau container?"
            )
    return params, None


def _select_window_keys(
    by_window: dict[tuple[float | None, float | None], list[BaseDataContainer]],
    windows_keep: list[int] | None,
) -> tuple[list[tuple[float | None, float | None]], str | None]:
    """Return the window keys to keep in energy-sorted order.

    Sorts ``by_window`` keys by ``energy_limit_left`` ascending (treating
    ``None`` as ``-inf``), with ``energy_limit_right`` (``None`` -> ``+inf``)
    as a tie-breaker so windows that share a left bound are ordered
    deterministically by their upper edge. If ``windows_keep`` is
    ``None``, returns all keys. Otherwise filters by the supplied
    0-based indices.

    Returns ``(keys, error_message)``. On error the keys list is empty.
    """
    def sort_key(k: tuple[float | None, float | None]) -> tuple[float, float]:
        lo, hi = k
        lo_val = float("-inf") if lo is None else float(lo)
        hi_val = float("inf") if hi is None else float(hi)
        return lo_val, hi_val

    ordered = sorted(by_window.keys(), key=sort_key)
    if windows_keep is None:
        return ordered, None

    n = len(ordered)
    bad = [i for i in windows_keep if i < 0 or i >= n]
    if bad:
        return [], (
            f"--windows index {bad[0]} out of range; discovered "
            f"{n} windows (valid: 0..{n - 1})"
        )
    keep = sorted(set(windows_keep))
    return [ordered[i] for i in keep], None


def _trim_entropy_bins(
    df: pd.DataFrame, emin: float | None, emax: float | None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with bins outside ``(emin, emax)`` dropped.

    The kept interval is open: a bin at exactly ``emin`` is dropped, and
    a bin at exactly ``emax`` is dropped. Either bound may be ``None``,
    in which case that side is not trimmed. The returned DataFrame may
    be empty.
    """
    mask = pd.Series(True, index=df.index)
    if emin is not None:
        mask &= df["energy"] > emin
    if emax is not None:
        mask &= df["energy"] < emax
    return df.loc[mask].reset_index(drop=True)


def _format_window_summary(
    kept_keys: list[tuple[float | None, float | None]],
    total: int,
) -> str:
    """Build the 'kept K of W windows: ...' fragment for the success line."""
    pairs = ", ".join(f"({lo}, {hi})" for lo, hi in kept_keys)
    return f"kept {len(kept_keys)} of {total} windows: {pairs}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.containers:
        if len(args.inputs) < 2:
            print(
                "error: --containers needs at least two "
                "WangLandauDataContainer files",
                file=sys.stderr,
            )
            return 2
        dcs, err = _load_loose(args.inputs)
        sources = [str(p) for p in args.inputs]
    else:
        if len(args.inputs) != 1:
            print(
                "error: checkpoint mode expects exactly one input file "
                "(use --containers to pass individual "
                "WangLandauDataContainer files instead)",
                file=sys.stderr,
            )
            return 2
        dcs, err = _load_from_checkpoint(args.inputs[0])
        sources = [
            f"checkpoint {args.inputs[0]} replica {i}" for i in range(len(dcs))
        ]
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2

    params_per_dc: list[dict[str, float | None]] = []
    for dc, src in zip(dcs, sources, strict=True):
        params, err = _get_window_params(dc, src)
        if err is not None:
            print(f"error: {err}", file=sys.stderr)
            return 2
        params_per_dc.append(params)

    spacings = {p["energy_spacing"] for p in params_per_dc}
    if len(spacings) != 1:
        print(
            f"error: containers disagree on energy_spacing: "
            f"{sorted(spacings, key=lambda v: (v is None, v))}",
            file=sys.stderr,
        )
        return 2
    energy_spacing = float(next(iter(spacings)))  # type: ignore[arg-type]

    by_window: dict[
        tuple[float | None, float | None], list[BaseDataContainer]
    ] = defaultdict(list)
    for dc, params in zip(dcs, params_per_dc, strict=True):
        key = (params["energy_limit_left"], params["energy_limit_right"])
        by_window[key].append(dc)

    if args.emin is not None and args.emax is not None and args.emin >= args.emax:
        print(
            f"error: --emin ({args.emin}) must be < --emax ({args.emax})",
            file=sys.stderr,
        )
        return 2

    if len(by_window) < 2:
        print(
            f"error: stitching needs at least two distinct windows; got "
            f"{len(by_window)} (all containers share the same "
            f"energy_limit_left/right)",
            file=sys.stderr,
        )
        return 2

    kept_keys, err = _select_window_keys(by_window, args.windows)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2

    per_window: list[pd.DataFrame] = []
    surviving_keys: list[tuple[float | None, float | None]] = []
    for lo, hi in kept_keys:
        containers = by_window[(lo, hi)]
        result = WindowResult(
            energy_limit_left=float(lo) if lo is not None else float("-inf"),
            energy_limit_right=float(hi) if hi is not None else float("inf"),
            energy_spacing=energy_spacing,
            containers=tuple(containers),
        )
        df = result.get_entropy(fill_factor_limit=args.fill_factor_limit)
        if df is None or df.empty:
            print(
                f"error: window ({lo}, {hi}) produced no entropy data "
                f"({len(containers)} container(s))",
                file=sys.stderr,
            )
            return 2
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
        # Reachable only with filters_active=True (discovery-time
        # guard above handles the no-filter case).
        active_parts: list[str] = []
        if args.windows is not None:
            joined = ",".join(str(i) for i in args.windows)
            active_parts.append(f"--windows={joined}")
        if args.emin is not None:
            active_parts.append(f"--emin={args.emin}")
        if args.emax is not None:
            active_parts.append(f"--emax={args.emax}")
        print(
            f"error: filters left fewer than 2 windows for stitching "
            f"({' '.join(active_parts)}; "
            f"{_format_window_summary(surviving_keys, len(by_window))})",
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
    msg = f"wrote {args.output} ({len(stitched)} rows)"
    if errors:
        msg += f"; max overlap std = {max(errors.values()):.3g}"
    if filters_active:
        msg += "; " + _format_window_summary(surviving_keys, len(by_window))
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
