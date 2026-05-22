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

from mchammer.data_containers.base_data_container import BaseDataContainer
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt import read_hdf5
from mchammer_pt.analysis.dos import stitch_entropy
from mchammer_pt.wl_result import WindowResult

_REQUIRED_PARAMS = ("energy_spacing", "energy_limit_left", "energy_limit_right")


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

    if len(by_window) < 2:
        print(
            f"error: stitching needs at least two distinct windows; got "
            f"{len(by_window)} (all containers share the same "
            f"energy_limit_left/right)",
            file=sys.stderr,
        )
        return 2

    per_window = []
    for (lo, hi), containers in by_window.items():
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
        per_window.append(df)

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
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
