"""``mchammer-pt-stitch-observables`` console script.

Merge the per-walker microcanonical observable moment stores recorded
during a frozen-g measurement run into one CSV per observer tag, summed
bin-wise onto the energy grid. Sibling to ``mchammer-pt-stitch`` (which
stitches the density of states); the output of this script feeds
``mchammer-pt-reweight-observables``.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mchammer_pt.analysis.observables import stitch_observable_moments
from mchammer_pt.history import read_hdf5


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``mchammer-pt-stitch-observables``."""
    p = argparse.ArgumentParser(
        prog="mchammer-pt-stitch-observables",
        description=(
            "Merge the per-walker microcanonical observable moment stores "
            "recorded during a frozen-g measurement run into one CSV per "
            "observer tag, summed bin-wise onto the energy grid."
        ),
    )
    p.add_argument(
        "checkpoint",
        type=Path,
        help="A measurement-run mchammer-pt checkpoint HDF5 file.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("observables"),
        help="Output directory for the per-tag CSVs. Default: observables/ (CWD).",
    )
    return p


def _read_energy_spacing(
    meta: Mapping[str, Any], containers: list[Any]
) -> float | None:
    """Return the energy spacing from the checkpoint meta or containers.

    Prefers ``meta["energy_spacing"]`` (always present on mchammer-pt
    checkpoints); falls back to the first container's
    ``ensemble_parameters``.

    Args:
        meta: Checkpoint metadata mapping from :func:`read_hdf5`.
        containers: The per-replica data containers.

    Returns:
        The energy spacing in eV, or ``None`` if neither source records it.
    """
    spacing = meta.get("energy_spacing")
    if spacing is None and containers:
        spacing = containers[0].ensemble_parameters.get("energy_spacing")
    return None if spacing is None else float(spacing)


def main(argv: list[str] | None = None) -> int:
    """Stitch per-walker observable moments from a checkpoint into per-tag CSVs.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        ``0`` on success; ``2`` on any read, content, or validation error.
    """
    args = _build_parser().parse_args(argv)

    try:
        _, containers, meta = read_hdf5(args.checkpoint)
    except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
        print(
            f"error: could not read checkpoint {args.checkpoint}: {exc}",
            file=sys.stderr,
        )
        return 2

    if not containers:
        print(
            f"error: checkpoint {args.checkpoint} contains no replicas",
            file=sys.stderr,
        )
        return 2

    energy_spacing = _read_energy_spacing(meta, containers)
    if energy_spacing is None:
        print(
            f"error: checkpoint {args.checkpoint} does not record an "
            f"energy_spacing",
            file=sys.stderr,
        )
        return 2

    try:
        frames = stitch_observable_moments(containers, energy_spacing)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not frames:
        print(
            "error: no observable records found in the checkpoint; attach "
            "observers with record_observable before the measurement run "
            "(this may be a pre-measurement density-of-states checkpoint)",
            file=sys.stderr,
        )
        return 2

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, df in sorted(frames.items()):
        path = outdir / f"{tag}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows) for observable {tag!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
