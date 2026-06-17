"""``mchammer-pt-reweight-observables`` console script.

Reweight stitched microcanonical observable moments (one CSV per tag, as
written by ``mchammer-pt-stitch-observables``) against a stitched density
of states into canonical ``<O>(T)``, ``<O^2>(T)`` and the Binder cumulant
``U(T)``. Sibling to ``mchammer-pt-reweight``.

The ``moments`` argument may be a single per-tag CSV or a directory of
them; a directory is reweighted tag-by-tag in one invocation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mchammer_pt.analysis.observables import reweight_observables


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``mchammer-pt-reweight-observables``."""
    p = argparse.ArgumentParser(
        prog="mchammer-pt-reweight-observables",
        description=(
            "Reweight microcanonical observable moments to canonical "
            "<O>(T), <O^2>(T) and the Binder cumulant U(T), using a "
            "stitched Wang-Landau density of states."
        ),
    )
    p.add_argument(
        "moments",
        type=Path,
        help=(
            "A per-tag moments CSV (energy, count, <name>_sum/_sum2/_sum4) "
            "or a directory of them (as written by "
            "mchammer-pt-stitch-observables)."
        ),
    )
    p.add_argument(
        "dos_csv",
        type=Path,
        help="Stitched DOS CSV with 'energy' and 'entropy' (ln g) columns.",
    )
    p.add_argument(
        "--T-min", type=float, default=200.0,
        help="Minimum temperature (K). Default: 200.",
    )
    p.add_argument(
        "--T-max", type=float, default=800.0,
        help="Maximum temperature (K). Default: 800.",
    )
    p.add_argument(
        "--T-step", type=float, default=2.0,
        help="Temperature step (K). Default: 2.0.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help=(
            "Output path. For a directory input: an output directory "
            "(default: canonical_observables/); for a single-file input: an "
            "output CSV (default: <stem>_canonical.csv)."
        ),
    )
    return p


def _temperature_grid(
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, str | None]:
    """Build the temperature grid, mirroring ``mchammer-pt-reweight``.

    Returns ``(grid, None)`` on success or ``(None, message)`` if the
    ``--T-*`` arguments are inconsistent.
    """
    if not (args.T_min < args.T_max):
        return None, f"require T-min < T-max, got {args.T_min}, {args.T_max}"
    if args.T_step <= 0:
        return None, f"T-step must be > 0, got {args.T_step}"
    if args.T_min <= 0:
        return None, f"T-min must be > 0 K, got {args.T_min}"
    n = (args.T_max - args.T_min) / args.T_step
    n_int = round(n)
    if abs(n - n_int) > 1e-9:
        return None, (
            f"T-step={args.T_step} K does not divide "
            f"[{args.T_min}, {args.T_max}] into an integer number of "
            f"intervals (got {n:.6g}); pick a step that divides the range "
            f"evenly"
        )
    return np.linspace(args.T_min, args.T_max, int(n_int) + 1), None


def main(argv: list[str] | None = None) -> int:
    """Reweight microcanonical observable moments to canonical averages.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        ``0`` on success; ``2`` on any argument, read, or content error.
    """
    args = _build_parser().parse_args(argv)

    ts, err = _temperature_grid(args)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    assert ts is not None  # a grid is always returned when err is None

    try:
        dos = pd.read_csv(args.dos_csv)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: could not read {args.dos_csv}: {exc}", file=sys.stderr)
        return 2
    if not {"energy", "entropy"}.issubset(dos.columns):
        print(
            f"error: {args.dos_csv} must contain 'energy' and 'entropy' "
            f"columns",
            file=sys.stderr,
        )
        return 2

    if args.moments.is_dir():
        inputs = sorted(args.moments.glob("*.csv"))
        if not inputs:
            print(
                f"error: no CSV files found in {args.moments}",
                file=sys.stderr,
            )
            return 2
        outdir = (
            args.output if args.output is not None
            else Path("canonical_observables")
        )
        outdir.mkdir(parents=True, exist_ok=True)
        outputs = [outdir / f"{p.stem}.csv" for p in inputs]
    else:
        inputs = [args.moments]
        default = args.moments.with_name(f"{args.moments.stem}_canonical.csv")
        outputs = [args.output if args.output is not None else default]

    for src, dst in zip(inputs, outputs, strict=True):
        try:
            moments = pd.read_csv(src)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            print(f"error: could not read {src}: {exc}", file=sys.stderr)
            return 2
        if not {"energy", "count"}.issubset(moments.columns):
            print(
                f"error: {src} must contain 'energy' and 'count' columns",
                file=sys.stderr,
            )
            return 2
        try:
            canonical = reweight_observables(moments, dos, ts)
        except ValueError as exc:
            print(f"error: {src}: {exc}", file=sys.stderr)
            return 2
        canonical.to_csv(dst, index=False)
        print(f"wrote {dst} ({len(canonical)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
