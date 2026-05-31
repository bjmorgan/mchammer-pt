"""Standalone equal-area coexistence CLI: ``mchammer-pt-coexistence``.

Reads a stitched DOS CSV (columns ``energy``, ``entropy``, with
``entropy`` treated as ``ln g(E)``) and writes a one-row result
containing the equal-area coexistence temperature, the phase peak
locations, the dividing energy, latent heat, barrier height and
bisection diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mchammer_pt.analysis.coexistence import (
    NoBracketError,
    NotBimodalError,
    equal_area_temperature,
)

# Relative tolerance on the uniform-grid check. Float-round bin
# centres can differ by ~1e-12 of the bin spacing on real CSVs.
_UNIFORM_GRID_RTOL = 1e-6


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mchammer-pt-coexistence",
        description=(
            "Equal-area coexistence point from a stitched "
            "Wang-Landau DOS."
        ),
    )
    p.add_argument(
        "dos_csv", type=Path,
        help="Input CSV with 'energy' and 'entropy' (ln g) columns.",
    )
    p.add_argument(
        "-o", "--output", type=Path,
        default=Path("coexistence.json"),
        help="Output path. Default: coexistence.json (CWD).",
    )
    p.add_argument(
        "--format", choices=("json", "csv"), default="json",
        help="Output format. Default: json.",
    )
    p.add_argument(
        "--T-bracket", type=float, nargs=2, metavar=("T_LO", "T_HI"),
        default=None,
        help=(
            "Optional temperature bracket (K) for the bisection. "
            "If omitted, built from a coarse imbalance scan."
        ),
    )
    p.add_argument(
        "--xtol", type=float, default=1e-4,
        help="Relative bisection tolerance on T. Default: 1e-4.",
    )
    p.add_argument(
        "--min-peak-separation", type=int, default=5,
        help=(
            "Minimum bin separation between the two phase peaks. "
            "Default: 5."
        ),
    )
    p.add_argument(
        "--smooth-sigma", type=float, default=2.0,
        help=(
            "Gaussian standard deviation in bins applied to ln g for "
            "topology detection only. Default 2.0. Set to 0 to disable "
            "smoothing."
        ),
    )
    p.add_argument(
        "--no-self-consistent", action="store_true",
        help=(
            "Disable the (T_c, E_star) self-consistency iteration "
            "(single-pass solve). Default off; iteration runs."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        dos = pd.read_csv(args.dos_csv)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"error: could not read {args.dos_csv}: {e}", file=sys.stderr)
        return 2
    if not {"energy", "entropy"}.issubset(dos.columns):
        print(
            f"error: {args.dos_csv} must contain 'energy' and "
            f"'entropy' columns",
            file=sys.stderr,
        )
        return 2
    try:
        dos = dos.astype({"energy": float, "entropy": float})
    except (ValueError, TypeError) as e:
        print(
            f"error: 'energy' and 'entropy' columns in {args.dos_csv} "
            f"must be numeric: {e}",
            file=sys.stderr,
        )
        return 2

    energies = dos["energy"].to_numpy()
    ln_g = dos["entropy"].to_numpy()
    if energies.size < 2:
        print(
            f"error: {args.dos_csv} has fewer than two rows; need a "
            f"grid",
            file=sys.stderr,
        )
        return 2
    if not (np.isfinite(energies).all() and np.isfinite(ln_g).all()):
        print(
            f"error: {args.dos_csv} contains non-finite (NaN/inf) "
            f"values in 'energy' or 'entropy'",
            file=sys.stderr,
        )
        return 2
    diffs = np.diff(energies)
    spacing = float(diffs[0])
    if spacing <= 0.0 or not np.allclose(
        diffs, spacing, rtol=_UNIFORM_GRID_RTOL, atol=0.0,
    ):
        print(
            f"error: 'energy' column in {args.dos_csv} is not on a "
            f"uniform ascending grid (first spacing = {spacing:.6g})",
            file=sys.stderr,
        )
        return 2

    T_bracket = (
        (float(args.T_bracket[0]), float(args.T_bracket[1]))
        if args.T_bracket is not None else None
    )
    try:
        result = equal_area_temperature(
            dos,
            T_bracket=T_bracket,
            xtol=args.xtol,
            min_peak_separation=args.min_peak_separation,
            smoothing_sigma=args.smooth_sigma,
            max_self_consistent_iter=0 if args.no_self_consistent else 20,
        )
    except (NotBimodalError, NoBracketError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not result.self_consistent_converged:
        print(
            "warning: self-consistency iteration did not converge "
            "within 20 passes; reported T_K may differ from the "
            "true fixed point by more than self_consistent_tol_K. "
            "Consider re-running with a different --smooth-sigma "
            "or inspecting the DOS for data-quality issues.",
            file=sys.stderr,
        )

    row = {
        "T_K": result.T_K,
        "E_peak_low": result.split.E_peak_low,
        "E_peak_high": result.split.E_peak_high,
        "E_star": result.split.E_star,
        "latent_heat": result.latent_heat,
        "barrier_height": result.barrier_height,
        "weight_imbalance": result.weight_imbalance,
        "n_brentq_iterations": result.n_brentq_iterations,
        "n_self_consistent_iter": result.n_self_consistent_iter,
        "self_consistent_converged": result.self_consistent_converged,
    }

    if args.format == "json":
        args.output.write_text(json.dumps(row, indent=2) + "\n")
    else:
        pd.DataFrame([row]).to_csv(args.output, index=False)
    print(
        f"wrote {args.output}: T_c = {result.T_K:.3f} K, "
        f"latent heat = {result.latent_heat:.4g} eV, "
        f"barrier = {result.barrier_height:.4g} eV"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
