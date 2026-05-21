"""Standalone canonical-reweighting CLI: ``mchammer-pt-reweight``.

Reads a stitched DOS CSV (columns ``energy``, ``entropy``, with
``entropy`` treated as ``ln g(E)``), writes a canonical-moments CSV
(``T_K``, ``E_mean``, ``var_E``, ``Cv``), and optionally emits a
two-panel ``C_v`` / ``<E>(T)`` PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mchammer_pt.dos import reweight_canonical_from_dos


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mchammer-pt-reweight",
        description=(
            "Canonical reweighting from a stitched Wang-Landau DOS."
        ),
    )
    p.add_argument(
        "dos_csv", type=Path,
        help="Input CSV with 'energy' and 'entropy' (ln g) columns.",
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
        "-o", "--output", type=Path,
        default=Path("canonical_reweighted.csv"),
        help="Output CSV path. Default: canonical_reweighted.csv (CWD).",
    )
    p.add_argument(
        "--plot", type=Path, default=None,
        help="Optional PNG path for a Cv + <E>(T) plot.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not (args.T_min < args.T_max):
        print(
            f"error: require T-min < T-max, got {args.T_min}, {args.T_max}",
            file=sys.stderr,
        )
        return 2
    if args.T_step <= 0:
        print(
            f"error: T-step must be > 0, got {args.T_step}",
            file=sys.stderr,
        )
        return 2
    if args.T_min <= 0:
        print(
            f"error: T-min must be > 0 K, got {args.T_min}",
            file=sys.stderr,
        )
        return 2

    dos = pd.read_csv(args.dos_csv)
    if not {"energy", "entropy"}.issubset(dos.columns):
        print(
            f"error: {args.dos_csv} must contain 'energy' and 'entropy' columns",
            file=sys.stderr,
        )
        return 2
    n_T = int(round((args.T_max - args.T_min) / args.T_step)) + 1
    Ts = np.linspace(args.T_min, args.T_max, n_T)
    canonical = reweight_canonical_from_dos(dos, Ts)
    canonical.to_csv(args.output, index=False)
    iCv = int(np.argmax(canonical["Cv"].to_numpy()))
    T_peak = float(canonical["T_K"].iloc[iCv])
    print(
        f"wrote {args.output} ({len(canonical)} rows); "
        f"Cv peak at {T_peak:.1f} K"
    )

    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_cv, ax_e) = plt.subplots(1, 2, figsize=(12, 5))
        ax_cv.plot(canonical["T_K"], canonical["Cv"], lw=1.5)
        ax_cv.axvline(
            T_peak, color="grey", ls="--", lw=1, alpha=0.7,
            label=f"peak: {T_peak:.1f} K",
        )
        ax_cv.set_xlabel("T (K)")
        ax_cv.set_ylabel("C_v (eV/K)")
        ax_cv.set_title("Heat capacity")
        ax_cv.legend(fontsize=9)
        ax_cv.grid(alpha=0.3)
        ax_e.plot(canonical["T_K"], canonical["E_mean"], lw=1.5)
        ax_e.set_xlabel("T (K)")
        ax_e.set_ylabel("<E>(T) (eV)")
        ax_e.set_title("Canonical <E>(T)")
        ax_e.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        plt.close(fig)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
