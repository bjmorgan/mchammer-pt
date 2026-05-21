"""Smoke tests for mchammer_pt.dos_cli."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mchammer_pt.dos_cli import main


def _write_dos(path: Path) -> None:
    dos = pd.DataFrame({
        "energy": np.array([-1.0, 0.0]),
        "entropy": np.array([0.0, 0.0]),
    })
    dos.to_csv(path, index=False)


def test_dos_cli_writes_canonical_csv(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_csv = tmp_path / "canonical.csv"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "1", "--T-max", "1000000", "--T-step", "999999",
        "-o", str(out_csv),
    ])
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["T_K", "E_mean", "var_E", "Cv"]
    assert df["E_mean"].iloc[0] < -0.99


def test_dos_cli_rejects_bad_temperature_range(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "800", "--T-max", "200", "--T-step", "10",
        "-o", str(tmp_path / "x.csv"),
    ])
    assert rc != 0


@pytest.mark.slow
def test_dos_cli_writes_plot(tmp_path):
    pytest.importorskip("matplotlib")
    dos_csv = tmp_path / "dos.csv"
    out_csv = tmp_path / "canonical.csv"
    plot_png = tmp_path / "plot.png"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "100", "--T-max", "1000", "--T-step", "50",
        "-o", str(out_csv),
        "--plot", str(plot_png),
    ])
    assert rc == 0
    assert plot_png.is_file()
