"""Smoke tests for mchammer_pt.cli.reweight."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mchammer_pt.cli.reweight import main


def _write_dos(path: Path) -> None:
    dos = pd.DataFrame({
        "energy": np.array([-1.0, 0.0]),
        "entropy": np.array([0.0, 0.0]),
    })
    dos.to_csv(path, index=False)


def test_dos_cli_writes_canonical_csv(tmp_path):
    # Two-level system at -1 eV and 0 eV, equal degeneracy.
    # At T=1 K, ground state dominates: <E> -> -1.0.
    # At T=1e10 K, weights equalise: <E> -> -0.5 (mean of -1 and 0).
    dos_csv = tmp_path / "dos.csv"
    out_csv = tmp_path / "canonical.csv"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "1", "--T-max", "1e10", "--T-step", "1e10",
        "-o", str(out_csv),
    ])
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["T_K", "E_mean", "var_E", "Cv"]
    assert len(df) == 2
    assert df["E_mean"].iloc[0] < -0.99           # low-T limit
    assert abs(df["E_mean"].iloc[1] - (-0.5)) < 1e-6  # high-T limit


def test_dos_cli_rejects_bad_temperature_range(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "800", "--T-max", "200", "--T-step", "10",
        "-o", str(tmp_path / "x.csv"),
    ])
    assert rc != 0


def test_dos_cli_rejects_non_positive_T_min(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "0", "--T-max", "100", "--T-step", "10",
        "-o", str(tmp_path / "x.csv"),
    ])
    assert rc != 0


def test_dos_cli_rejects_step_not_dividing_range(tmp_path, capsys):
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv)
    rc = main([
        str(dos_csv),
        "--T-min", "200", "--T-max", "801", "--T-step", "2",
        "-o", str(tmp_path / "x.csv"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "does not divide" in err


def test_dos_cli_rejects_missing_columns(tmp_path, capsys):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"energy": [0.0, 1.0]}).to_csv(bad_csv, index=False)
    rc = main([
        str(bad_csv),
        "--T-min", "100", "--T-max", "200", "--T-step", "10",
        "-o", str(tmp_path / "x.csv"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "energy" in err and "entropy" in err


def test_dos_cli_rejects_missing_file(tmp_path, capsys):
    rc = main([
        str(tmp_path / "does_not_exist.csv"),
        "--T-min", "100", "--T-max", "200", "--T-step", "10",
        "-o", str(tmp_path / "x.csv"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "could not read" in err


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
