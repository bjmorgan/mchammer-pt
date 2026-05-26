"""Smoke tests for mchammer_pt.cli.coexistence."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mchammer_pt.cli.coexistence import main
from tests._coexistence_fixtures import (
    single_gaussian_dos,
    two_gaussian_dos,
)


def _asymmetric_two_gaussian_dos():
    # Asymmetric weights so a finite equal-area T_c exists.
    return two_gaussian_dos(
        E_low=-1.0, E_high=1.0,
        sigma_low=0.1, sigma_high=0.1,
        weight_low=1.0, weight_high=2.0,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )


def _write_dos(path: Path, dos: pd.DataFrame) -> None:
    dos.to_csv(path, index=False)


def test_cli_writes_json_with_expected_fields(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _asymmetric_two_gaussian_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_json),
        "--format", "json",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    for key in (
        "T_K", "E_peak_low", "E_peak_high", "E_star",
        "latent_heat", "barrier_height", "weight_imbalance",
        "n_bisection_steps",
    ):
        assert key in data, f"missing key in JSON: {key}"


def test_cli_writes_csv_row(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_csv = tmp_path / "result.csv"
    _write_dos(dos_csv, _asymmetric_two_gaussian_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_csv),
        "--format", "csv",
    ])
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert "T_K" in df.columns
    assert "latent_heat" in df.columns


def test_cli_fails_on_unimodal_dos(tmp_path, capsys):
    dos = single_gaussian_dos(
        E_centre=0.0, sigma=0.5,
        E_min=-2.0, E_max=2.0, energy_spacing=0.01,
    )
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv, dos)
    rc = main([
        str(dos_csv),
        "--output", str(tmp_path / "out.json"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    # The diagnostic identifies the underlying cause (no bimodality).
    assert "error" in err.lower()
    assert "bimodal" in err.lower() or "maxima" in err.lower()


def test_cli_forwards_user_t_bracket(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _asymmetric_two_gaussian_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_json),
        "--T-bracket", "1000", "200000",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["T_K"] >= 1000.0
    assert data["T_K"] <= 200000.0


def test_cli_rejects_missing_dos_columns(tmp_path, capsys):
    bogus = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    dos_csv = tmp_path / "bogus.csv"
    bogus.to_csv(dos_csv, index=False)
    rc = main([str(dos_csv), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "energy" in err and "entropy" in err


def test_cli_rejects_unreadable_dos(tmp_path, capsys):
    # Path to a non-existent file: pd.read_csv raises FileNotFoundError
    # (an OSError subclass), which the CLI catches and reports.
    missing = tmp_path / "does_not_exist.csv"
    rc = main([str(missing), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "could not read" in err


def test_cli_rejects_non_numeric_columns(tmp_path, capsys):
    bogus = pd.DataFrame({"energy": ["x", "y"], "entropy": ["a", "b"]})
    dos_csv = tmp_path / "bogus.csv"
    bogus.to_csv(dos_csv, index=False)
    rc = main([str(dos_csv), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "numeric" in err


def test_cli_rejects_non_finite_values(tmp_path, capsys):
    dos = _asymmetric_two_gaussian_dos()
    dos.loc[5, "entropy"] = float("nan")
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv, dos)
    rc = main([str(dos_csv), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "non-finite" in err or "NaN" in err


def test_cli_rejects_non_uniform_grid(tmp_path, capsys):
    # Construct a DOS whose energy column has a clearly non-uniform
    # spacing (one bin shifted). The CLI's grid check should fire.
    dos = _asymmetric_two_gaussian_dos().copy()
    dos.loc[10, "energy"] = float(dos.loc[10, "energy"]) + 0.005
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv, dos)
    rc = main([str(dos_csv), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "uniform" in err
