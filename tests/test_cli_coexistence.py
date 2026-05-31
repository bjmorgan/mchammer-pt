"""Smoke tests for mchammer_pt.cli.coexistence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mchammer_pt.cli.coexistence import main
from tests._coexistence_fixtures import lattice_like_dos


def _coexistence_dos():
    # Lattice-like DOS: canonical phi at beta_c=10 is the designed
    # double-well a*(E**2 - c**2)**2 with phase peaks at E = +/- 1;
    # bimodal P(E|T) for T in ~(1006, 1372) K.
    return lattice_like_dos(
        a=1.0, beta_c=10.0, c=1.0,
        E_min=-1.5, E_max=1.5, energy_spacing=0.001,
    )


def _write_dos(path: Path, dos: pd.DataFrame) -> None:
    dos.to_csv(path, index=False)


def test_cli_writes_json_with_expected_fields(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _coexistence_dos())
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
        "n_brentq_iterations", "n_self_consistent_iter",
        "self_consistent_converged",
    ):
        assert key in data, f"missing key in JSON: {key}"


def test_cli_writes_csv_row(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_csv = tmp_path / "result.csv"
    _write_dos(dos_csv, _coexistence_dos())
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
    # Single-bump ln g: no bimodal P(E|T) at any T.
    energies = np.linspace(-2.0, 2.0, 401)
    # Parabola peaked at zero, monotonically decreasing on both sides.
    ln_g = -(energies ** 2)
    ln_g -= ln_g.min()
    dos = pd.DataFrame({"energy": energies, "entropy": ln_g})
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv, dos)
    rc = main([
        str(dos_csv),
        "--output", str(tmp_path / "out.json"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "error" in err.lower()
    # Diagnostic must name the underlying cause (no bimodal P(E|T)
    # anywhere in the scan range), not just "error".
    assert "phi" in err.lower() or "bimodal" in err.lower()


def test_cli_forwards_user_t_bracket(tmp_path):
    # Bracket sits inside the fixture's bimodal-P window (T ~ 1006-1372 K).
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _coexistence_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_json),
        "--T-bracket", "1050", "1350",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["T_K"] >= 1050.0
    assert data["T_K"] <= 1350.0


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
    dos = _coexistence_dos()
    dos.loc[5, "entropy"] = float("nan")
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv, dos)
    rc = main([str(dos_csv), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "non-finite" in err or "NaN" in err


def test_cli_accepts_smooth_sigma_flag(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _coexistence_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_json),
        "--smooth-sigma", "2.5",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    # The result should still expose the new diagnostic fields
    # (already covered by other tests) and produce a valid T_K
    # under non-default smoothing.
    assert "T_K" in data
    assert data["T_K"] > 0


def test_cli_no_self_consistent_flag(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _coexistence_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_json),
        "--no-self-consistent",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    # With iteration disabled, n_self_consistent_iter must be 0
    # and self_consistent_converged is True (degenerate convergence
    # — see equal_area_temperature semantics).
    assert data["n_self_consistent_iter"] == 0
    assert data["self_consistent_converged"] is True


def test_cli_rejects_negative_smooth_sigma(tmp_path):
    dos_csv = tmp_path / "dos.csv"
    out_json = tmp_path / "result.json"
    _write_dos(dos_csv, _coexistence_dos())
    rc = main([
        str(dos_csv),
        "--output", str(out_json),
        "--smooth-sigma", "-1.0",
    ])
    assert rc != 0


def test_cli_rejects_non_uniform_grid(tmp_path, capsys):
    # Construct a DOS whose energy column has a clearly non-uniform
    # spacing (one bin shifted). The CLI's grid check should fire.
    dos = _coexistence_dos().copy()
    dos.loc[10, "energy"] = float(dos.loc[10, "energy"]) + 0.005
    dos_csv = tmp_path / "dos.csv"
    _write_dos(dos_csv, dos)
    rc = main([str(dos_csv), "--output", str(tmp_path / "out.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "uniform" in err
