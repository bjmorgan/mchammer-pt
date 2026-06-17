"""Tests for mchammer_pt.cli.reweight_observables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mchammer_pt.cli.reweight_observables import main

# 300, 400, 500 K -> 3 temperatures.
GRID = ["--T-min", "300", "--T-max", "500", "--T-step", "100"]


def _write_dos(path: Path, energies=(0.0, 1.0, 2.0)) -> None:
    pd.DataFrame(
        {"energy": list(energies), "entropy": [0.0] * len(energies)}
    ).to_csv(path, index=False)


def _write_moments(
    path: Path, energies=(0.0, 1.0, 2.0), c: float = 2.0, name: str = "m",
    count: int = 10,
) -> None:
    """Write a constant-observable moments CSV (<O>(E) = c on every bin)."""
    n = len(energies)
    pd.DataFrame(
        {
            "energy": list(energies),
            "count": [count] * n,
            f"{name}_sum": [count * c] * n,
            f"{name}_sum2": [count * c * c] * n,
            f"{name}_sum4": [count * c**4] * n,
        }
    ).to_csv(path, index=False)


def test_file_mode_happy_path(tmp_path):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    mom = tmp_path / "energy.csv"
    _write_moments(mom, c=2.0)
    out = tmp_path / "out.csv"

    rc = main([str(mom), str(dos), "-o", str(out), *GRID])

    assert rc == 0
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "T_K", "coverage", "m_mean", "m_sq_mean", "m_binder",
    ]
    assert len(df) == 3
    assert np.allclose(df["m_mean"], 2.0)
    assert np.allclose(df["m_binder"], 2.0 / 3.0)


def test_directory_mode_two_tags(tmp_path):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    momdir = tmp_path / "moms"
    momdir.mkdir()
    _write_moments(momdir / "energy.csv", name="energy")
    _write_moments(momdir / "mag.csv", name="mag", c=5.0)
    outdir = tmp_path / "canon"

    rc = main([str(momdir), str(dos), "-o", str(outdir), *GRID])

    assert rc == 0
    assert (outdir / "energy.csv").exists()
    assert (outdir / "mag.csv").exists()
    assert np.allclose(pd.read_csv(outdir / "mag.csv")["mag_mean"], 5.0)


def test_default_output_name_file_mode(tmp_path):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    mom = tmp_path / "energy.csv"
    _write_moments(mom)

    rc = main([str(mom), str(dos), *GRID])  # no -o

    assert rc == 0
    assert (tmp_path / "energy_canonical.csv").exists()


def test_default_output_dir_directory_mode(tmp_path, monkeypatch):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    momdir = tmp_path / "moms"
    momdir.mkdir()
    _write_moments(momdir / "energy.csv", name="energy")
    monkeypatch.chdir(tmp_path)

    rc = main([str(momdir), str(dos), *GRID])  # no -o

    assert rc == 0
    assert (tmp_path / "canonical_observables" / "energy.csv").exists()


def test_t_min_ge_t_max_exits_2(tmp_path, capsys):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    mom = tmp_path / "energy.csv"
    _write_moments(mom)
    rc = main(
        [str(mom), str(dos), "--T-min", "500", "--T-max", "300", "--T-step", "100"]
    )
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_t_step_does_not_divide_exits_2(tmp_path, capsys):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    mom = tmp_path / "energy.csv"
    _write_moments(mom)
    rc = main(
        [str(mom), str(dos), "--T-min", "300", "--T-max", "500", "--T-step", "70"]
    )
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_dos_missing_columns_exits_2(tmp_path, capsys):
    dos = tmp_path / "dos.csv"
    pd.DataFrame({"energy": [0.0, 1.0], "x": [0.0, 0.0]}).to_csv(dos, index=False)
    mom = tmp_path / "energy.csv"
    _write_moments(mom)
    rc = main([str(mom), str(dos), *GRID])
    assert rc == 2
    assert "entropy" in capsys.readouterr().err


def test_moments_missing_columns_exits_2(tmp_path, capsys):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    mom = tmp_path / "energy.csv"
    pd.DataFrame({"energy": [0.0, 1.0], "x": [1, 1]}).to_csv(mom, index=False)
    rc = main([str(mom), str(dos), *GRID])
    assert rc == 2
    assert "count" in capsys.readouterr().err


def test_empty_directory_exits_2(tmp_path, capsys):
    dos = tmp_path / "dos.csv"
    _write_dos(dos)
    momdir = tmp_path / "empty"
    momdir.mkdir()
    rc = main([str(momdir), str(dos), *GRID])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
