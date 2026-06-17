"""Tests for mchammer_pt.cli.stitch_observables."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt.cli.stitch_observables import main


def _make_record(
    tag: str,
    names: list[str],
    bins: list[int],
    counts: list[int],
    sums: list[list[float]],
    sum2s: list[list[float]],
    sum4s: list[list[float]],
    interval: int = 1,
) -> dict:
    return {
        "tag": tag,
        "names": names,
        "interval": interval,
        "bins": bins,
        "count": counts,
        "sum": sums,
        "sum2": sum2s,
        "sum4": sum4s,
        "skipped": {},
    }


def _mock_wl_dc(
    observable_records: dict,
    energy_spacing: float,
    energy_limit_left: float | None,
    energy_limit_right: float | None,
) -> object:
    dc = MagicMock(spec=WangLandauDataContainer)
    dc._last_state = {
        "entropy": {},
        "histogram": {},
        "fill_factor": 0.5,
        "fill_factor_history": {},
        "entropy_history": {},
        "observable_records": observable_records,
    }
    dc.ensemble_parameters = {
        "energy_spacing": energy_spacing,
        "energy_limit_left": energy_limit_left,
        "energy_limit_right": energy_limit_right,
    }
    return dc


def _two_window_containers() -> tuple[object, object]:
    """Two containers with a single 'energy' tag, one per window."""
    r0 = _make_record(
        tag="energy", names=["energy"],
        bins=[0, 1, 2], counts=[10, 10, 10],
        sums=[[1.0], [2.0], [3.0]],
        sum2s=[[1.0], [4.0], [9.0]],
        sum4s=[[1.0], [16.0], [81.0]],
    )
    r1 = _make_record(
        tag="energy", names=["energy"],
        bins=[2, 3, 4], counts=[8, 8, 8],
        sums=[[3.0], [4.0], [5.0]],
        sum2s=[[9.0], [16.0], [25.0]],
        sum4s=[[81.0], [256.0], [625.0]],
    )
    dc0 = _mock_wl_dc(
        observable_records={"energy": r0},
        energy_spacing=0.5,
        energy_limit_left=None,
        energy_limit_right=None,
    )
    dc1 = _mock_wl_dc(
        observable_records={"energy": r1},
        energy_spacing=0.5,
        energy_limit_left=None,
        energy_limit_right=None,
    )
    return dc0, dc1


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_main_happy_path_single_tag(tmp_path, monkeypatch, capsys):
    dc0, dc1 = _two_window_containers()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc0, dc1], {"energy_spacing": 0.5}),
    )
    outdir = tmp_path / "obs_out"
    rc = main([str(tmp_path / "run.h5"), "-o", str(outdir)])

    assert rc == 0
    assert (outdir / "energy.csv").exists()

    df = pd.read_csv(outdir / "energy.csv")
    assert "energy" in df.columns
    assert "count" in df.columns
    assert "energy_sum" in df.columns
    assert "energy_sum2" in df.columns
    assert "energy_sum4" in df.columns
    assert len(df) > 0

    out = capsys.readouterr().out
    assert "energy" in out


def test_main_happy_path_two_tags(tmp_path, monkeypatch):
    r_energy = _make_record(
        tag="energy", names=["energy"],
        bins=[0, 1], counts=[5, 5],
        sums=[[1.0], [2.0]], sum2s=[[1.0], [4.0]], sum4s=[[1.0], [16.0]],
    )
    r_mag = _make_record(
        tag="mag", names=["mag"],
        bins=[0, 1], counts=[3, 3],
        sums=[[0.1], [0.2]], sum2s=[[0.01], [0.04]], sum4s=[[0.0001], [0.0016]],
    )
    dc = _mock_wl_dc(
        observable_records={"energy": r_energy, "mag": r_mag},
        energy_spacing=1.0,
        energy_limit_left=None,
        energy_limit_right=None,
    )
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc], {"energy_spacing": 1.0}),
    )
    outdir = tmp_path / "obs_out2"
    rc = main([str(tmp_path / "run.h5"), "-o", str(outdir)])

    assert rc == 0
    assert (outdir / "energy.csv").exists()
    assert (outdir / "mag.csv").exists()

    energy_df = pd.read_csv(outdir / "energy.csv")
    mag_df = pd.read_csv(outdir / "mag.csv")
    assert len(energy_df) == 2
    assert len(mag_df) == 2


def test_main_creates_output_directory(tmp_path, monkeypatch):
    dc, _ = _two_window_containers()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc], {"energy_spacing": 0.5}),
    )
    outdir = tmp_path / "nested" / "obs_dir"
    assert not outdir.exists()

    rc = main([str(tmp_path / "run.h5"), "-o", str(outdir)])
    assert rc == 0
    assert outdir.exists()


def test_main_default_output_dir_is_observables(tmp_path, monkeypatch, capsys):
    """Without -o, the default output directory is 'observables/' relative to CWD."""
    dc, _ = _two_window_containers()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc], {"energy_spacing": 0.5}),
    )
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rc = main([str(tmp_path / "run.h5")])
        assert rc == 0
        assert (tmp_path / "observables" / "energy.csv").exists()
    finally:
        os.chdir(old_cwd)


def test_main_prints_tag_and_row_count(tmp_path, monkeypatch, capsys):
    dc0, dc1 = _two_window_containers()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc0, dc1], {"energy_spacing": 0.5}),
    )
    outdir = tmp_path / "obs_rows"
    rc = main([str(tmp_path / "run.h5"), "-o", str(outdir)])
    assert rc == 0
    out = capsys.readouterr().out
    # Should mention the tag name and the row count
    assert "energy" in out
    # Row count should appear somewhere (5 bins: 0,1,2,3,4)
    assert any(str(n) in out for n in range(1, 20))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_main_unreadable_checkpoint_exits_2(tmp_path, monkeypatch, capsys):
    def _raise(_):
        raise OSError("no such file")
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        _raise,
    )
    rc = main([str(tmp_path / "missing.h5"), "-o", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_main_no_observable_records_exits_2(tmp_path, monkeypatch, capsys):
    """A checkpoint with no observable_records in any container exits 2."""
    dc = _mock_wl_dc(
        observable_records={},
        energy_spacing=1.0,
        energy_limit_left=None,
        energy_limit_right=None,
    )
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc], {"energy_spacing": 1.0}),
    )
    outdir = tmp_path / "empty_out"
    rc = main([str(tmp_path / "run.h5"), "-o", str(outdir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err
    # Message should hint that no recorders were attached
    assert "record" in err.lower() or "observable" in err.lower()


def test_main_missing_observable_records_key_exits_2(tmp_path, monkeypatch, capsys):
    """A container without the observable_records key at all (pre-measurement checkpoint)."""
    dc = MagicMock(spec=WangLandauDataContainer)
    dc._last_state = {"entropy": {}}
    dc.ensemble_parameters = {
        "energy_spacing": 1.0,
        "energy_limit_left": None,
        "energy_limit_right": None,
    }
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc], {"energy_spacing": 1.0}),
    )
    rc = main([str(tmp_path / "run.h5"), "-o", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_main_empty_containers_list_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [], {}),
    )
    rc = main([str(tmp_path / "run.h5"), "-o", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_main_csv_columns_correct_for_s2_observer(tmp_path, monkeypatch):
    """S=2 observer ('a', 'b') produces correctly named columns in output CSV."""
    r = _make_record(
        tag="vec",
        names=["a", "b"],
        bins=[0, 1],
        counts=[10, 10],
        sums=[[1.0, 2.0], [3.0, 4.0]],
        sum2s=[[1.0, 4.0], [9.0, 16.0]],
        sum4s=[[1.0, 16.0], [81.0, 256.0]],
    )
    dc = _mock_wl_dc(
        observable_records={"vec": r},
        energy_spacing=1.0,
        energy_limit_left=None,
        energy_limit_right=None,
    )
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch_observables.read_hdf5",
        lambda _: (None, [dc], {"energy_spacing": 1.0}),
    )
    outdir = tmp_path / "s2_out"
    rc = main([str(tmp_path / "run.h5"), "-o", str(outdir)])
    assert rc == 0
    df = pd.read_csv(outdir / "vec.csv")
    assert "a_sum" in df.columns
    assert "a_sum2" in df.columns
    assert "a_sum4" in df.columns
    assert "b_sum" in df.columns
    assert "b_sum2" in df.columns
    assert "b_sum4" in df.columns
