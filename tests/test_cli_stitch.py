"""Tests for mchammer_pt.cli.stitch."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mchammer.data_containers.base_data_container import BaseDataContainer
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt.cli.stitch import (
    _build_parser,
    _format_window_summary,
    _parse_window_indices,
    _select_window_keys,
    _trim_entropy_bins,
    main,
)


def _mock_dc(
    entropy: dict[int, float],
    energy_spacing: float,
    energy_limit_left: float | None,
    energy_limit_right: float | None,
    *,
    fill_factor: float = 0.5,
    fill_factor_history: dict[int, float] | None = None,
    entropy_history: dict[int, dict[int, float]] | None = None,
) -> object:
    """Mock matching what `read_hdf5` returns for a WL checkpoint after
    the WL-aware dispatch: a `WangLandauDataContainer` with int-keyed
    `_last_state` fields. `_load_loose` produces the same shape via
    `WangLandauDataContainer.read`.
    """
    dc = MagicMock(spec=WangLandauDataContainer)
    dc._last_state = {
        "entropy": dict(entropy),
        "histogram": {k: 1 for k in entropy},
        "fill_factor": fill_factor,
        "fill_factor_history": dict(fill_factor_history or {}),
        "entropy_history": dict(entropy_history or {}),
    }
    dc.ensemble_parameters = {
        "energy_spacing": energy_spacing,
        "energy_limit_left": energy_limit_left,
        "energy_limit_right": energy_limit_right,
    }
    dc.fill_factor = fill_factor
    return dc


# --- Mode dispatch -------------------------------------------------------

def test_stitch_cli_rejects_checkpoint_mode_with_multiple_inputs(
    tmp_path, capsys,
):
    rc = main([
        str(tmp_path / "a.h5"), str(tmp_path / "b.h5"),
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "checkpoint mode expects exactly one input" in err


def test_stitch_cli_rejects_containers_mode_with_one_input(tmp_path, capsys):
    rc = main([
        "--containers", str(tmp_path / "a.dc"),
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "needs at least two" in err


# --- Error paths: containers --------------------------------------------

def test_stitch_cli_rejects_missing_container_files(tmp_path, capsys):
    rc = main([
        "--containers",
        str(tmp_path / "missing_a.dc"),
        str(tmp_path / "missing_b.dc"),
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc != 0
    assert "could not read" in capsys.readouterr().err


def test_stitch_cli_reports_missing_ensemble_param(tmp_path, capsys, monkeypatch):
    # A container that's valid HDF5 but lacks `energy_limit_left` -- e.g.
    # a canonical-ensemble container mixed into a directory of WL files.
    bad = MagicMock(spec=WangLandauDataContainer)
    bad.ensemble_parameters = {"energy_spacing": 0.5}  # no window bounds
    good = _mock_dc(
        entropy={0: 0.0, 1: 1.0},
        energy_spacing=0.5,
        energy_limit_left=0.0,
        energy_limit_right=0.5,
    )
    by_path = {Path("a.dc"): bad, Path("b.dc"): good}
    monkeypatch.setattr(
        WangLandauDataContainer, "read",
        lambda p: by_path[Path(p)],
    )

    rc = main(["--containers", "a.dc", "b.dc", "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "missing required ensemble parameter" in err
    assert "energy_limit_left" in err


# --- Error paths: checkpoints --------------------------------------------

def test_stitch_cli_rejects_unreadable_checkpoint(tmp_path, capsys, monkeypatch):
    def fake_read(_):
        raise KeyError("missing required top-level group 'exchanges'")

    monkeypatch.setattr("mchammer_pt.cli.stitch.read_hdf5", fake_read)
    rc = main([str(tmp_path / "broken.h5"), "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    assert "could not read checkpoint" in capsys.readouterr().err


def test_stitch_cli_rejects_empty_checkpoint(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [], None),
    )
    rc = main([str(tmp_path / "empty.h5"), "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    assert "contains no replica data containers" in capsys.readouterr().err


def test_stitch_cli_rejects_non_wl_container_in_checkpoint(
    tmp_path, capsys, monkeypatch,
):
    # Canonical-ensemble container surviving inside a checkpoint: has
    # `_last_state` but no WL window bounds in `ensemble_parameters`.
    # The missing-param check is the gate.
    canonical = MagicMock(spec=BaseDataContainer)
    canonical._last_state = {}
    canonical.ensemble_parameters = {"energy_spacing": 0.5, "temperature": 300.0}
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [canonical], None),
    )
    rc = main([str(tmp_path / "mix.h5"), "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "missing required ensemble parameter" in err
    assert "energy_limit_left" in err


def test_stitch_cli_rejects_disagreeing_energy_spacing(
    tmp_path, capsys, monkeypatch,
):
    a = _mock_dc(
        entropy={-2: 0.0, -1: 0.4},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=0.0,
    )
    b = _mock_dc(
        entropy={-1: 0.9, 0: 1.2},
        energy_spacing=0.25,  # mismatched
        energy_limit_left=-0.5,
        energy_limit_right=0.5,
    )
    by_path = {Path("a.dc"): a, Path("b.dc"): b}
    monkeypatch.setattr(
        WangLandauDataContainer, "read",
        lambda p: by_path[Path(p)],
    )

    rc = main(["--containers", "a.dc", "b.dc", "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    assert "disagree on energy_spacing" in capsys.readouterr().err


# --- Happy paths ----------------------------------------------------------

def test_stitch_cli_writes_csv_from_containers(tmp_path, monkeypatch, capsys):
    a = _mock_dc(
        entropy={-2: 0.0, -1: 0.4, 0: 0.7},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=0.0,
    )
    b = _mock_dc(
        entropy={-1: 0.9, 0: 1.2, 1: 1.4},
        energy_spacing=0.5,
        energy_limit_left=-0.5,
        energy_limit_right=0.5,
    )
    by_path = {Path("a.dc"): a, Path("b.dc"): b}
    monkeypatch.setattr(
        WangLandauDataContainer, "read",
        lambda p: by_path[Path(p)],
    )

    out = tmp_path / "dos.csv"
    rc = main(["--containers", "a.dc", "b.dc", "-o", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["energy", "entropy"]
    assert len(df) == 4
    assert df["entropy"].min() == pytest.approx(0.0, abs=1e-12)
    stdout = capsys.readouterr().out
    assert "kept" not in stdout


def test_stitch_cli_writes_csv_from_checkpoint(tmp_path, monkeypatch):
    a = _mock_dc(
        entropy={-2: 0.0, -1: 0.4, 0: 0.7},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=0.0,
    )
    b = _mock_dc(
        entropy={-1: 0.9, 0: 1.2, 1: 1.4},
        energy_spacing=0.5,
        energy_limit_left=-0.5,
        energy_limit_right=0.5,
    )
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b], None),
    )

    out = tmp_path / "dos.csv"
    rc = main([str(tmp_path / "run.h5"), "-o", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["energy", "entropy"]
    assert len(df) == 4
    assert df["entropy"].min() == pytest.approx(0.0, abs=1e-12)


def test_stitch_cli_honours_fill_factor_limit(tmp_path, monkeypatch):
    # `--fill-factor-limit` exercises `_last_state["entropy_history"]`
    # and `["fill_factor_history"]`. The history snapshot at the
    # selected step replaces the current entropy. Build two windows
    # whose current entropy is empty but whose history contains a
    # snapshot at ff=0.1 -- the stitched DOS should reflect the
    # history, not the (empty) current state.
    a = _mock_dc(
        entropy={},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=0.0,
        fill_factor=0.05,
        fill_factor_history={10: 0.1},
        entropy_history={10: {-2: 0.0, -1: 0.4, 0: 0.7}},
    )
    b = _mock_dc(
        entropy={},
        energy_spacing=0.5,
        energy_limit_left=-0.5,
        energy_limit_right=0.5,
        fill_factor=0.05,
        fill_factor_history={10: 0.1},
        entropy_history={10: {-1: 0.9, 0: 1.2, 1: 1.4}},
    )
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b], None),
    )

    out = tmp_path / "dos.csv"
    rc = main([
        str(tmp_path / "run.h5"),
        "--fill-factor-limit", "0.1",
        "-o", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    # Stitching from history -> same shape as the current-state test.
    assert len(df) == 4
    assert df["entropy"].min() == pytest.approx(0.0, abs=1e-12)


def test_stitch_cli_translates_stitch_value_error(tmp_path, monkeypatch, capsys):
    # Two windows that don't overlap (disjoint bins) -> stitch_entropy
    # raises ValueError; CLI should translate to exit 2.
    a = _mock_dc(
        entropy={-2: 0.0, -1: 0.4},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=-0.5,
    )
    b = _mock_dc(
        entropy={2: 0.9, 3: 1.1},
        energy_spacing=0.5,
        energy_limit_left=1.0,
        energy_limit_right=1.5,
    )
    by_path = {Path("a.dc"): a, Path("b.dc"): b}
    monkeypatch.setattr(
        WangLandauDataContainer, "read",
        lambda p: by_path[Path(p)],
    )

    rc = main(["--containers", "a.dc", "b.dc", "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    assert "stitching failed" in capsys.readouterr().err


def test_parser_accepts_windows_flag():
    parser = _build_parser()
    args = parser.parse_args(
        ["input.h5", "--windows", "0,2,3"]
    )
    assert args.windows == [0, 2, 3]


def test_parser_accepts_emin_emax_flags():
    parser = _build_parser()
    args = parser.parse_args(
        ["input.h5", "--emin", "-10.5", "--emax", "-5.0"]
    )
    assert args.emin == pytest.approx(-10.5)
    assert args.emax == pytest.approx(-5.0)


def test_parser_defaults_filter_flags_to_none():
    parser = _build_parser()
    args = parser.parse_args(["input.h5"])
    assert args.windows is None
    assert args.emin is None
    assert args.emax is None


def test_parse_window_indices_rejects_non_integer():
    with pytest.raises(
        argparse.ArgumentTypeError, match="expected comma-separated integers"
    ):
        _parse_window_indices("0,abc,2")


def test_select_window_keys_returns_all_sorted_when_none():
    by_window = {
        (-5.0, -3.0): ["c2"],
        (-10.0, -8.0): ["c0"],
        (-7.0, -5.0): ["c1"],
    }
    keys, err = _select_window_keys(by_window, windows_keep=None)
    assert err is None
    assert keys == [(-10.0, -8.0), (-7.0, -5.0), (-5.0, -3.0)]


def test_select_window_keys_filters_by_index():
    by_window = {
        (-5.0, -3.0): ["c2"],
        (-10.0, -8.0): ["c0"],
        (-7.0, -5.0): ["c1"],
    }
    keys, err = _select_window_keys(by_window, windows_keep=[0, 2])
    assert err is None
    assert keys == [(-10.0, -8.0), (-5.0, -3.0)]


def test_select_window_keys_rejects_out_of_range_index():
    by_window = {
        (-10.0, -8.0): ["c0"],
        (-7.0, -5.0): ["c1"],
    }
    keys, err = _select_window_keys(by_window, windows_keep=[0, 5])
    assert err is not None
    assert "out of range" in err
    assert "5" in err
    assert keys == []


def test_select_window_keys_rejects_negative_index():
    by_window = {(-10.0, -8.0): ["c0"], (-7.0, -5.0): ["c1"]}
    keys, err = _select_window_keys(by_window, windows_keep=[-1])
    assert err is not None
    assert "out of range" in err


def test_select_window_keys_dedupes_and_returns_energy_sorted():
    # User passes duplicates and out-of-order indices; helper should
    # dedupe and emit in energy-sorted order so downstream stitching
    # sees a deterministic input regardless of how the user typed it.
    by_window = {
        (-5.0, -3.0): ["c2"],
        (-10.0, -8.0): ["c0"],
        (-7.0, -5.0): ["c1"],
    }
    keys, err = _select_window_keys(by_window, windows_keep=[2, 0, 0, 1])
    assert err is None
    assert keys == [(-10.0, -8.0), (-7.0, -5.0), (-5.0, -3.0)]


def test_trim_entropy_bins_no_filter_returns_input_unchanged():
    df = pd.DataFrame({"energy": [-2.0, -1.0, 0.0], "entropy": [0.0, 0.5, 1.0]})
    out = _trim_entropy_bins(df, emin=None, emax=None)
    assert out["energy"].tolist() == [-2.0, -1.0, 0.0]
    assert out["entropy"].tolist() == [0.0, 0.5, 1.0]


def test_trim_entropy_bins_emin_drops_strictly_below_or_equal():
    df = pd.DataFrame(
        {"energy": [-2.0, -1.0, 0.0, 1.0], "entropy": [0.0, 0.5, 1.0, 1.5]}
    )
    out = _trim_entropy_bins(df, emin=-1.0, emax=None)
    # bins at E <= -1.0 dropped; -1.0 itself goes (strict ``<=``)
    assert out["energy"].tolist() == [0.0, 1.0]


def test_trim_entropy_bins_emax_drops_strictly_above_or_equal():
    df = pd.DataFrame(
        {"energy": [-2.0, -1.0, 0.0, 1.0], "entropy": [0.0, 0.5, 1.0, 1.5]}
    )
    out = _trim_entropy_bins(df, emin=None, emax=0.0)
    # bins at E >= 0.0 dropped
    assert out["energy"].tolist() == [-2.0, -1.0]


def test_trim_entropy_bins_both_filters_compose():
    df = pd.DataFrame({"energy": [-3.0, -2.0, -1.0, 0.0, 1.0], "entropy": np.zeros(5)})
    out = _trim_entropy_bins(df, emin=-2.0, emax=0.0)
    # keep -2.0 < E < 0.0 -> only -1.0 survives
    assert out["energy"].tolist() == [-1.0]


def test_trim_entropy_bins_returns_empty_when_everything_dropped():
    df = pd.DataFrame({"energy": [-2.0, -1.0], "entropy": [0.0, 0.5]})
    out = _trim_entropy_bins(df, emin=0.0, emax=None)
    assert out.empty


def test_format_window_summary_lists_kept_windows():
    kept = [(-10.0, -8.0), (-7.0, -5.0)]
    out = _format_window_summary(kept_keys=kept, total=3)
    assert "kept 2 of 3 windows" in out
    assert "(-10.0, -8.0)" in out
    assert "(-7.0, -5.0)" in out


def test_format_window_summary_handles_none_bounds():
    # An outer window may have None for left or right energy limit
    # (no-bound case from mchammer's ensemble_parameters).
    kept = [(None, -8.0), (-7.0, None)]
    out = _format_window_summary(kept_keys=kept, total=2)
    assert "(None, -8.0)" in out
    assert "(-7.0, None)" in out


def test_format_window_summary_no_kept_windows():
    out = _format_window_summary(kept_keys=[], total=5)
    assert "kept 0 of 5 windows" in out


def _three_window_mocks() -> tuple[object, object, object]:
    """Three adjacent windows with simple linear ``ln g`` in each.

    ``_mock_dc``'s ``entropy`` dict uses integer bin keys; with
    ``energy_spacing=0.5`` the per-window bins are:

    - Window A (energy_limit_left=-2.0, right=-1.0): bins -2.0, -1.5, -1.0
    - Window B (energy_limit_left=-1.5, right=-0.5): bins -1.5, -1.0, -0.5
    - Window C (energy_limit_left=-1.0, right= 0.0): bins -1.0, -0.5,  0.0
    """
    a = _mock_dc(
        entropy={-4: 0.0, -3: 0.4, -2: 0.7},
        energy_spacing=0.5,
        energy_limit_left=-2.0,
        energy_limit_right=-1.0,
    )
    b = _mock_dc(
        entropy={-3: 0.9, -2: 1.0, -1: 1.2},
        energy_spacing=0.5,
        energy_limit_left=-1.5,
        energy_limit_right=-0.5,
    )
    c = _mock_dc(
        entropy={-2: 1.4, -1: 1.5, 0: 1.7},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=0.0,
    )
    return a, b, c


def test_stitch_cli_windows_filter_drops_lowest(tmp_path, monkeypatch, capsys):
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    out = tmp_path / "dos.csv"
    rc = main([
        str(tmp_path / "run.h5"),
        "--windows", "1,2",
        "-o", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    # Window 0 (energies -2.0 .. -1.0) dropped; stitched DOS covers
    # the bins from windows 1 and 2 only.
    assert df["energy"].min() >= -1.5 - 1e-9
    stdout = capsys.readouterr().out
    assert "kept 2 of 3 windows" in stdout


def test_stitch_cli_windows_with_hole_fails_at_stitch(
    tmp_path, monkeypatch, capsys,
):
    # A user who selects non-adjacent windows (skipping the middle
    # two of a four-window ladder) leaves a gap that stitch_entropy
    # cannot bridge — windows 0 and 3 share no overlap bins. The CLI
    # surfaces this via the existing stitch_entropy ValueError catch.
    a = _mock_dc(
        entropy={-4: 0.0, -3: 0.4, -2: 0.7},
        energy_spacing=0.5,
        energy_limit_left=-2.0,
        energy_limit_right=-1.0,
    )
    b = _mock_dc(
        entropy={-3: 0.9, -2: 1.0, -1: 1.2},
        energy_spacing=0.5,
        energy_limit_left=-1.5,
        energy_limit_right=-0.5,
    )
    c = _mock_dc(
        entropy={-2: 1.4, -1: 1.5, 0: 1.7},
        energy_spacing=0.5,
        energy_limit_left=-1.0,
        energy_limit_right=0.0,
    )
    d = _mock_dc(
        entropy={1: 1.9, 2: 2.0, 3: 2.2},
        energy_spacing=0.5,
        energy_limit_left=0.5,
        energy_limit_right=1.5,
    )
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c, d], None),
    )
    rc = main([
        str(tmp_path / "run.h5"),
        "--windows", "0,3",
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "stitching failed" in err
    assert "overlap" in err.lower()


def test_stitch_cli_windows_and_emin_compose(tmp_path, monkeypatch, capsys):
    # --windows indexes discovery order, applied first; --emin then
    # trims bins from each surviving window. With --windows=1,2 and
    # --emin=-1.0: window B keeps -0.5; window C keeps -0.5 and 0.0.
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    out = tmp_path / "dos.csv"
    rc = main([
        str(tmp_path / "run.h5"),
        "--windows", "1,2",
        "--emin", "-1.0",
        "-o", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert (df["energy"] > -1.0).all()
    assert df["energy"].max() == pytest.approx(0.0)
    stdout = capsys.readouterr().out
    assert "kept 2 of 3 windows" in stdout


def test_stitch_cli_emin_trims_low_bins(tmp_path, monkeypatch, capsys):
    # --emin=-1.5 drops bins at E <= -1.5. Window A keeps -1.0;
    # window B keeps -1.0 and -0.5; window C keeps all three.
    # Stitched DOS contains no bins at or below -1.5.
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    out = tmp_path / "dos.csv"
    rc = main([
        str(tmp_path / "run.h5"),
        "--emin", "-1.5",
        "-o", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert (df["energy"] > -1.5).all()
    stdout = capsys.readouterr().out
    assert "kept 3 of 3 windows" in stdout


def test_stitch_cli_emin_drops_entire_window_when_out_of_range(
    tmp_path, monkeypatch, capsys,
):
    # --emin=-1.0 drops every bin in window A; its trimmed DataFrame
    # is empty and the window is silently dropped from per_window.
    # Window B keeps -0.5 only; window C keeps -0.5 and 0.0;
    # stitch sees the overlap at -0.5.
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    out = tmp_path / "dos.csv"
    rc = main([
        str(tmp_path / "run.h5"),
        "--emin", "-1.0",
        "-o", str(out),
    ])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "kept 2 of 3 windows" in stdout
    # The stitched DOS must contain no bins at or below -1.0.
    df = pd.read_csv(out)
    assert (df["energy"] > -1.0).all()


def test_stitch_cli_emax_trims_high_bins(tmp_path, monkeypatch, capsys):
    # Symmetric counterpart to test_stitch_cli_emin_trims_low_bins:
    # --emax=-0.5 drops bins at E >= -0.5. Window A keeps all three;
    # window B keeps -1.5 and -1.0; window C keeps -1.0 only.
    # Stitched DOS contains no bins at or above -0.5.
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    out = tmp_path / "dos.csv"
    rc = main([
        str(tmp_path / "run.h5"),
        "--emax", "-0.5",
        "-o", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert (df["energy"] < -0.5).all()
    stdout = capsys.readouterr().out
    assert "kept 3 of 3 windows" in stdout


def test_stitch_cli_rejects_emin_not_below_emax(
    tmp_path, monkeypatch, capsys,
):
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    rc = main([
        str(tmp_path / "run.h5"),
        "--emin", "-1.0", "--emax", "-2.0",
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--emin" in err and "--emax" in err


def test_stitch_cli_rejects_windows_out_of_range(
    tmp_path, monkeypatch, capsys,
):
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    rc = main([
        str(tmp_path / "run.h5"),
        "--windows", "0,7",
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "out of range" in err
    assert "7" in err


def test_stitch_cli_filters_reducing_below_two_windows_has_distinct_error(
    tmp_path, monkeypatch, capsys,
):
    a, b, c = _three_window_mocks()
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda _: (None, [a, b, c], None),
    )
    rc = main([
        str(tmp_path / "run.h5"),
        "--windows", "0",
        "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "fewer than 2 windows" in err
    assert "--windows" in err
    assert "all containers share" not in err
