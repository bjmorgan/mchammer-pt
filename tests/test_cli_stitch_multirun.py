"""Tests for mchammer_pt.cli.stitch_multirun."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt.cli.stitch import main
from mchammer_pt.cli.stitch_multirun import _load_runs, merge_runs_per_window


def test_merge_runs_per_window_recovers_aligned_mean():
    # Two runs, one window, differing in shape on shared bins. The merge
    # aligns each run on the common bins then averages; with two equal
    # contributors the bin-1 value lands halfway and is rebased to min 0.
    runs = [
        {(0.0, 1.0): {0: 0.0, 1: 2.0}},
        {(0.0, 1.0): {0: 0.0, 1: 0.0}},
    ]
    merged = merge_runs_per_window(runs)
    assert set(merged) == {(0.0, 1.0)}
    curve = merged[(0.0, 1.0)]
    assert curve[0] == pytest.approx(0.0)
    assert curve[1] == pytest.approx(1.0)


def test_merge_runs_per_window_weights_repeated_run_by_multiplicity():
    # Passing a run twice (a bootstrap draw with replacement) weights it
    # accordingly: the bin-1 value shifts from 1.0 (A,B equal) to 4/3
    # (A,A,B) because A is now two of three contributors.
    a = {0: 0.0, 1: 2.0}
    b = {0: 0.0, 1: 0.0}
    equal = merge_runs_per_window([{(0.0, 1.0): a}, {(0.0, 1.0): b}])
    doubled = merge_runs_per_window(
        [{(0.0, 1.0): a}, {(0.0, 1.0): a}, {(0.0, 1.0): b}]
    )
    assert equal[(0.0, 1.0)][1] == pytest.approx(1.0)
    assert doubled[(0.0, 1.0)][1] == pytest.approx(4.0 / 3.0)


def test_merge_runs_per_window_rejects_mismatched_window_keys():
    runs = [
        {(0.0, 1.0): {0: 0.0}, (1.0, 2.0): {2: 0.0}},
        {(0.0, 1.0): {0: 0.0}},
    ]
    with pytest.raises(ValueError, match="window keys differ"):
        merge_runs_per_window(runs)


def test_merge_runs_per_window_rejects_empty():
    with pytest.raises(ValueError, match="at least one run"):
        merge_runs_per_window([])


def test_load_runs_reads_each_checkpoint(monkeypatch):
    seen = []

    def fake_read(path):
        seen.append(path)
        return (None, [f"dc-of-{path}"], None)

    monkeypatch.setattr("mchammer_pt.cli.stitch.read_hdf5", fake_read)
    runs, err = _load_runs([Path("r1.h5"), Path("r2.h5")])
    assert err is None
    assert runs == [["dc-of-r1.h5"], ["dc-of-r2.h5"]]
    assert seen == [Path("r1.h5"), Path("r2.h5")]


def test_load_runs_propagates_read_error(monkeypatch):
    def raise_oserror(_):
        raise OSError("nope")

    monkeypatch.setattr("mchammer_pt.cli.stitch.read_hdf5", raise_oserror)
    runs, err = _load_runs([Path("r1.h5")])
    assert runs == []
    assert err is not None
    assert "could not read checkpoint" in err


def _mock_dc(entropy, energy_spacing, lo, hi):
    dc = MagicMock(spec=WangLandauDataContainer)
    dc._last_state = {
        "entropy": dict(entropy),
        "histogram": {k: 1 for k in entropy},
        "fill_factor": 0.5,
        "fill_factor_history": {},
        "entropy_history": {},
    }
    dc.ensemble_parameters = {
        "energy_spacing": energy_spacing,
        "energy_limit_left": lo,
        "energy_limit_right": hi,
    }
    dc.fill_factor = 0.5
    return dc


def _two_window_run(bump=0.0):
    # Two overlapping windows on a 0.5 grid; `bump` perturbs the second
    # window's shape so distinct runs are genuinely different.
    a = _mock_dc({-2: 0.0, -1: 0.4, 0: 0.7}, 0.5, -1.0, 0.0)
    b = _mock_dc({-1: 0.9 + bump, 0: 1.2, 1: 1.4}, 0.5, -0.5, 0.5)
    return [a, b]


def _patch_runs(monkeypatch, mapping):
    monkeypatch.setattr(
        "mchammer_pt.cli.stitch.read_hdf5",
        lambda p: (None, mapping[Path(p)], None),
    )


def test_multi_run_writes_consensus_csv(tmp_path, monkeypatch, capsys):
    _patch_runs(monkeypatch, {
        Path("rA.h5"): _two_window_run(bump=0.0),
        Path("rB.h5"): _two_window_run(bump=0.2),
    })
    out = tmp_path / "dos.csv"
    rc = main(["--multi-run", "rA.h5", "rB.h5", "-o", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["energy", "entropy"]
    assert df["entropy"].min() == pytest.approx(0.0, abs=1e-12)
    assert "merged 2 runs" in capsys.readouterr().out


def test_multi_run_requires_two_inputs(tmp_path, monkeypatch, capsys):
    _patch_runs(monkeypatch, {Path("rA.h5"): _two_window_run()})
    rc = main(["--multi-run", "rA.h5", "-o", str(tmp_path / "dos.csv")])
    assert rc != 0
    assert "at least two run checkpoints" in capsys.readouterr().err


def test_multi_run_accepts_repeated_checkpoint(tmp_path, monkeypatch):
    _patch_runs(monkeypatch, {
        Path("rA.h5"): _two_window_run(bump=0.0),
        Path("rB.h5"): _two_window_run(bump=0.2),
    })
    out = tmp_path / "dos.csv"
    rc = main(["--multi-run", "rA.h5", "rA.h5", "rB.h5", "-o", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["energy", "entropy"]


def test_multi_run_rejects_mismatched_window_keys(tmp_path, monkeypatch, capsys):
    # rB has a single window at a key rA does not share.
    rb = [_mock_dc({-2: 0.0, -1: 0.4, 0: 0.7}, 0.5, -3.0, -2.0),
          _mock_dc({-1: 0.9, 0: 1.2, 1: 1.4}, 0.5, -0.5, 0.5)]
    _patch_runs(monkeypatch, {
        Path("rA.h5"): _two_window_run(),
        Path("rB.h5"): rb,
    })
    rc = main([
        "--multi-run", "rA.h5", "rB.h5", "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc != 0
    assert "different window keys" in capsys.readouterr().err


def test_multi_run_rejects_mismatched_spacing(tmp_path, monkeypatch, capsys):
    rb = [_mock_dc({-2: 0.0, -1: 0.4, 0: 0.7}, 0.25, -1.0, 0.0),
          _mock_dc({-1: 0.9, 0: 1.2, 1: 1.4}, 0.25, -0.5, 0.5)]
    _patch_runs(monkeypatch, {
        Path("rA.h5"): _two_window_run(),
        Path("rB.h5"): rb,
    })
    rc = main([
        "--multi-run", "rA.h5", "rB.h5", "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc != 0
    assert "disagree on energy_spacing" in capsys.readouterr().err


def test_multi_run_emits_overlap_diagnostic_to_stderr(tmp_path, monkeypatch, capsys):
    _patch_runs(monkeypatch, {
        Path("rA.h5"): _two_window_run(bump=0.0),
        Path("rB.h5"): _two_window_run(bump=0.2),
    })
    rc = main([
        "--multi-run", "rA.h5", "rB.h5", "-o", str(tmp_path / "dos.csv"),
    ])
    assert rc == 0
    assert "overlap std window" in capsys.readouterr().err


def test_multi_run_notes_partial_window_coverage(tmp_path, monkeypatch, capsys):
    # Run B never populated the lower window; the consensus for that
    # window comes from run A alone, and the gap is flagged on stderr.
    rb = [
        _mock_dc({}, 0.5, -1.0, 0.0),
        _mock_dc({-1: 0.9, 0: 1.2, 1: 1.4}, 0.5, -0.5, 0.5),
    ]
    _patch_runs(monkeypatch, {
        Path("rA.h5"): _two_window_run(),
        Path("rB.h5"): rb,
    })
    out = tmp_path / "dos.csv"
    rc = main(["--multi-run", "rA.h5", "rB.h5", "-o", str(out)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "no data" in err.lower()
    assert "(-1.0, 0.0)" in err
    df = pd.read_csv(out)
    assert list(df.columns) == ["energy", "entropy"]
