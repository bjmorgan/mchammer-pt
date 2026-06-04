"""Tests for mchammer_pt.cli.stitch_multirun."""
from __future__ import annotations

from pathlib import Path

import pytest

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
