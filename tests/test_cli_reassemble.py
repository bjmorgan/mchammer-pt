"""Tests for mchammer_pt.cli.reassemble."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt.cli.reassemble import main, reassemble_pieces
from mchammer_pt.history import ExchangeHistory, read_hdf5, write_hdf5

WL_FQN = "mchammer.ensembles.wang_landau_ensemble.WangLandauEnsemble"


def _mock_dc(lo, hi, *, energy_spacing=0.5, n_sites=32):
    dc = MagicMock(spec=WangLandauDataContainer)
    dc.structure = [0] * n_sites
    dc.ensemble_parameters = {
        "energy_spacing": energy_spacing,
        "energy_limit_left": lo,
        "energy_limit_right": hi,
    }
    return dc


def _piece(label, dcs, *, ce="ce-hash", spacing=0.5, fqn=WL_FQN):
    meta = {
        "ensemble_cls_fqn": fqn,
        "ce_identity": ce,
        "energy_spacing": spacing,
    }
    return (label, meta, dcs)


def test_reassemble_unions_disjoint_windows():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0)])
    b = _piece("B.h5", [_mock_dc(0.0, 1.0)])
    containers, meta = reassemble_pieces([a, b])
    assert len(containers) == 2
    # window-sorted: (-1, 0) then (0, 1)
    assert containers[0].ensemble_parameters["energy_limit_left"] == -1.0
    assert containers[1].ensemble_parameters["energy_limit_left"] == 0.0


def test_reassemble_preserves_multi_walker_multiplicity():
    # Two walkers in one window inside a single piece are legitimate and
    # must both survive into the union, not be flagged as a collision.
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0), _mock_dc(-1.0, 0.0)])
    b = _piece("B.h5", [_mock_dc(0.0, 1.0)])
    containers, meta = reassemble_pieces([a, b])
    assert len(containers) == 3
    np.testing.assert_array_equal(
        np.asarray(meta["walkers_per_window"]), np.array([2, 1])
    )


def test_reassemble_output_meta_describes_the_union():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0)], ce="cafe", spacing=0.5)
    b = _piece("B.h5", [_mock_dc(0.0, 1.0)], ce="cafe", spacing=0.5)
    _, meta = reassemble_pieces([a, b])
    assert "WangLandau" in str(meta["ensemble_cls_fqn"])
    assert meta["ce_identity"] == "cafe"
    assert float(meta["energy_spacing"]) == 0.5
    assert bool(meta["reassembled"]) is True
    np.testing.assert_allclose(
        np.asarray(meta["windows"]), np.array([[-1.0, 0.0], [0.0, 1.0]])
    )


def test_reassemble_rejects_fewer_than_two_pieces():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0)])
    with pytest.raises(ValueError, match="at least two checkpoint pieces"):
        reassemble_pieces([a])


def test_reassemble_rejects_non_wl_checkpoint():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0)],
               fqn="mchammer.ensembles.CanonicalEnsemble")
    b = _piece("B.h5", [_mock_dc(0.0, 1.0)])
    with pytest.raises(ValueError, match="not a Wang-Landau checkpoint"):
        reassemble_pieces([a, b])


def test_reassemble_rejects_ce_mismatch():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0)], ce="aaaa")
    b = _piece("B.h5", [_mock_dc(0.0, 1.0)], ce="bbbb")
    with pytest.raises(ValueError, match="cluster-expansion identity"):
        reassemble_pieces([a, b])


def test_reassemble_rejects_system_size_mismatch():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0, n_sites=32)])
    b = _piece("B.h5", [_mock_dc(0.0, 1.0, n_sites=64)])
    with pytest.raises(ValueError, match="system size"):
        reassemble_pieces([a, b])


def test_reassemble_rejects_energy_spacing_mismatch():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0, energy_spacing=0.5)], spacing=0.5)
    b = _piece("B.h5", [_mock_dc(0.0, 1.0, energy_spacing=0.25)], spacing=0.25)
    with pytest.raises(ValueError, match="energy_spacing"):
        reassemble_pieces([a, b])


def test_reassemble_rejects_window_collision_with_multirun_hint():
    a = _piece("A.h5", [_mock_dc(-1.0, 0.0)])
    b = _piece("B.h5", [_mock_dc(-1.0, 0.0)])  # same window key as A
    with pytest.raises(ValueError) as exc:
        reassemble_pieces([a, b])
    msg = str(exc.value)
    assert "appears in more than one input" in msg
    assert "A.h5" in msg and "B.h5" in msg
    assert "--multi-run" in msg


def _patch_reads(monkeypatch, mapping):
    monkeypatch.setattr(
        "mchammer_pt.cli.reassemble.read_hdf5",
        lambda p: (None, mapping[Path(p)][2], mapping[Path(p)][1]),
    )


def test_main_requires_two_inputs(tmp_path, capsys):
    rc = main(["only.h5", "-o", str(tmp_path / "out.h5")])
    assert rc == 2
    assert "at least two checkpoint pieces" in capsys.readouterr().err


def test_main_propagates_read_error(tmp_path, monkeypatch, capsys):
    def boom(_):
        raise OSError("nope")

    monkeypatch.setattr("mchammer_pt.cli.reassemble.read_hdf5", boom)
    rc = main(["a.h5", "b.h5", "-o", str(tmp_path / "out.h5")])
    assert rc == 2
    assert "could not read checkpoint" in capsys.readouterr().err


def test_main_reports_collision(tmp_path, monkeypatch, capsys):
    a = _piece("a.h5", [_mock_dc(-1.0, 0.0)])
    b = _piece("b.h5", [_mock_dc(-1.0, 0.0)])
    _patch_reads(monkeypatch, {Path("a.h5"): a, Path("b.h5"): b})
    rc = main(["a.h5", "b.h5", "-o", str(tmp_path / "out.h5")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "appears in more than one input" in err
    assert "--multi-run" in err


def test_main_reports_ce_mismatch(tmp_path, monkeypatch, capsys):
    a = _piece("a.h5", [_mock_dc(-1.0, 0.0)], ce="aaaa")
    b = _piece("b.h5", [_mock_dc(0.0, 1.0)], ce="bbbb")
    _patch_reads(monkeypatch, {Path("a.h5"): a, Path("b.h5"): b})
    rc = main(["a.h5", "b.h5", "-o", str(tmp_path / "out.h5")])
    assert rc == 2
    assert "cluster-expansion identity" in capsys.readouterr().err


def _real_atoms():
    return bulk("Au", "fcc", a=4.0, cubic=True).repeat((2, 2, 2))


def _real_wl_container(atoms, lo, hi, spacing, entropy):
    dc = WangLandauDataContainer(
        structure=atoms.copy(),
        ensemble_parameters={
            "energy_spacing": spacing,
            "energy_limit_left": lo,
            "energy_limit_right": hi,
            "trial_move": "swap",
        },
    )
    dc._last_state = {
        "entropy": dict(entropy),
        "histogram": {k: 1 for k in entropy},
        "fill_factor": 0.5,
        "fill_factor_history": {},
        "entropy_history": {},
    }
    return dc


def _four_windows(atoms, spacing=1.0):
    # Four windows on a unit grid, each overlapping its neighbour by one
    # bin (bins -1, 1, 3). The shared boundary-bin entropies are
    # deliberately equal across the two windows that share them (e.g. bin
    # -1 is 1.0 in both windows 0 and 1), so stitch aligns them with zero
    # offset and no gaps.
    return [
        _real_wl_container(atoms, -3.5, -0.5, spacing,
                           {-3: 0.0, -2: 0.5, -1: 1.0}),
        _real_wl_container(atoms, -1.5, 1.5, spacing,
                           {-1: 1.0, 0: 1.5, 1: 2.0}),
        _real_wl_container(atoms, 0.5, 3.5, spacing,
                           {1: 2.0, 2: 2.5, 3: 3.0}),
        _real_wl_container(atoms, 2.5, 5.5, spacing,
                           {3: 3.0, 4: 3.5, 5: 4.0}),
    ]


def _write_checkpoint(path, containers, *, ce="ce-hash", spacing=1.0,
                      fqn=WL_FQN):
    write_hdf5(
        path,
        history=ExchangeHistory.empty(n_cycles=0, n_replicas=len(containers)),
        replica_containers=containers,
        meta={
            "ensemble_cls_fqn": fqn,
            "ce_identity": ce,
            "energy_spacing": spacing,
        },
    )


def test_reassemble_then_stitch_matches_single_checkpoint(tmp_path):
    from mchammer_pt.cli.stitch import main as stitch_main

    atoms = _real_atoms()
    windows = _four_windows(atoms)

    complete = tmp_path / "complete.h5"
    _write_checkpoint(complete, windows)

    piece_a = tmp_path / "pieceA.h5"
    piece_b = tmp_path / "pieceB.h5"
    _write_checkpoint(piece_a, windows[:2])
    _write_checkpoint(piece_b, windows[2:])

    reassembled = tmp_path / "reassembled.h5"
    assert main([str(piece_a), str(piece_b), "-o", str(reassembled)]) == 0

    dos_complete = tmp_path / "dos_complete.csv"
    dos_reassembled = tmp_path / "dos_reassembled.csv"
    assert stitch_main([str(complete), "-o", str(dos_complete)]) == 0
    assert stitch_main([str(reassembled), "-o", str(dos_reassembled)]) == 0

    df_c = pd.read_csv(dos_complete)
    df_r = pd.read_csv(dos_reassembled)
    np.testing.assert_allclose(df_c["energy"], df_r["energy"])
    np.testing.assert_allclose(df_c["entropy"], df_r["entropy"], atol=1e-12)


def test_main_writes_analysis_only_artifact(tmp_path):
    atoms = _real_atoms()
    windows = _four_windows(atoms)
    piece_a = tmp_path / "pieceA.h5"
    piece_b = tmp_path / "pieceB.h5"
    _write_checkpoint(piece_a, windows[:2])
    _write_checkpoint(piece_b, windows[2:])

    out = tmp_path / "reassembled.h5"
    assert main([str(piece_a), str(piece_b), "-o", str(out)]) == 0

    _, containers, meta = read_hdf5(out)
    assert len(containers) == 4
    assert bool(meta["reassembled"]) is True
    with h5py.File(out, "r") as f:
        assert "orchestrator" not in f
        assert "sites_by_species" not in f


def test_resume_refuses_reassembled_artifact(tmp_path, toy_ce, toy_atoms):
    from mchammer_pt.checkpoint import _compute_ce_identity
    from mchammer_pt.wl import WangLandauParallelTempering
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce_hash = _compute_ce_identity(toy_ce)
    fqn = (
        f"{CoordinatedWangLandauEnsemble.__module__}."
        f"{CoordinatedWangLandauEnsemble.__qualname__}"
    )
    windows = _four_windows(toy_atoms)
    piece_a = tmp_path / "pieceA.h5"
    piece_b = tmp_path / "pieceB.h5"
    _write_checkpoint(piece_a, windows[:2], ce=ce_hash, fqn=fqn)
    _write_checkpoint(piece_b, windows[2:], ce=ce_hash, fqn=fqn)

    out = tmp_path / "reassembled.h5"
    assert main([str(piece_a), str(piece_b), "-o", str(out)]) == 0

    # The artifact carries the real CE identity and ensemble FQN, so resume
    # passes the identity checks and fails specifically on the missing
    # run-execution metadata (block_size) -- it is not resumable.
    with pytest.raises(KeyError):
        WangLandauParallelTempering.resume(out, cluster_expansion=toy_ce)
