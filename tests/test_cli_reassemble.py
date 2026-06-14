"""Tests for mchammer_pt.cli.reassemble."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt.cli.reassemble import reassemble_pieces

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
