"""Tests for ExchangeHistory dataclass and HDF5 round-trip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mchammer_pt.history import (
    ExchangeHistory,
    read_hdf5,
    write_hdf5,
)


def _make_history(n_cycles: int = 4, n_replicas: int = 3) -> ExchangeHistory:
    h = ExchangeHistory.empty(n_cycles=n_cycles, n_replicas=n_replicas)
    h.energies_per_cycle[:] = np.arange(h.energies_per_cycle.size).reshape(
        h.energies_per_cycle.shape
    )
    h.replica_labels_per_cycle[:] = np.tile(
        np.arange(n_replicas), (n_cycles + 1, 1)
    )
    h.swap_attempted[:] = 10
    h.swap_accepted[:] = 5
    return h


def test_empty_has_correct_shapes():
    h = ExchangeHistory.empty(n_cycles=5, n_replicas=4)
    assert h.energies_per_cycle.shape == (6, 4)
    assert h.replica_labels_per_cycle.shape == (6, 4)
    assert h.swap_attempted.shape == (3,)
    assert h.swap_accepted.shape == (3,)


def test_hdf5_round_trip(tmp_path: Path):
    h = _make_history()
    meta = {
        "temperatures": np.array([100.0, 200.0, 400.0]),
        "block_size": 100,
        "random_seed": 0,
    }
    write_hdf5(tmp_path / "pt.h5", history=h, replica_containers=[], meta=meta)
    h_back, containers_back, meta_back = read_hdf5(tmp_path / "pt.h5")
    np.testing.assert_array_equal(h.energies_per_cycle, h_back.energies_per_cycle)
    np.testing.assert_array_equal(
        h.replica_labels_per_cycle, h_back.replica_labels_per_cycle
    )
    np.testing.assert_array_equal(h.swap_attempted, h_back.swap_attempted)
    np.testing.assert_array_equal(h.swap_accepted, h_back.swap_accepted)
    np.testing.assert_array_equal(meta["temperatures"], meta_back["temperatures"])
    assert meta_back["block_size"] == 100
    assert meta_back["random_seed"] == 0
    assert containers_back == []


def test_hdf5_round_trip_with_containers(tmp_path: Path, toy_ce, toy_atoms):
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.ensembles import CanonicalEnsemble

    # Build two throwaway ensembles, run briefly, snapshot their data containers.
    containers = []
    for seed in (1, 2):
        calc = ClusterExpansionCalculator(toy_atoms.copy(), toy_ce)
        ens = CanonicalEnsemble(
            structure=toy_atoms.copy(),
            calculator=calc,
            temperature=500.0,
            random_seed=seed,
        )
        ens.run(20)
        containers.append(ens.data_container)

    h = _make_history(n_cycles=2, n_replicas=2)
    write_hdf5(
        tmp_path / "pt.h5",
        history=h,
        replica_containers=containers,
        meta={"temperatures": np.array([100.0, 400.0])},
    )
    _, containers_back, _ = read_hdf5(tmp_path / "pt.h5")
    assert len(containers_back) == 2
    # BaseDataContainer reports the number of rows via `data`.
    assert len(containers_back[0].data) > 0


def test_read_hdf5_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_hdf5(tmp_path / "nope.h5")


def test_meta_round_trips_all_supported_scalar_types(tmp_path: Path):
    """Pin that the documented meta types survive round-trip."""
    h = _make_history(n_cycles=1, n_replicas=2)
    meta = {
        "a_str": "hello",
        "an_int": 42,
        "a_float": 3.14,
        "a_bool_true": True,
        "a_bool_false": False,
        "an_array": np.array([1.0, 2.0, 3.0]),
    }
    write_hdf5(tmp_path / "pt.h5", history=h, replica_containers=[], meta=meta)
    _, _, meta_back = read_hdf5(tmp_path / "pt.h5")
    assert meta_back["a_str"] == "hello"
    assert meta_back["an_int"] == 42
    assert meta_back["a_float"] == 3.14
    assert bool(meta_back["a_bool_true"]) is True
    assert bool(meta_back["a_bool_false"]) is False
    np.testing.assert_array_equal(meta_back["an_array"], meta["an_array"])


def test_write_hdf5_overwrites_existing_file(tmp_path: Path):
    """A second write to the same path truncates and replaces."""
    path = tmp_path / "pt.h5"
    first = _make_history(n_cycles=1, n_replicas=2)
    first.swap_attempted[:] = 1
    write_hdf5(path, history=first, replica_containers=[], meta={"tag": "first"})

    second = _make_history(n_cycles=1, n_replicas=2)
    second.swap_attempted[:] = 99
    write_hdf5(path, history=second, replica_containers=[], meta={"tag": "second"})

    history_back, _, meta_back = read_hdf5(path)
    np.testing.assert_array_equal(history_back.swap_attempted, [99])
    assert meta_back["tag"] == "second"
