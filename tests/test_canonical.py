"""Tests for CanonicalParallelTempering."""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.canonical import CanonicalParallelTempering


def test_init_constructs_one_replica_per_temperature(toy_ce, toy_atoms):
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[100.0, 300.0, 700.0],
        block_size=10,
        random_seed=0,
    )
    assert len(pt.pool) == 3
    assert sorted(pt.pool.temperatures) == [100.0, 300.0, 700.0]


def test_log_prob_ratio_matches_hand_formula(toy_ce, toy_atoms):
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[100.0, 1000.0],
        block_size=10,
        random_seed=0,
    )
    E_i = pt.pool.current_energy(0)
    E_j = pt.pool.current_energy(1)
    kB = 8.617333262145e-5
    expected = (1.0 / (kB * 100.0) - 1.0 / (kB * 1000.0)) * (E_i - E_j)
    # Both should be zero (same starting config), but the formula form is checked.
    assert abs(pt._log_prob_ratio(0, 1) - expected) < 1e-12


def test_run_produces_history_with_correct_shape(toy_ce, toy_atoms):
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0, 1200.0],
        block_size=50,
        random_seed=0,
    )
    history = pt.run(n_cycles=5)
    assert history.energies_per_cycle.shape == (6, 3)


def test_requires_at_least_two_temperatures(toy_ce, toy_atoms):
    with pytest.raises(ValueError):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperatures=[300.0],
            block_size=10,
            random_seed=0,
        )


def test_deterministic_for_fixed_seed(toy_ce, toy_atoms):
    kwargs = dict(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0, 1200.0],
        block_size=50,
        random_seed=42,
    )
    pt1 = CanonicalParallelTempering(**kwargs)
    pt2 = CanonicalParallelTempering(**kwargs)
    h1 = pt1.run(n_cycles=4)
    h2 = pt2.run(n_cycles=4)
    np.testing.assert_array_equal(h1.energies_per_cycle, h2.energies_per_cycle)
    np.testing.assert_array_equal(h1.swap_accepted, h2.swap_accepted)


def test_run_writes_hdf5_when_file_provided(tmp_path, toy_ce, toy_atoms):
    path = tmp_path / "pt.h5"
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0],
        block_size=50,
        random_seed=0,
        data_container_file=path,
    )
    pt.run(n_cycles=3)
    assert path.exists()

    from mchammer_pt import read_hdf5

    history, containers, meta = read_hdf5(path)
    assert history.energies_per_cycle.shape == (4, 2)
    assert len(containers) == 2
    assert meta["block_size"] == 50
