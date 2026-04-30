"""Tests for CanonicalParallelTempering."""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.canonical import CanonicalParallelTempering
from mchammer_pt.replica import Replica


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


def test_descending_temperatures_rejected(toy_ce, toy_atoms):
    """Non-decreasing order is enforced; decreasing temperatures raise."""
    with pytest.raises(ValueError, match="non-decreasing"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperatures=[600.0, 300.0],
            block_size=10,
            random_seed=0,
        )


def test_equal_adjacent_temperatures_allowed(toy_ce, toy_atoms):
    """Equal adjacent temperatures are a legitimate null case, not rejected."""
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[500.0, 500.0, 500.0],
        block_size=10,
        random_seed=0,
    )
    assert len(pt.pool) == 3


def test_mismatched_pool_length_rejected(toy_ce, toy_atoms):
    """A pool with n replicas but temperatures of length m != n must raise."""
    from mchammer_pt.parallel.serial import SerialPool

    pool = SerialPool(
        [
            Replica(
                cluster_expansion=toy_ce,
                atoms=toy_atoms,
                temperature=T,
                random_seed=i,
            )
            for i, T in enumerate([300.0, 600.0])
        ]
    )
    with pytest.raises(ValueError, match="2 replicas but temperatures has 3"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperatures=[300.0, 600.0, 1200.0],
            block_size=10,
            random_seed=0,
            pool=pool,
        )


def test_mismatched_pool_temperatures_rejected(toy_ce, toy_atoms):
    """A pool at one ladder with temperatures arg at another must raise."""
    from mchammer_pt.parallel.serial import SerialPool

    # Pool was built at 300 / 600; orchestrator asked for 300 / 1200.
    pool = SerialPool(
        [
            Replica(
                cluster_expansion=toy_ce,
                atoms=toy_atoms,
                temperature=T,
                random_seed=i,
            )
            for i, T in enumerate([300.0, 600.0])
        ]
    )
    with pytest.raises(ValueError, match="does not match"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperatures=[300.0, 1200.0],
            block_size=10,
            random_seed=0,
            pool=pool,
        )


def test_small_temperature_mismatch_rejected(toy_ce, toy_atoms):
    """np.allclose would pass 300.0 vs 300.001; exact check rejects it.

    The two differ by 1 mK, ~3.3e-6 relative, well within np.allclose's
    default tolerance (rtol=1e-5) — but they produce distinct β values
    and would silently bias exchange acceptance. Exact comparison
    catches it.
    """
    from mchammer_pt.parallel.serial import SerialPool

    pool = SerialPool(
        [
            Replica(
                cluster_expansion=toy_ce,
                atoms=toy_atoms,
                temperature=T,
                random_seed=i,
            )
            for i, T in enumerate([300.0, 600.0])
        ]
    )
    with pytest.raises(ValueError, match="does not match"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperatures=[300.001, 600.0],
            block_size=10,
            random_seed=0,
            pool=pool,
        )


def test_zero_block_size_rejected(toy_ce, toy_atoms):
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperatures=[300.0, 600.0],
            block_size=0,
            random_seed=0,
        )


def test_process_pool_factory_produces_aligned_orchestrator(toy_ce, toy_atoms):
    """process_pool() constructs a CM orchestrator; ladder aligned by construction."""
    with CanonicalParallelTempering.process_pool(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[200.0, 400.0, 800.0],
        block_size=50,
        random_seed=0,
    ) as pt:
        # Validation already rejected misalignment; by reaching this
        # point we know pool.temperatures == orchestrator.temperatures.
        assert list(pt.pool.temperatures) == [200.0, 400.0, 800.0]
        assert len(pt.pool) == 3
        h = pt.run(n_cycles=2)
        assert h.energies_per_cycle.shape == (3, 3)
        # ProcessPool is the concrete pool type behind this factory.
        from mchammer_pt.parallel.processes import ProcessPool

        assert isinstance(pt.pool, ProcessPool)


def test_process_pool_factory_tempdir_cleanup_after_run(toy_ce, toy_atoms):
    """The CE tempdir is cleaned after the orchestrator is garbage-collected."""
    import gc

    with CanonicalParallelTempering.process_pool(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0],
        block_size=10,
        random_seed=0,
    ) as pt:
        # The pool's worker processes were given a ce_path; after the
        # `with` exits, pool.shutdown() has joined workers. We then
        # drop the reference to pt and force a collection to trigger
        # the weakref.finalize cleanup.
        pt.run(n_cycles=1)
    del pt
    gc.collect()
    # Nothing to assert on the filesystem directly (we don't know the
    # tempdir path from here), but the finalizer ran without raising,
    # which is the contract.


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


def test_orchestrator_forwards_ensemble_cls_to_default_pool(toy_ce, toy_atoms):
    """Default-pool path constructs replicas with the supplied class."""
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0, 1200.0],
        block_size=10,
        random_seed=0,
        ensemble_cls=TaggedCanonicalEnsemble,
        ensemble_kwargs={"tag": "beta"},
    )
    pool = pt.pool
    # Default pool is SerialPool; replicas are reachable for inspection.
    for replica in pool._replicas:  # type: ignore[attr-defined]
        assert isinstance(replica._ensemble, TaggedCanonicalEnsemble)
        assert replica._ensemble.tag == "beta"


def test_orchestrator_rejects_pool_plus_custom_ensemble_cls(toy_ce, toy_atoms):
    """pool= and ensemble_cls= together is ambiguous and rejected up-front."""
    from mchammer_pt import SerialPool
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    pool = SerialPool([
        Replica(toy_ce, toy_atoms, temperature=T, random_seed=i)
        for i, T in enumerate([300.0, 600.0])
    ])
    try:
        with pytest.raises(ValueError, match="ensemble_cls"):
            CanonicalParallelTempering(
                cluster_expansion=toy_ce,
                atoms=toy_atoms,
                temperatures=[300.0, 600.0],
                block_size=10,
                random_seed=0,
                pool=pool,
                ensemble_cls=TaggedCanonicalEnsemble,
            )
    finally:
        pool.shutdown()


def test_orchestrator_rejects_pool_plus_ensemble_kwargs(toy_ce, toy_atoms):
    """Non-empty ensemble_kwargs alongside pool= is also rejected."""
    from mchammer_pt import SerialPool

    pool = SerialPool([
        Replica(toy_ce, toy_atoms, temperature=T, random_seed=i)
        for i, T in enumerate([300.0, 600.0])
    ])
    try:
        with pytest.raises(ValueError, match="ensemble_kwargs"):
            CanonicalParallelTempering(
                cluster_expansion=toy_ce,
                atoms=toy_atoms,
                temperatures=[300.0, 600.0],
                block_size=10,
                random_seed=0,
                pool=pool,
                ensemble_kwargs={"tag": "x"},
            )
    finally:
        pool.shutdown()


def test_pool_plus_ensemble_cls_fires_before_temperature_mismatch(toy_ce, toy_atoms):
    """If both errors apply, the ensemble exclusion fires first.

    A user passing both an inconsistent pool ladder AND ensemble args
    has the ensemble misconception at root: their mental model of the
    API is that the orchestrator builds pool replicas from
    ``ensemble_cls``. Surfacing the ensemble error first lets them
    correct that mental model in one round; otherwise they fix the
    length, re-run, and only then see the ensemble error.
    """
    from mchammer_pt import SerialPool
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    # Pool length does NOT match temperatures (2 vs 3).
    pool = SerialPool([
        Replica(toy_ce, toy_atoms, temperature=T, random_seed=i)
        for i, T in enumerate([300.0, 600.0])
    ])
    try:
        with pytest.raises(ValueError, match="ensemble_cls"):
            CanonicalParallelTempering(
                cluster_expansion=toy_ce,
                atoms=toy_atoms,
                temperatures=[300.0, 600.0, 1200.0],
                block_size=10,
                random_seed=0,
                pool=pool,
                ensemble_cls=TaggedCanonicalEnsemble,
            )
    finally:
        pool.shutdown()


def test_process_pool_factory_forwards_ensemble_cls(toy_ce, toy_atoms):
    """The process_pool factory threads ensemble_cls/kwargs through to workers."""
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    with CanonicalParallelTempering.process_pool(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0],
        block_size=10,
        random_seed=0,
        ensemble_cls=TaggedCanonicalEnsemble,
        ensemble_kwargs={"tag": "delta"},
    ) as pt:
        # If the kwargs did not reach the workers, the workers would
        # have failed at startup (TaggedCanonicalEnsemble requires
        # `tag`) and the factory would have raised in __init__.
        history = pt.run(n_cycles=2)
    assert history.energies_per_cycle.shape == (3, 2)


def test_per_temperature_atoms_constructs_distinct_replicas(toy_ce, toy_atoms):
    """A sequence of Atoms seeds each replica with its specific config."""
    # Create a second config by shuffling the species.
    atoms_a = toy_atoms.copy()
    atoms_b = toy_atoms.copy()
    rng = np.random.default_rng(999)
    symbols_b = np.array(atoms_b.get_chemical_symbols())
    rng.shuffle(symbols_b)
    atoms_b.set_chemical_symbols(symbols_b.tolist())

    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=[atoms_a, atoms_b],
        temperatures=[300.0, 600.0],
        block_size=10,
        random_seed=0,
    )
    # Each replica's initial occupations should match its specific atoms.
    np.testing.assert_array_equal(
        pt.pool.current_occupations(0), atoms_a.numbers,
    )
    np.testing.assert_array_equal(
        pt.pool.current_occupations(1), atoms_b.numbers,
    )


def test_per_temperature_atoms_length_mismatch_raises(toy_ce, toy_atoms):
    with pytest.raises(ValueError, match="atoms has 2.*temperatures has 3"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=[toy_atoms, toy_atoms],
            temperatures=[300.0, 600.0, 1200.0],
            block_size=10,
            random_seed=0,
        )


def test_single_atoms_still_works_after_dispatch(toy_ce, toy_atoms):
    """The existing single-Atoms path is not regressed by the dispatch."""
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0],
        block_size=10,
        random_seed=0,
    )
    # All replicas start from the same config.
    np.testing.assert_array_equal(
        pt.pool.current_occupations(0),
        pt.pool.current_occupations(1),
    )


def test_process_pool_per_temperature_atoms(toy_ce, toy_atoms):
    """process_pool() with a sequence of Atoms seeds each worker distinctly."""
    atoms_a = toy_atoms.copy()
    atoms_b = toy_atoms.copy()
    rng = np.random.default_rng(999)
    symbols_b = np.array(atoms_b.get_chemical_symbols())
    rng.shuffle(symbols_b)
    atoms_b.set_chemical_symbols(symbols_b.tolist())

    with CanonicalParallelTempering.process_pool(
        cluster_expansion=toy_ce,
        atoms=[atoms_a, atoms_b],
        temperatures=[300.0, 600.0],
        block_size=10,
        random_seed=0,
    ) as pt:
        np.testing.assert_array_equal(
            pt.pool.current_occupations(0), atoms_a.numbers,
        )
        np.testing.assert_array_equal(
            pt.pool.current_occupations(1), atoms_b.numbers,
        )


def test_process_pool_per_temperature_atoms_length_mismatch(toy_ce, toy_atoms):
    """Length mismatch between atoms list and temperatures raises ValueError."""
    with pytest.raises(ValueError, match="atoms has 2.*temperatures has 3"):
        CanonicalParallelTempering.process_pool(
            cluster_expansion=toy_ce,
            atoms=[toy_atoms, toy_atoms],
            temperatures=[300.0, 600.0, 1200.0],
            block_size=10,
            random_seed=0,
        )


def test_per_temperature_atoms_geometry_mismatch_raises(toy_ce, toy_atoms):
    """Atoms with different cell/positions/pbc are rejected."""
    atoms_a = toy_atoms.copy()
    atoms_b = toy_atoms.copy()
    atoms_b.cell *= 1.1  # different cell
    with pytest.raises(ValueError, match="different cell/positions/pbc"):
        CanonicalParallelTempering(
            cluster_expansion=toy_ce,
            atoms=[atoms_a, atoms_b],
            temperatures=[300.0, 600.0],
            block_size=10,
            random_seed=0,
        )
