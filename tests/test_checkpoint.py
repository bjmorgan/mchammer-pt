"""Tests for the checkpoint/resume machinery."""

from __future__ import annotations

import random as stdlib_random

import numpy as np
from mchammer.data_containers.base_data_container import BaseDataContainer

from mchammer_pt.replica import Replica


def test_replica_restart_from_restores_per_replica_state(
    toy_ce, toy_atoms, tmp_path
):
    """`Replica.restart_from(container)` restores step count, accepted-trial
    count, occupations, and stdlib `random` state from the saved container."""
    # Drive a fresh replica for some steps, then write its data container
    # to disk. mchammer populates `_last_state` only inside
    # `write_data_container`, so a round-trip through disk is the
    # realistic way to obtain a container with the saved per-replica state.
    original = Replica(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperature=300.0,
        random_seed=0,
    )
    original.advance(n_steps=50)

    dc_path = tmp_path / "replica.dc"
    original.ensemble.write_data_container(str(dc_path))
    container = BaseDataContainer.read(str(dc_path))

    saved_step = container._last_state["last_step"]
    saved_accepted = container._last_state["accepted_trials"]
    saved_occupations = container._last_state["occupations"]
    saved_random_state = container._last_state["random_state"]

    # Reconstruct via restart_from. The new replica should match the
    # saved state exactly.
    restored = Replica.restart_from(
        container,
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperature=300.0,
        random_seed=0,
    )

    assert restored.ensemble._step == saved_step
    assert restored.ensemble._accepted_trials == saved_accepted
    np.testing.assert_array_equal(
        restored.current_occupations(), saved_occupations
    )
    # The replica's saved RNG snapshot was taken at the end of
    # restart_from; restoring it to stdlib random and comparing
    # tuples confirms the state was actually pulled in.
    caller_state = stdlib_random.getstate()
    try:
        stdlib_random.setstate(restored._rng_state)
        assert stdlib_random.getstate() == saved_random_state
    finally:
        stdlib_random.setstate(caller_state)


def test_compute_ce_identity_is_deterministic_for_same_ce(toy_ce):
    """Hashing the same CE twice gives the same digest."""
    from mchammer_pt.checkpoint import _compute_ce_identity

    assert _compute_ce_identity(toy_ce) == _compute_ce_identity(toy_ce)


def test_compute_ce_identity_differs_for_different_parameters(
    toy_ce, toy_cluster_space
):
    """Different parameters → different digest."""
    from icet import ClusterExpansion

    from mchammer_pt.checkpoint import _compute_ce_identity

    other = ClusterExpansion(
        cluster_space=toy_cluster_space,
        parameters=np.zeros(len(toy_cluster_space)),
    )
    assert _compute_ce_identity(toy_ce) != _compute_ce_identity(other)


def test_compute_ce_identity_differs_for_different_chemistry(toy_ce):
    """Different chemical_symbols → different digest."""
    from ase.build import bulk
    from icet import ClusterExpansion, ClusterSpace

    from mchammer_pt.checkpoint import _compute_ce_identity

    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    other_cs = ClusterSpace(
        structure=primitive,
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Ag"],
    )
    other_ce = ClusterExpansion(
        cluster_space=other_cs,
        parameters=np.zeros(len(other_cs)),
    )
    # toy_ce uses Cu/Au; reconstruct an Au-version for a fair comparison
    # with matching parameter vector length.
    toy_au_cs = ClusterSpace(
        structure=primitive,
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )
    toy_au_ce = ClusterExpansion(
        cluster_space=toy_au_cs,
        parameters=np.zeros(len(toy_au_cs)),
    )
    assert _compute_ce_identity(other_ce) != _compute_ce_identity(toy_au_ce)


def test_compute_ce_identity_differs_for_different_cutoffs():
    """Different cutoffs → different digest."""
    from ase.build import bulk
    from icet import ClusterExpansion, ClusterSpace

    from mchammer_pt.checkpoint import _compute_ce_identity

    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    cs_short = ClusterSpace(
        structure=primitive, cutoffs=[3.5], chemical_symbols=["Cu", "Au"]
    )
    cs_long = ClusterSpace(
        structure=primitive, cutoffs=[4.5], chemical_symbols=["Cu", "Au"]
    )
    ce_short = ClusterExpansion(
        cluster_space=cs_short, parameters=np.zeros(len(cs_short))
    )
    ce_long = ClusterExpansion(
        cluster_space=cs_long, parameters=np.zeros(len(cs_long))
    )
    assert _compute_ce_identity(ce_short) != _compute_ce_identity(ce_long)


def test_compute_ce_identity_differs_for_different_primitive_structure():
    """Different primitive structure → different digest."""
    from ase.build import bulk
    from icet import ClusterExpansion, ClusterSpace

    from mchammer_pt.checkpoint import _compute_ce_identity

    cs_a = ClusterSpace(
        structure=bulk("Cu", "fcc", a=4.0, cubic=True),
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )
    cs_b = ClusterSpace(
        structure=bulk("Cu", "fcc", a=4.1, cubic=True),  # different lattice constant
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )
    ce_a = ClusterExpansion(
        cluster_space=cs_a, parameters=np.zeros(len(cs_a))
    )
    ce_b = ClusterExpansion(
        cluster_space=cs_b, parameters=np.zeros(len(cs_b))
    )
    assert _compute_ce_identity(ce_a) != _compute_ce_identity(ce_b)


def test_compute_ensemble_kwargs_hash_handles_picklable_and_unpicklable():
    """Picklable kwargs hash deterministically; unpicklable kwargs return the
    sentinel ``""``."""
    from mchammer_pt.checkpoint import _compute_ensemble_kwargs_hash

    # None and {} both hash to the same canonical empty value.
    assert (
        _compute_ensemble_kwargs_hash(None)
        == _compute_ensemble_kwargs_hash({})
    )
    # Picklable kwargs hash deterministically.
    h1 = _compute_ensemble_kwargs_hash({"a": 1, "b": "x"})
    h2 = _compute_ensemble_kwargs_hash({"a": 1, "b": "x"})
    assert h1 == h2 and h1 != ""
    # Different picklable kwargs give different hashes.
    assert _compute_ensemble_kwargs_hash({"a": 1}) != _compute_ensemble_kwargs_hash(
        {"a": 2}
    )

    # Unpicklable kwargs return the sentinel.
    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    assert _compute_ensemble_kwargs_hash({"x": _Unpicklable()}) == ""


def test_compute_ensemble_kwargs_hash_returns_sentinel_for_local_class():
    """Instances of locally-defined classes return the sentinel rather than
    crashing — pickle can't resolve their qualified name."""
    from mchammer_pt.checkpoint import _compute_ensemble_kwargs_hash

    class _LocalClass:
        pass

    # pickle raises AttributeError (or similar) for local-class instances;
    # the broadened except in `_compute_ensemble_kwargs_hash` should catch.
    assert _compute_ensemble_kwargs_hash({"x": _LocalClass()}) == ""
