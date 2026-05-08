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
