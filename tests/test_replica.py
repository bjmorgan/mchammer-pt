"""Tests for the Replica per-temperature ensemble handle."""
from __future__ import annotations

import numpy as np

from mchammer_pt.replica import Replica


def test_replica_reports_its_temperature(toy_ce, toy_atoms):
    rep = Replica(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperature=500.0,
        random_seed=1,
    )
    assert rep.temperature == 500.0


def test_current_energy_matches_ce_predict(toy_ce, toy_atoms):
    rep = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    expected = float(toy_ce.predict(toy_atoms)) * len(toy_atoms)
    assert abs(rep.current_energy() - expected) < 1e-10


def test_current_occupations_returns_a_copy(toy_ce, toy_atoms):
    rep = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    occ = rep.current_occupations()
    occ[0] = -999
    assert rep.current_occupations()[0] != -999


def test_advance_changes_occupations_at_high_T(toy_ce, toy_atoms):
    rep = Replica(toy_ce, toy_atoms, temperature=100_000.0, random_seed=1)
    before = rep.current_occupations().copy()
    rep.advance(n_steps=500)
    after = rep.current_occupations()
    # At absurd T most proposed swaps are accepted, so the config must drift.
    assert np.any(before != after)


def test_set_occupations_updates_energy_correctly(toy_ce, toy_atoms):
    rep_a = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    rep_b = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=2)
    rep_b.advance(500)
    occ_b = rep_b.current_occupations()
    energy_b = rep_b.current_energy()

    rep_a.set_occupations(occ_b)
    assert np.array_equal(rep_a.current_occupations(), occ_b)
    assert abs(rep_a.current_energy() - energy_b) < 1e-10


def test_data_container_is_mchammer_native(toy_ce, toy_atoms):
    from mchammer.data_containers.base_data_container import BaseDataContainer

    rep = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    rep.advance(50)
    assert isinstance(rep.data_container(), BaseDataContainer)


def test_attach_mchammer_observer_fires_during_advance(toy_ce, toy_atoms):
    """Attached observers must fire inside advance; not just be accepted.

    Confirms the observer pipeline is wired, not merely the type check.
    """
    from mchammer.observers.base_observer import BaseObserver

    class CountingObserver(BaseObserver):
        def __init__(self, interval: int) -> None:
            super().__init__(interval=interval, return_type=int, tag="counter")
            self.n_calls = 0

        def get_observable(self, structure) -> int:
            self.n_calls += 1
            return self.n_calls

    rep = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    observer = CountingObserver(interval=10)
    rep.attach_mchammer_observer(observer)
    rep.advance(n_steps=50)
    assert observer.n_calls > 0


def test_co_tenant_replicas_have_independent_rng_streams(toy_ce, toy_atoms):
    """Two Replicas built in the same process must evolve independently.

    mchammer drives its MC from Python's global `random` module. Without
    per-Replica RNG isolation, constructing a second Replica would reseed
    the shared stream and the first Replica's `advance()` would pick up
    the wrong trajectory. Pin the contract: seed=1 alone produces the
    same trajectory as seed=1 in the presence of a co-tenant seed=2.
    """
    # Reference trajectory: one Replica, alone.
    solo = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    solo.advance(n_steps=200)
    solo_occ = solo.current_occupations()

    # Two Replicas in the same process. Construct the second with a
    # different seed before advancing the first — this tests that the
    # second construction does not clobber the first's RNG state.
    co_first = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    _co_other = Replica(toy_ce, toy_atoms, temperature=500.0, random_seed=2)
    co_first.advance(n_steps=200)

    np.testing.assert_array_equal(solo_occ, co_first.current_occupations())
