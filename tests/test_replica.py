"""Tests for the Replica per-temperature ensemble handle."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_ensemble_property_returns_underlying_canonical_ensemble(toy_ce, toy_atoms):
    """`Replica.ensemble` returns the underlying mchammer ensemble."""
    from mchammer.ensembles import CanonicalEnsemble

    rep = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    assert isinstance(rep.ensemble, CanonicalEnsemble)
    assert rep.ensemble is rep._ensemble


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


def test_rng_isolation_survives_interleaved_advance(toy_ce, toy_atoms):
    """Advancing two co-tenant Replicas in interleaved chunks is equivalent
    to advancing each in one go.

    This is the real contract the save/restore dance has to honour:
    not just "construct-time isolation" but "interleaved-advance
    isolation across the same shared global RNG". If the save/restore
    were missing or applied only at construction, this test would
    diverge because replica A's RNG state would advance in-place while
    replica B draws, and then replica A's next advance would pick up
    B's increments.
    """
    # Reference trajectory: replica 1 advanced alone for 400 steps.
    ref1 = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    ref1.advance(n_steps=400)
    ref1_occ = ref1.current_occupations()

    # Reference trajectory: replica 2 advanced alone for 400 steps.
    ref2 = Replica(toy_ce, toy_atoms, temperature=500.0, random_seed=2)
    ref2.advance(n_steps=400)
    ref2_occ = ref2.current_occupations()

    # Interleaved schedule: replica 1 and 2 share the process; advance
    # each in four 100-step chunks, alternating between them.
    r1 = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=1)
    r2 = Replica(toy_ce, toy_atoms, temperature=500.0, random_seed=2)
    for _ in range(4):
        r1.advance(n_steps=100)
        r2.advance(n_steps=100)

    np.testing.assert_array_equal(r1.current_occupations(), ref1_occ)
    np.testing.assert_array_equal(r2.current_occupations(), ref2_occ)


def test_replica_construction_leaves_caller_random_state_untouched(toy_ce, toy_atoms):
    """Building a Replica must not observably mutate Python's global random.

    `mchammer.CanonicalEnsemble.__init__` calls `random.seed(random_seed)`
    internally. Replica catches and isolates that side effect so external
    code using `random` is unaffected.
    """
    import random

    random.seed(12345)
    reference = [random.random() for _ in range(5)]

    random.seed(12345)
    # Construct a Replica mid-stream — this must not perturb the sequence.
    Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=99)
    observed = [random.random() for _ in range(5)]

    assert observed == reference


def test_replica_uses_supplied_ensemble_class(toy_ce, toy_atoms):
    """Replica builds the ensemble from the supplied class, not CanonicalEnsemble."""
    from tests._ensemble_fixtures import TaggedCanonicalEnsemble

    rep = Replica(
        toy_ce,
        toy_atoms,
        temperature=300.0,
        random_seed=1,
        ensemble_cls=TaggedCanonicalEnsemble,
        ensemble_kwargs={"tag": "alpha"},
    )
    assert isinstance(rep._ensemble, TaggedCanonicalEnsemble)
    assert rep._ensemble.tag == "alpha"


@pytest.mark.parametrize(
    "reserved",
    ["structure", "calculator", "temperature", "random_seed"],
)
def test_replica_rejects_reserved_ensemble_kwargs(toy_ce, toy_atoms, reserved):
    """Replica must raise if ensemble_kwargs shadows a reserved name.

    The four kwargs (`structure`, `calculator`, `temperature`,
    `random_seed`) are computed and owned by Replica. Allowing the
    caller to pass them through `ensemble_kwargs` would either
    silently shadow Replica's value (wrong physics) or be silently
    overwritten (confusing). Up-front rejection keeps the contract
    explicit.
    """
    with pytest.raises(ValueError, match=reserved):
        Replica(
            toy_ce,
            toy_atoms,
            temperature=300.0,
            random_seed=1,
            ensemble_kwargs={reserved: "anything"},
        )


def test_replica_cluster_expansion_path_default_is_none(toy_ce, toy_atoms):
    """Path defaults to None when not supplied."""
    rep = Replica(toy_ce, toy_atoms, temperature=300.0, random_seed=0)
    assert rep.cluster_expansion_path is None


def test_replica_cluster_expansion_path_returns_supplied_value(
    toy_ce, toy_atoms, tmp_path
):
    """Supplied path round-trips through the property."""
    rep = Replica(
        toy_ce,
        toy_atoms,
        temperature=300.0,
        random_seed=0,
        cluster_expansion_path=str(tmp_path / "my.ce"),
    )
    assert rep.cluster_expansion_path == str(tmp_path / "my.ce")


def test_replica_cluster_expansion_path_not_validated_at_construction(
    toy_ce, toy_atoms
):
    """Construction does not require the path to exist or be readable.

    Validation is the caller's problem; Replica only stores the
    string for later access by factory-path observers.
    """
    rep = Replica(
        toy_ce,
        toy_atoms,
        temperature=300.0,
        random_seed=0,
        cluster_expansion_path="/absolutely/does/not/exist.ce",
    )
    assert rep.cluster_expansion_path == "/absolutely/does/not/exist.ce"
