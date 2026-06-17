"""Tests for observable moment store persistence in WangLandauReplica checkpoints.

Covers:
1. Round-trip: refresh/restore preserves recorder state; re-attach resumes.
2. Absent key (legacy checkpoint): restore does not raise; empty pending/recorders.
3. Accumulate-on-resume: post-restore advance adds to restored baseline.
4. Signature mismatch: restoring a store then attaching a mismatched observer raises.
5. New tag starts fresh: a new tag starts empty alongside a restorable old tag.
6. Unbound preservation: unbound restored stores are preserved across refresh.
7. Two-cycle unbound preservation: unbound stores survive two restore/refresh cycles.
8. Duplicate tag: attaching two observers with the same tag raises.
"""

from __future__ import annotations

from typing import Any

import pytest
from mchammer.observers.base_observer import BaseObserver

from mchammer_pt.wl_replica import WangLandauReplica
from tests._wl_fixtures import make_wl_atoms, make_wl_ce

# ---------------------------------------------------------------------------
# Minimal test observers
# ---------------------------------------------------------------------------


class _ScalarObserver(BaseObserver):
    """Returns a fixed scalar value."""

    def __init__(
        self, value: float = 1.0, interval: int = 1, tag: str = "scalar"
    ) -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)
        self._value = value

    def get_observable(self, structure: Any) -> float:
        return self._value


class _TwoScalarObserver(BaseObserver):
    """Returns a 2-element list of scalars."""

    def __init__(self, interval: int = 1, tag: str = "two_scalar") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)

    def get_observable(self, structure: Any) -> list[float]:
        return [1.0, 2.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frozen_replica(seed: int = 7) -> WangLandauReplica:
    """Build a frozen_g replica with a warmed-up DOS planted in."""
    from mchammer.calculators import ClusterExpansionCalculator


    ce = make_wl_ce()
    atoms = make_wl_atoms()
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers
    ))

    # Warm up a normal replica to get g(E).
    warm = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=seed,
    )
    warm.advance(500)
    g0 = dict(warm.ensemble._entropy)
    f0 = float(warm.ensemble._fill_factor)
    assert len(g0) > 1, "warm-up must populate more than one physical bin"

    # Build a frozen replica carrying that DOS.
    frozen = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=seed + 100,
        ensemble_kwargs={"frozen_g": True},
    )
    frozen.ensemble._entropy = g0
    frozen.ensemble._fill_factor = f0
    frozen.ensemble._reached_energy_window = True
    return frozen


def _make_plain_replica(seed: int = 0) -> WangLandauReplica:
    """Build a standard (non-frozen) WangLandauReplica on the toy CE."""
    from mchammer.calculators import ClusterExpansionCalculator

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    e0 = float(ClusterExpansionCalculator(atoms, ce).calculate_total(
        occupations=atoms.numbers
    ))
    return WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=seed,
    )


# ---------------------------------------------------------------------------
# Test 1: round-trip
# ---------------------------------------------------------------------------


def test_observable_records_round_trip_through_checkpoint():
    """refresh_last_state writes observable_records; restore + re-attach resumes.

    Steps:
      1. Build a frozen replica, attach observer, advance to populate store.
      2. refresh_last_state; assert observable_records present and non-empty.
      3. restore_state into fresh frozen replica; no observer attached yet,
         recorders empty, pending store held.
      4. record_observable(same observer); assert recorder state equals saved.
    """
    observer = _ScalarObserver(tag="myobs", interval=1)

    replica = _make_frozen_replica(seed=7)
    replica.record_observable(observer)
    replica.advance(200)

    replica.refresh_last_state()
    saved = replica.ensemble._data_container._last_state.get("observable_records")
    assert saved is not None
    assert "myobs" in saved
    assert saved["myobs"]["bins"]  # non-empty — some observations recorded

    # Restore into a fresh frozen replica.
    fresh = _make_frozen_replica(seed=7)
    fresh.restore_state(replica.data_container())

    # No observer attached yet: recorders empty, pending store held.
    assert fresh.ensemble._recorders == {}
    assert "myobs" in fresh.ensemble._restored_observable_records

    # Re-attach the same observer; recorder state should match what was saved.
    fresh.record_observable(observer)
    assert "myobs" in fresh.ensemble._recorders
    restored_state = fresh.ensemble._recorders["myobs"].to_state()

    assert restored_state["bins"] == saved["myobs"]["bins"]
    assert restored_state["count"] == saved["myobs"]["count"]


# ---------------------------------------------------------------------------
# Test 2: absent key (legacy checkpoint)
# ---------------------------------------------------------------------------


def test_restore_state_absent_observable_records_is_harmless():
    """A checkpoint without observable_records does not raise; stores empty."""
    replica = _make_plain_replica(seed=0)
    replica.advance(10)
    replica.refresh_last_state()

    container = replica.data_container()
    # Simulate legacy checkpoint by removing the key.
    container._last_state.pop("observable_records", None)

    fresh = _make_plain_replica(seed=0)
    fresh.restore_state(container)  # must not raise

    assert fresh.ensemble._restored_observable_records == {}
    assert fresh.ensemble._recorders == {}


# ---------------------------------------------------------------------------
# Test 3: accumulate-on-resume
# ---------------------------------------------------------------------------


def test_observable_accumulates_after_resume():
    """Post-restore advance adds to the restored baseline (no reset).

    Saves the total observation count from a first run, then resumes and
    advances the same number of steps again. The total count across all
    bins must be strictly greater than the saved total (new observations
    piled on top of the restored baseline).
    """
    observer = _ScalarObserver(tag="acc", interval=1)

    replica = _make_frozen_replica(seed=10)
    replica.record_observable(observer)
    replica.advance(300)

    replica.refresh_last_state()
    saved = replica.ensemble._data_container._last_state["observable_records"]["acc"]
    saved_total = sum(saved["count"])
    assert saved_total > 0, "first run must record at least one observation"

    fresh = _make_frozen_replica(seed=10)
    fresh.restore_state(replica.data_container())
    fresh.record_observable(_ScalarObserver(tag="acc", interval=1))
    fresh.advance(300)

    resumed_state = fresh.ensemble._recorders["acc"].to_state()
    resumed_total = sum(resumed_state["count"])

    assert resumed_total > saved_total, (
        f"resumed total {resumed_total} not > saved total {saved_total}; "
        "accumulation did not build on the restored baseline"
    )

    # Also verify that the saved bins' counts are present in the resumed
    # recorder at or above their saved values (restored baseline was not reset).
    saved_counts = dict(zip(saved["bins"], saved["count"], strict=True))
    resumed_counts = dict(
        zip(resumed_state["bins"], resumed_state["count"], strict=True)
    )
    for b, cnt in saved_counts.items():
        assert b in resumed_counts, f"saved bin {b} missing from resumed state"
        assert resumed_counts[b] >= cnt, (
            f"bin {b}: resumed count {resumed_counts[b]} < saved {cnt}; "
            "the restored baseline must be included in the resumed counts"
        )


# ---------------------------------------------------------------------------
# Test 4: signature mismatch raises
# ---------------------------------------------------------------------------


def test_observable_signature_mismatch_raises_on_reattach():
    """Restoring a 1-scalar store then attaching a 2-scalar observer raises.

    restore_state with a non-empty stored state sets _names/_size from the
    store but does NOT validate the live observer's signature until the first
    record() call (driven by advance), where it raises ValueError.
    """
    one_scalar_obs = _ScalarObserver(tag="sig", interval=1)

    replica = _make_frozen_replica(seed=20)
    replica.record_observable(one_scalar_obs)
    replica.advance(200)
    replica.refresh_last_state()

    fresh = _make_frozen_replica(seed=20)
    fresh.restore_state(replica.data_container())

    # Re-attach a 2-scalar observer under the same tag; mismatch is deferred
    # to the first record() call triggered by advance.
    two_scalar_obs = _TwoScalarObserver(tag="sig", interval=1)
    with pytest.raises(ValueError, match="observer signature mismatch"):
        fresh.record_observable(two_scalar_obs)
        fresh.advance(10)


# ---------------------------------------------------------------------------
# Test 5: new tag starts fresh
# ---------------------------------------------------------------------------


def test_new_tag_after_restore_starts_empty():
    """A new tag attached after restore starts with an empty recorder.

    The stored tag ("old") is in the pending store; a new tag ("new")
    gets a fresh recorder starting from zero counts.
    """
    observer_old = _ScalarObserver(tag="old", interval=1)

    replica = _make_frozen_replica(seed=30)
    replica.record_observable(observer_old)
    replica.advance(200)
    replica.refresh_last_state()

    fresh = _make_frozen_replica(seed=30)
    fresh.restore_state(replica.data_container())

    # Attach only the new tag.
    fresh.record_observable(_ScalarObserver(tag="new", interval=1))

    assert "old" in fresh.ensemble._restored_observable_records
    assert "new" in fresh.ensemble._recorders
    new_state = fresh.ensemble._recorders["new"].to_state()
    assert new_state["bins"] == []
    assert new_state["count"] == []


# ---------------------------------------------------------------------------
# Test 6: unbound preservation across refresh
# ---------------------------------------------------------------------------


def test_unbound_restored_store_preserved_through_refresh():
    """Unbound (never re-attached) restored stores survive a subsequent refresh.

    Restores a checkpoint with a stored tag, does NOT re-attach any
    observer, then calls refresh_last_state() again. The
    observable_records in the written _last_state must still contain
    the unbound tag's data.
    """
    observer = _ScalarObserver(tag="unbound", interval=1)

    replica = _make_frozen_replica(seed=40)
    replica.record_observable(observer)
    replica.advance(200)
    replica.refresh_last_state()

    saved_bins = (
        replica.ensemble._data_container._last_state["observable_records"]["unbound"]["bins"]
    )
    assert saved_bins, "first run must populate bins"

    fresh = _make_frozen_replica(seed=40)
    fresh.restore_state(replica.data_container())
    # Do NOT re-attach. Just refresh.
    fresh.refresh_last_state()

    second_save = fresh.ensemble._data_container._last_state.get("observable_records")
    assert second_save is not None
    assert "unbound" in second_save
    assert second_save["unbound"]["bins"] == saved_bins


# ---------------------------------------------------------------------------
# Test 7: two-cycle unbound preservation
# ---------------------------------------------------------------------------


def test_unbound_store_preserved_across_two_checkpoint_cycles():
    """Unbound stores survive two sequential restore→refresh cycles.

    Pins the idempotency of the restore→refresh union loop against
    future refactors: the observable_records written by refresh_last_state
    must survive even when no observer is re-attached in either cycle.

    Steps:
      1. Build a frozen replica, attach observer, advance, refresh. Capture
         the saved bins/count from observable_records.
      2. Cycle 1: restore into fresh1, attach nothing, refresh. Assert the
         unbound store is still in _last_state["observable_records"].
      3. Cycle 2: take fresh1's container, restore into fresh2, attach
         nothing, refresh. Assert the unbound store is STILL present and
         has the same bins and counts as the original.
    """
    observer = _ScalarObserver(tag="passthru", interval=1)

    replica = _make_frozen_replica(seed=50)
    replica.record_observable(observer)
    replica.advance(200)
    replica.refresh_last_state()

    obs_records = replica.ensemble._data_container._last_state["observable_records"]
    orig = obs_records["passthru"]
    assert orig["bins"], "first run must populate bins"
    orig_bins = list(orig["bins"])
    orig_count = list(orig["count"])

    # --- Cycle 1 ---
    fresh1 = _make_frozen_replica(seed=50)
    fresh1.restore_state(replica.data_container())
    # No observer attached.
    fresh1.refresh_last_state()

    cycle1 = fresh1.ensemble._data_container._last_state.get("observable_records")
    assert cycle1 is not None, "observable_records absent after cycle 1 refresh"
    assert "passthru" in cycle1, "unbound tag lost after cycle 1 refresh"

    # --- Cycle 2 ---
    fresh2 = _make_frozen_replica(seed=50)
    fresh2.restore_state(fresh1.data_container())
    # No observer attached.
    fresh2.refresh_last_state()

    cycle2 = fresh2.ensemble._data_container._last_state.get("observable_records")
    assert cycle2 is not None, "observable_records absent after cycle 2 refresh"
    assert "passthru" in cycle2, "unbound tag lost after cycle 2 refresh"
    assert cycle2["passthru"]["bins"] == orig_bins, (
        "bins changed across two cycles"
    )
    assert cycle2["passthru"]["count"] == orig_count, (
        "counts changed across two cycles"
    )


# ---------------------------------------------------------------------------
# Test 8: duplicate tag raises
# ---------------------------------------------------------------------------


def test_record_observable_duplicate_tag_raises():
    """Attaching two observers with the same tag raises ValueError."""
    replica = _make_plain_replica(seed=0)
    replica.record_observable(_ScalarObserver(tag="dup", interval=1))
    with pytest.raises(ValueError, match="dup"):
        replica.record_observable(_ScalarObserver(tag="dup", interval=1))
