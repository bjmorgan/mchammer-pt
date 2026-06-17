"""Tests for EnergyBinnedObservableRecorder.

Covers:
- moments-per-bin accumulation (count, sum, sum2, sum4)
- observer return-value coercion: bare scalar, dict, sequence
- non-finite observation drop and skipped tally
- columnar round-trip identity (to_state / from_state)
- signature validation on from_state
- accumulate-on-resume: from_state then record adds to existing data
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ase import Atoms
from mchammer.observers.base_observer import BaseObserver

from mchammer_pt.wl_observable_recorder import EnergyBinnedObservableRecorder

# ---------------------------------------------------------------------------
# Minimal test observers
# ---------------------------------------------------------------------------


class _SumObserver(BaseObserver):
    """Returns sum of atomic numbers as a single float."""

    def __init__(self, interval: int = 1, tag: str = "sumobs") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)

    def get_observable(self, structure: Any) -> float:
        return float(np.sum(structure.numbers))


class _DictObserver(BaseObserver):
    """Returns a dict with two scalar values."""

    def __init__(self, interval: int = 1, tag: str = "dictobs") -> None:
        super().__init__(interval=interval, return_type=dict, tag=tag)

    def get_observable(self, structure: Any) -> dict[str, float]:
        total = float(np.sum(structure.numbers))
        return {"b_val": total * 2, "a_val": total}


class _SeqObserver(BaseObserver):
    """Returns a two-element list of scalars."""

    def __init__(self, interval: int = 1, tag: str = "seqobs") -> None:
        super().__init__(interval=interval, return_type=list, tag=tag)

    def get_observable(self, structure: Any) -> list[float]:
        total = float(np.sum(structure.numbers))
        return [total, total * 3]


class _NanObserver(BaseObserver):
    """Returns NaN on first call, a valid float on subsequent calls."""

    def __init__(self, interval: int = 1, tag: str = "nanobs") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)
        self._calls = 0

    def get_observable(self, structure: Any) -> float:
        self._calls += 1
        if self._calls == 1:
            return float("nan")
        return 1.0


class _InfObserver(BaseObserver):
    """Returns +inf unconditionally."""

    def __init__(self, interval: int = 1, tag: str = "infobs") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)

    def get_observable(self, structure: Any) -> float:
        return float("inf")


class _ThreeValueObserver(BaseObserver):
    """Returns a three-element sequence — different size from _SumObserver."""

    def __init__(self, interval: int = 1, tag: str = "threeobs") -> None:
        super().__init__(interval=interval, return_type=list, tag=tag)

    def get_observable(self, structure: Any) -> list[float]:
        return [1.0, 2.0, 3.0]


class _NdarrayObserver(BaseObserver):
    """Returns a numpy array with two values."""

    def __init__(self, interval: int = 1, tag: str = "arrobs") -> None:
        super().__init__(interval=interval, return_type=list, tag=tag)

    def get_observable(self, structure: Any) -> np.ndarray:
        total = float(np.sum(structure.numbers))
        return np.array([total, total * 2.0])


class _AltDictObserver(BaseObserver):
    """Returns a dict with two scalar values under different keys to _DictObserver."""

    def __init__(self) -> None:
        super().__init__(interval=1, return_type=dict, tag="alt")

    def get_observable(self, structure: Any) -> dict[str, float]:
        return {"x_val": 1.0, "y_val": 2.0}


# ---------------------------------------------------------------------------
# Shared fixture: small ASE Atoms structure
# ---------------------------------------------------------------------------


@pytest.fixture
def atoms() -> Atoms:
    """A minimal ASE Atoms object with two hydrogen atoms (numbers=[1,1])."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# Moments-per-bin accumulation
# ---------------------------------------------------------------------------


def test_moments_single_bin_two_records(atoms: Atoms) -> None:
    """record twice into bin 3: count=2, sum=2*o, sum2=2*o^2, sum4=2*o^4."""
    obs = _SumObserver()
    rec = EnergyBinnedObservableRecorder(obs)

    o = obs.get_observable(atoms)  # deterministic

    rec.record(atoms, bin_index=3)
    rec.record(atoms, bin_index=3)

    state = rec.to_state()
    idx = state["bins"].index(3)

    assert state["count"][idx] == 2
    assert state["sum"][idx] == pytest.approx([2 * o])
    assert state["sum2"][idx] == pytest.approx([2 * o**2])
    assert state["sum4"][idx] == pytest.approx([2 * o**4])


def test_moments_second_bin_single_record(atoms: Atoms) -> None:
    """record once into bin 5: count=1."""
    obs = _SumObserver()
    rec = EnergyBinnedObservableRecorder(obs)

    rec.record(atoms, bin_index=3)
    rec.record(atoms, bin_index=3)
    rec.record(atoms, bin_index=5)

    state = rec.to_state()
    idx5 = state["bins"].index(5)
    assert state["count"][idx5] == 1


def test_bins_sorted_ascending_in_to_state(atoms: Atoms) -> None:
    """to_state returns bins in ascending order."""
    obs = _SumObserver()
    rec = EnergyBinnedObservableRecorder(obs)

    rec.record(atoms, bin_index=10)
    rec.record(atoms, bin_index=1)
    rec.record(atoms, bin_index=5)

    state = rec.to_state()
    assert state["bins"] == sorted(state["bins"])


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def test_coercion_bare_scalar_gives_s1(atoms: Atoms) -> None:
    """A bare float observer produces S=1, names=[tag]."""
    obs = _SumObserver(tag="myobs")
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    s, names = rec.signature
    assert s == 1
    assert names == ("myobs",)


def test_coercion_dict_orders_by_sorted_key(atoms: Atoms) -> None:
    """A dict observer: S=2, names sorted by key."""
    obs = _DictObserver(tag="dictobs")
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    s, names = rec.signature
    assert s == 2
    # Sorted keys: a_val < b_val
    assert names == ("a_val", "b_val")

    state = rec.to_state()
    o = float(np.sum(atoms.numbers))
    # a_val = o, b_val = 2*o; sorted order: a_val first
    assert state["sum"][0] == pytest.approx([o, 2 * o])


def test_coercion_sequence_gives_tag_indexed_names(atoms: Atoms) -> None:
    """A sequence observer with S>1: names default to tag_0, tag_1, ..."""
    obs = _SeqObserver(tag="seq")
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    s, names = rec.signature
    assert s == 2
    assert names == ("seq_0", "seq_1")


# ---------------------------------------------------------------------------
# Non-finite drop
# ---------------------------------------------------------------------------


def test_nan_observation_increments_skipped_not_count(atoms: Atoms) -> None:
    """NaN observation increments skipped tally, does not change count/sum."""
    obs = _NanObserver()
    rec = EnergyBinnedObservableRecorder(obs)

    # first call returns NaN
    rec.record(atoms, bin_index=7)
    # second call returns 1.0
    rec.record(atoms, bin_index=7)

    state = rec.to_state()
    idx = state["bins"].index(7)

    assert state["count"][idx] == 1, "only the finite observation is counted"
    assert state["sum"][idx] == pytest.approx([1.0])
    assert state["skipped"].get(7, 0) == 1


def test_inf_observation_increments_skipped(atoms: Atoms) -> None:
    """+inf observation increments skipped, does not touch count."""
    obs = _InfObserver()
    rec = EnergyBinnedObservableRecorder(obs)

    rec.record(atoms, bin_index=2)
    rec.record(atoms, bin_index=2)

    state = rec.to_state()
    assert 2 not in state["bins"], (
        "bin must not be created if all observations are non-finite"
    )
    assert state["skipped"].get(2, 0) == 2


# ---------------------------------------------------------------------------
# Columnar round-trip identity
# ---------------------------------------------------------------------------


def test_round_trip_identity(atoms: Atoms) -> None:
    """from_state(rec.to_state(), observer).to_state() == rec.to_state()."""
    obs = _SumObserver()
    rec = EnergyBinnedObservableRecorder(obs)

    rec.record(atoms, bin_index=0)
    rec.record(atoms, bin_index=0)
    rec.record(atoms, bin_index=3)

    state_a = rec.to_state()
    restored = EnergyBinnedObservableRecorder.from_state(state_a, _SumObserver())
    state_b = restored.to_state()

    assert state_a == state_b


# ---------------------------------------------------------------------------
# Signature validation on from_state
# ---------------------------------------------------------------------------


def test_from_state_mismatched_signature_raises(atoms: Atoms) -> None:
    """Restoring with a wrong-size observer raises ValueError on first record()."""
    obs1 = _SumObserver()  # S=1
    rec = EnergyBinnedObservableRecorder(obs1)
    rec.record(atoms, bin_index=0)

    state = rec.to_state()

    obs3 = _ThreeValueObserver()  # S=3
    restored = EnergyBinnedObservableRecorder.from_state(state, obs3)
    with pytest.raises(ValueError, match="signature"):
        restored.record(atoms, bin_index=0)


# ---------------------------------------------------------------------------
# Accumulate-on-resume
# ---------------------------------------------------------------------------


def test_accumulate_on_resume_adds_to_existing(atoms: Atoms) -> None:
    """from_state then record adds on top of restored data, does not reset."""
    obs = _SumObserver()
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    prior_state = rec.to_state()

    # Restore, then record again into the same bin
    restored = EnergyBinnedObservableRecorder.from_state(prior_state, _SumObserver())
    restored.record(atoms, bin_index=0)

    state = restored.to_state()
    idx = state["bins"].index(0)

    # count must be 2 (1 prior + 1 new), not reset to 1
    assert state["count"][idx] == 2

    o = obs.get_observable(atoms)
    assert state["sum"][idx] == pytest.approx([2 * o])


def test_accumulate_on_resume_new_bin(atoms: Atoms) -> None:
    """from_state then record into a NEW bin adds that bin without touching old ones."""
    obs = _SumObserver()
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    prior_state = rec.to_state()
    restored = EnergyBinnedObservableRecorder.from_state(prior_state, _SumObserver())
    restored.record(atoms, bin_index=99)

    state = restored.to_state()
    assert 0 in state["bins"]
    assert 99 in state["bins"]

    idx0 = state["bins"].index(0)
    assert state["count"][idx0] == 1  # prior data unchanged


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_tag_and_interval_properties() -> None:
    """tag and interval delegate to the underlying observer."""
    obs = _SumObserver(interval=5, tag="myobs")
    rec = EnergyBinnedObservableRecorder(obs)
    assert rec.tag == "myobs"
    assert rec.interval == 5


# ---------------------------------------------------------------------------
# numpy-array observer coercion
# ---------------------------------------------------------------------------


def test_coercion_ndarray_gives_s2_and_correct_moments(atoms: Atoms) -> None:
    """An observer returning np.ndarray: S=2, names tag_0/tag_1, correct moments."""
    obs = _NdarrayObserver(tag="arr")
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    s, names = rec.signature
    assert s == 2
    assert names == ("arr_0", "arr_1")

    o0 = float(np.sum(atoms.numbers))
    o1 = o0 * 2.0

    state = rec.to_state()
    idx = state["bins"].index(0)
    assert state["count"][idx] == 1
    assert state["sum"][idx] == pytest.approx([o0, o1])
    assert state["sum2"][idx] == pytest.approx([o0**2, o1**2])
    assert state["sum4"][idx] == pytest.approx([o0**4, o1**4])


# ---------------------------------------------------------------------------
# Signature validation — names mismatch (same size)
# ---------------------------------------------------------------------------


def test_from_state_names_mismatch_same_size_raises(atoms: Atoms) -> None:
    """Restoring with a same-size but different-named observer raises ValueError."""
    obs = _DictObserver()  # S=2, sorted keys: a_val, b_val
    rec = EnergyBinnedObservableRecorder(obs)
    rec.record(atoms, bin_index=0)

    state = rec.to_state()

    # _AltDictObserver has S=2 but keys x_val, y_val
    restored = EnergyBinnedObservableRecorder.from_state(state, _AltDictObserver())
    with pytest.raises(ValueError, match="signature"):
        restored.record(atoms, bin_index=0)


class _EmptyObserver(BaseObserver):
    """Returns an empty sequence -- a degenerate, no-scalar observable."""

    def __init__(self, tag: str = "empty") -> None:
        super().__init__(interval=1, return_type=list, tag=tag)

    def get_observable(self, structure: Any) -> list[float]:
        return []


def test_record_empty_observation_raises(atoms: Atoms) -> None:
    """An observer yielding no scalars is rejected, not silently dropped."""
    rec = EnergyBinnedObservableRecorder(_EmptyObserver())
    with pytest.raises(ValueError, match="no scalars"):
        rec.record(atoms, bin_index=0)


def test_from_state_corrupt_bins_count_mismatch_raises() -> None:
    """from_state rejects a state whose bins and count lengths disagree."""
    corrupt = {
        "tag": "sumobs",
        "names": ["sumobs"],
        "interval": 1,
        "bins": [0, 1],
        "count": [3],  # length 1, but two bins
        "sum": [[1.0], [2.0]],
        "sum2": [[1.0], [4.0]],
        "sum4": [[1.0], [16.0]],
        "skipped": {},
    }
    with pytest.raises(ValueError, match="corrupt"):
        EnergyBinnedObservableRecorder.from_state(corrupt, _SumObserver())


def test_from_state_empty_store_round_trips_and_coerces_skipped() -> None:
    """A never-recorded store (S=0) restores cleanly; skipped keys coerce to int."""
    empty_state = {
        "tag": "empty",
        "names": [],
        "interval": 1,
        "bins": [],
        "count": [],
        "sum": [],
        "sum2": [],
        "sum4": [],
        "skipped": {"7": 2},  # str key, as JSON round-trips integer keys
    }
    rec = EnergyBinnedObservableRecorder.from_state(empty_state, _EmptyObserver())
    state = rec.to_state()
    assert state["bins"] == []
    assert state["skipped"] == {7: 2}  # coerced to int, matching the populated branch
