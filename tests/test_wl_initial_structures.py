"""Unit tests for expand_initial_structures."""

from __future__ import annotations

import pytest
from ase import Atoms

from mchammer_pt.wl_initial_structures import expand_initial_structures


def _atoms(symbol: str = "H") -> Atoms:
    return Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True)


def test_broadcast_single_atoms_repeats_to_walker_count():
    a = _atoms()
    out = expand_initial_structures([a], [3])
    assert len(out) == 1
    assert len(out[0]) == 3
    # Broadcast repeats the same reference (no copy in the helper).
    assert out[0][0] is a and out[0][1] is a and out[0][2] is a


def test_per_walker_sequence_returned_in_order():
    a, b = _atoms("H"), _atoms("He")
    out = expand_initial_structures([[a, b]], [2])
    assert out[0][0] is a
    assert out[0][1] is b


def test_mixed_windows_broadcast_and_sequence():
    a, b, c = _atoms("H"), _atoms("He"), _atoms("Li")
    out = expand_initial_structures([a, [b, c]], [2, 2])
    assert out[0] == [a, a]
    assert out[1] == [b, c]


def test_sequence_length_mismatch_raises_naming_window():
    a = _atoms()
    with pytest.raises(ValueError, match=r"window 1 has 3 walkers"):
        expand_initial_structures([a, [a, a]], [1, 3])


def test_empty_sequence_raises():
    a = _atoms()
    with pytest.raises(ValueError, match=r"atoms\[1\] is an empty sequence"):
        expand_initial_structures([a, []], [1, 2])


def test_non_atoms_element_raises():
    a = _atoms()
    with pytest.raises(ValueError, match=r"not an Atoms"):
        expand_initial_structures([[a, "nope"]], [2])
