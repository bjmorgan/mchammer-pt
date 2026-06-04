"""Tests for bottom/top anchor assignment."""

from __future__ import annotations

import pytest

from mchammer_pt.seeding.anchoring import (
    assign_anchors,
    validate_anchor_override,
)


def test_finite_windows_split_on_midpoint():
    # e_gs=0, e_top=10 -> midpoint 5. Window centre below 5 -> bottom.
    windows = [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]
    assert assign_anchors(windows, e_gs=0.0, e_top=10.0) == [
        "bottom",
        "top",
        "top",
    ]


def test_centre_exactly_at_midpoint_is_top():
    # centre (4+6)/2 = 5 == midpoint 5 -> "top" (strictly-below -> bottom).
    assert assign_anchors([(4.0, 6.0)], e_gs=0.0, e_top=10.0) == ["top"]


def test_unbounded_left_edge_forces_bottom():
    assert assign_anchors([(None, 1.0)], e_gs=0.0, e_top=10.0) == ["bottom"]


def test_unbounded_right_edge_forces_top():
    assert assign_anchors([(9.0, None)], e_gs=0.0, e_top=10.0) == ["top"]


def test_override_validates_length():
    with pytest.raises(ValueError, match="length"):
        validate_anchor_override(["bottom"], n_windows=2)


def test_override_validates_values():
    with pytest.raises(ValueError, match="bottom.*top|top.*bottom|invalid"):
        validate_anchor_override(["bottom", "middle"], n_windows=2)


def test_override_passes_through_valid():
    assert validate_anchor_override(["bottom", "top"], n_windows=2) == [
        "bottom",
        "top",
    ]
