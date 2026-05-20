"""Tests for per-halving merge diagnostics."""

from __future__ import annotations

import pytest


class TestMergeEventRecord:
    def test_construct_with_required_fields(self) -> None:
        from mchammer_pt.wl_merge_diagnostics import MergeEvent

        event = MergeEvent(
            slot_index=2,
            step=1500,
            merged_entropy={0: 0.0, 1: 1.5, 2: 3.0},
        )
        assert event.slot_index == 2
        assert event.step == 1500
        assert event.merged_entropy == {0: 0.0, 1: 1.5, 2: 3.0}

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from mchammer_pt.wl_merge_diagnostics import MergeEvent

        event = MergeEvent(slot_index=0, step=0, merged_entropy={})
        with pytest.raises(FrozenInstanceError):
            event.slot_index = 99  # type: ignore[misc]
