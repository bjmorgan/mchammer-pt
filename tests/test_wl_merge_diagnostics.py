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


from dataclasses import dataclass, field

from mchammer_pt.wl_coordinator import (
    CoordinatorPlan,
    WalkerPostBlockState,
)


@dataclass
class _StubWLSlot:
    """Minimal slot satisfying SerialWangLandauPool's structural needs.

    Stores pre-canned walker_states, records applied plans for
    inspection, and ignores advance(). Used to drive advance_all
    deterministically without running real MC.
    """

    walker_states: list[WalkerPostBlockState]
    energy_spacing: float = 1.0
    energy_window: tuple[float | None, float | None] = (0.0, 10.0)
    phase: str = "halving"
    schedule: str = "halving"
    flatness_limit: float = 0.8
    applied_plans: list[CoordinatorPlan] = field(default_factory=list)

    def advance(self, n_steps: int) -> None:
        pass

    def apply_plan(self, plan: CoordinatorPlan) -> None:
        self.applied_plans.append(plan)

    def reroll_exchange_idx(self) -> None:
        pass


def _flat_walker(
    *,
    step: int,
    entropy: dict[int, float],
    fill_factor: float = 1.0,
) -> WalkerPostBlockState:
    """Build a walker snapshot that the coordinator will treat as flat."""
    return WalkerPostBlockState(
        is_flat=True,
        fill_factor=fill_factor,
        entropy=dict(entropy),
        step=step,
        window_entry_step=0,
        histogram={k: 100 for k in entropy},
        reached_energy_window=True,
    )


class TestSerialPoolMergeEvents:
    def _make_pool(self, slots, **kwargs):
        from mchammer_pt.parallel.serial import SerialWangLandauPool

        return SerialWangLandauPool(
            slots, energy_spacing=1.0, **kwargs
        )

    def test_records_one_event_per_halving_merge(self) -> None:
        walkers = [
            _flat_walker(step=500, entropy={0: 1.0, 1: 2.0}),
            _flat_walker(step=500, entropy={0: 1.5, 1: 2.5}),
        ]
        slot = _StubWLSlot(walker_states=walkers)
        pool = self._make_pool([slot])

        pool.advance_all(0)

        assert len(pool.merge_events) == 1
        event = pool.merge_events[0]
        assert event.slot_index == 0
        assert event.step == 500
        assert min(event.merged_entropy.values()) == 0.0

    def test_event_merged_entropy_matches_merge_entropies(self) -> None:
        from mchammer_pt.wl_coordinator import merge_entropies

        entropies = [{0: 1.0, 1: 2.0}, {0: 1.5, 1: 2.5}]
        walkers = [
            _flat_walker(step=200, entropy=entropies[0]),
            _flat_walker(step=200, entropy=entropies[1]),
        ]
        slot = _StubWLSlot(walker_states=walkers)
        pool = self._make_pool([slot])

        pool.advance_all(0)

        assert pool.merge_events[0].merged_entropy == merge_entropies(entropies)

    def test_no_event_when_merge_cadence_never(self) -> None:
        walkers = [
            _flat_walker(step=300, entropy={0: 1.0, 1: 2.0}),
            _flat_walker(step=300, entropy={0: 1.5, 1: 2.5}),
        ]
        slot = _StubWLSlot(walker_states=walkers)
        pool = self._make_pool([slot], merge_cadence="never")

        pool.advance_all(0)

        assert pool.merge_events == ()

    def test_no_event_for_single_walker_slot(self) -> None:
        walkers = [_flat_walker(step=400, entropy={0: 1.0, 1: 2.0})]
        slot = _StubWLSlot(walker_states=walkers)
        pool = self._make_pool([slot])

        pool.advance_all(0)

        assert pool.merge_events == ()

    def test_events_per_slot_use_correct_slot_index(self) -> None:
        slot_a = _StubWLSlot(walker_states=[
            _flat_walker(step=100, entropy={0: 1.0, 1: 2.0}),
            _flat_walker(step=100, entropy={0: 1.5, 1: 2.5}),
        ])
        slot_b = _StubWLSlot(walker_states=[
            _flat_walker(step=100, entropy={0: 3.0, 1: 4.0}),
            _flat_walker(step=100, entropy={0: 3.5, 1: 4.5}),
        ])
        pool = self._make_pool([slot_a, slot_b])

        pool.advance_all(0)

        assert sorted(e.slot_index for e in pool.merge_events) == [0, 1]

    def test_merge_events_returns_tuple_copy(self) -> None:
        walkers = [
            _flat_walker(step=100, entropy={0: 1.0, 1: 2.0}),
            _flat_walker(step=100, entropy={0: 1.5, 1: 2.5}),
        ]
        slot = _StubWLSlot(walker_states=walkers)
        pool = self._make_pool([slot])

        pool.advance_all(0)

        events = pool.merge_events
        assert isinstance(events, tuple)
        # Mutating the returned tuple's underlying list (if it were one)
        # must not affect the pool's record.
        assert len(pool.merge_events) == 1


from tests._in_process_pool import make_in_process_wl_pool
from tests._wl_fixtures import make_wl_ce, make_wl_atoms


class TestProcessPoolMergeEventsSurface:
    def test_merge_events_starts_empty_on_a_fresh_pool(self, tmp_path) -> None:
        from mchammer.calculators import ClusterExpansionCalculator

        ce = make_wl_ce()
        atoms = make_wl_atoms()
        e0 = float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )
        with make_in_process_wl_pool(
            tmp_path,
            windows=[(e0 - 50.0, e0 + 50.0)],
            seeds=[0],
            n_walkers_per_window=2,
        ) as pool:
            assert pool.merge_events == ()


class TestProtocolSurface:
    def test_serial_pool_satisfies_runtime_protocol(self) -> None:
        from mchammer_pt.parallel.backend import WangLandauPool
        from mchammer_pt.parallel.serial import SerialWangLandauPool

        # SerialWangLandauPool is the concrete type the orchestrator uses;
        # it must satisfy the runtime-checkable protocol after the new
        # member is added.
        walkers = [
            _flat_walker(step=0, entropy={0: 0.0, 1: 0.0}),
            _flat_walker(step=0, entropy={0: 0.0, 1: 0.0}),
        ]
        slot = _StubWLSlot(walker_states=walkers)
        pool = SerialWangLandauPool([slot], energy_spacing=1.0)
        assert isinstance(pool, WangLandauPool)
        # The property is the new contract point.
        assert hasattr(pool, "merge_events")
