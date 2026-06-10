"""Tests for per-halving merge diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mchammer_pt.wl_coordinator import (
    CoordinatorPlan,
    WalkerPostBlockState,
)
from tests._in_process_pool import make_in_process_wl_pool
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


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
    one_over_t_entry: str = "window_clock"
    applied_plans: list[CoordinatorPlan] = field(default_factory=list)
    # Read by SerialWangLandauPool.ensemble_cls_fqn.
    ensemble: object = field(default_factory=object)

    def advance(self, n_steps: int) -> None:
        pass

    def apply_plan(self, plan: CoordinatorPlan) -> None:
        self.applied_plans.append(plan)


def _flat_walker(
    *,
    step: int,
    entropy: dict[int, float],
    fill_factor: float = 1.0,
) -> WalkerPostBlockState:
    """Build a walker snapshot that the coordinator will treat as flat."""
    return WalkerPostBlockState(
        halving_criterion_met=True,
        fill_factor=fill_factor,
        entropy=dict(entropy),
        step=step,
        window_entry_step=0,
        histogram={k: 100 for k in entropy},
        reached_energy_window=True,
        current_energy=0.0,
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

    def test_events_per_slot_carry_correct_slot_index_and_content(
        self,
    ) -> None:
        from mchammer_pt.wl_coordinator import merge_entropies

        entropies_a = [{0: 1.0, 1: 2.0}, {0: 1.5, 1: 2.5}]
        entropies_b = [{0: 3.0, 1: 4.0}, {0: 3.5, 1: 4.5}]
        slot_a = _StubWLSlot(walker_states=[
            _flat_walker(step=100, entropy=entropies_a[0]),
            _flat_walker(step=100, entropy=entropies_a[1]),
        ])
        slot_b = _StubWLSlot(walker_states=[
            _flat_walker(step=100, entropy=entropies_b[0]),
            _flat_walker(step=100, entropy=entropies_b[1]),
        ])
        pool = self._make_pool([slot_a, slot_b])

        pool.advance_all(0)

        by_slot = {e.slot_index: e.merged_entropy for e in pool.merge_events}
        assert by_slot.keys() == {0, 1}
        assert by_slot[0] == merge_entropies(entropies_a)
        assert by_slot[1] == merge_entropies(entropies_b)

    def test_merge_events_property_returns_immutable_snapshot(self) -> None:
        walkers = [
            _flat_walker(step=100, entropy={0: 1.0, 1: 2.0}),
            _flat_walker(step=100, entropy={0: 1.5, 1: 2.5}),
        ]
        slot = _StubWLSlot(walker_states=walkers)
        pool = self._make_pool([slot])

        pool.advance_all(0)

        snapshot = pool.merge_events
        assert isinstance(snapshot, tuple)
        # The snapshot must not observe later mutations of the pool's
        # internal list.
        pool._merge_events.append(
            snapshot[0]  # any MergeEvent; identity doesn't matter
        )
        assert len(snapshot) == 1
        assert len(pool.merge_events) == 2


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

    def test_advance_all_records_merge_events_on_halving(self, tmp_path) -> None:
        """Drives the process pool's master-side RECORD block via an
        in-process W=2 setup. The RECORD block in
        ``ProcessWangLandauPool.advance_all`` is structurally identical
        to the serial pool's, but exercised through a different code
        path; this test catches drift between the two (wrong walker
        index for step, missing dict copy, wrong list to append to)."""
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
            windows=[(e0 - 5.0, e0 + 5.0)],
            seeds=[0],
            n_walkers_per_window=2,
        ) as pool:
            for _ in range(30):
                pool.advance_all(200)

            assert pool.merge_events, (
                "test setup no longer triggers a halving merge"
            )
            for event in pool.merge_events:
                assert event.slot_index == 0
                assert event.merged_entropy
                # merge_entropies min-shifts to icet convention.
                assert min(event.merged_entropy.values()) == 0.0
                # Step alignment with walker history (mirrors the
                # contract pinned in TestStepKeyAlignment for the
                # serial pool).
                for _, conn in pool._slots[0].workers:
                    replica = conn._worker._replica
                    assert event.step in replica._ensemble._fill_factor_history


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


class TestOrchestratorDelegation:
    def test_pt_merge_events_returns_pool_merge_events(self) -> None:
        """The orchestrator property forwards directly to the pool's
        attribute. Tested via ``__new__`` + a stand-in pool object to
        keep this a true unit test of the delegator; the end-to-end
        path is covered by the smoke test in Task 7."""
        from types import SimpleNamespace

        from mchammer_pt.wl import WangLandauParallelTempering
        from mchammer_pt.wl_merge_diagnostics import MergeEvent

        event = MergeEvent(slot_index=0, step=42, merged_entropy={0: 0.0, 1: 1.0})
        pt = WangLandauParallelTempering.__new__(WangLandauParallelTempering)
        pt._pool = SimpleNamespace(merge_events=(event,))  # type: ignore[assignment]

        assert pt.merge_events == (event,)
        assert pt.merge_events[0].step == 42


class TestStepKeyAlignment:
    def test_event_step_keys_walker_histories(self, tmp_path) -> None:
        """event.step must match a key in every walker's mchammer
        ``_fill_factor_history`` / ``_entropy_history``. This is the
        join contract the diagnostic relies on."""
        from mchammer.calculators import ClusterExpansionCalculator

        from mchammer_pt.wl import WangLandauParallelTempering

        ce = make_wl_ce()
        atoms = make_wl_atoms()
        e0 = float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )
        pt = WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[atoms, atoms],
            windows=[(e0 - 5.0, e0 + 5.0), (e0 - 5.0, e0 + 5.0)],
            energy_spacing=0.5,
            block_size=200,
            random_seed=0,
            n_walkers_per_window=2,
        )
        pt.run(n_cycles=30)

        assert pt.merge_events, (
            "test setup no longer triggers a halving merge; "
            "bump block_size/n_cycles"
        )

        for event in pt.merge_events:
            slot = pt._pool._replicas[event.slot_index]
            for walker in slot._replicas:
                e = walker._ensemble
                assert event.step in e._fill_factor_history
                assert event.step in e._entropy_history


class TestOrchestratorSmoke:
    def test_pt_merge_events_count_consistent_with_per_window_halvings(
        self, tmp_path
    ) -> None:
        """End-to-end: under the default ``merge_cadence="at_halve"`` on
        a multi-walker slot, the event count per slot equals the
        halvings reported by ``per_window_stats()`` for that slot."""
        from mchammer.calculators import ClusterExpansionCalculator

        from mchammer_pt.wl import WangLandauParallelTempering

        ce = make_wl_ce()
        atoms = make_wl_atoms()
        e0 = float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )
        pt = WangLandauParallelTempering(
            cluster_expansion=ce,
            atoms=[atoms, atoms],
            windows=[(e0 - 5.0, e0 + 5.0), (e0 - 5.0, e0 + 5.0)],
            energy_spacing=0.5,
            block_size=200,
            random_seed=0,
            n_walkers_per_window=2,
        )
        pt.run(n_cycles=30)

        assert pt.merge_events, (
            "test setup no longer triggers a halving merge; "
            "bump block_size/n_cycles"
        )

        stats = pt._pool.per_window_stats()
        for slot_index, slot_stats in enumerate(stats):
            events_for_slot = [
                e for e in pt.merge_events if e.slot_index == slot_index
            ]
            assert len(events_for_slot) == slot_stats["halvings"]


class TestPublicExport:
    def test_merge_event_is_exported(self) -> None:
        import mchammer_pt

        assert hasattr(mchammer_pt, "MergeEvent")
        assert "MergeEvent" in mchammer_pt.__all__
        # Same identity as the source module.
        from mchammer_pt.wl_merge_diagnostics import MergeEvent

        assert mchammer_pt.MergeEvent is MergeEvent
