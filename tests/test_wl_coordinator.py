"""Direct unit tests for the WL coordinator policy.

Tests `decide_block_actions` against synthetic SlotView instances.
No replica plumbing, no IPC — the policy lives in one pure function
over a pure data structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from mchammer_pt.wl_coordinator import (
    CoordinatorPlan,
    SlotView,
    WalkerPostBlockState,
    _compute_filled_bins,
    _compute_per_walker_breakdown,
    _resolve_bins_filled,
    _summed_histogram_halving_criterion_met,
    _validate_bp_stall_multiple,
    _validate_one_over_t_gate,
    decide_block_actions,
)


def _state(
    *,
    halving_criterion_met: bool = True,
    fill_factor: float = 1.0,
    entropy: dict[int, float] | None = None,
    step: int = 1000,
    window_entry_step: int | None = 0,
    histogram: dict[int, int] | None = None,
    reached: bool = True,
    current_energy: float = 0.0,
) -> WalkerPostBlockState:
    return WalkerPostBlockState(
        halving_criterion_met=halving_criterion_met,
        fill_factor=fill_factor,
        entropy=dict(entropy) if entropy is not None else {0: 0.0, 1: 0.0},
        step=step,
        window_entry_step=window_entry_step,
        histogram=dict(histogram) if histogram is not None else {0: 100, 1: 100},
        reached_energy_window=reached,
        current_energy=current_energy,
    )


def _view(
    states: list[WalkerPostBlockState],
    *,
    phase: str = "halving",
    flatness_mode: str = "pooled",
    merge_cadence: str = "at_halve",
    schedule: str = "halving",
    flatness_limit: float = 0.8,
) -> SlotView:
    return SlotView(
        walker_states=tuple(states),
        phase=phase,
        flatness_mode=flatness_mode,  # type: ignore[arg-type]
        merge_cadence=merge_cadence,  # type: ignore[arg-type]
        schedule=schedule,
        flatness_limit=flatness_limit,
    )


class TestPhaseGate:
    def test_one_over_t_phase_returns_empty_plan(self) -> None:
        view = _view([_state(halving_criterion_met=True)], phase="1_over_t")
        plan = decide_block_actions(view)
        assert plan == CoordinatorPlan(
            halve=False, merged_entropy=None, switch_to_phase=None
        )


class TestHalveGatePerWalker:
    def test_all_flat_triggers_halve(self) -> None:
        view = _view(
            [_state(halving_criterion_met=True), _state(halving_criterion_met=True)],
            flatness_mode="per_walker",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True

    def test_one_not_flat_blocks_halve(self) -> None:
        view = _view(
            [_state(halving_criterion_met=True), _state(halving_criterion_met=False)],
            flatness_mode="per_walker",
        )
        plan = decide_block_actions(view)
        assert plan.halve is False


class TestHalveGatePooled:
    def test_summed_histogram_flat_triggers_halve(self) -> None:
        view = _view(
            [
                _state(histogram={0: 50, 1: 50}, halving_criterion_met=False),
                _state(histogram={0: 50, 1: 50}, halving_criterion_met=False),
            ],
            flatness_mode="pooled",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True

    def test_summed_histogram_not_flat_blocks_halve(self) -> None:
        view = _view(
            [
                _state(histogram={0: 10, 1: 100}, halving_criterion_met=False),
                _state(histogram={0: 10, 1: 100}, halving_criterion_met=False),
            ],
            flatness_mode="pooled",
        )
        plan = decide_block_actions(view)
        assert plan.halve is False

    def test_pooled_returns_false_when_any_walker_unentered(self) -> None:
        view = _view(
            [
                _state(reached=True),
                _state(reached=False),
            ],
            flatness_mode="pooled",
        )
        plan = decide_block_actions(view)
        assert plan.halve is False


class TestMergeCadence:
    def test_at_halve_with_w_gt_one_produces_merged_entropy(self) -> None:
        view = _view(
            [
                _state(entropy={0: 1.0, 1: 2.0}, halving_criterion_met=True),
                _state(entropy={0: 1.5, 1: 2.5}, halving_criterion_met=True),
            ],
            flatness_mode="per_walker",
            merge_cadence="at_halve",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True
        assert plan.merged_entropy is not None
        # Merged entropy is min-shifted so the lowest value is 0.
        assert min(plan.merged_entropy.values()) == 0

    def test_never_cadence_skips_merge(self) -> None:
        view = _view(
            [
                _state(entropy={0: 1.0, 1: 2.0}, halving_criterion_met=True),
                _state(entropy={0: 1.5, 1: 2.5}, halving_criterion_met=True),
            ],
            flatness_mode="per_walker",
            merge_cadence="never",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True
        assert plan.merged_entropy is None

    def test_w_eq_one_skips_merge_regardless_of_cadence(self) -> None:
        view = _view(
            [_state(entropy={0: 1.0, 1: 2.0}, halving_criterion_met=True)],
            flatness_mode="per_walker",
            merge_cadence="at_halve",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True
        assert plan.merged_entropy is None


class TestBPSwitch:
    def test_fires_when_all_walkers_have_inverse_t_above_post_halve_f(
        self,
    ) -> None:
        # post-halve fill factor = 0.5; need 1/t > 0.5, i.e. t < 2.
        # step - window_entry_step + 1 = 1 (step 0, entry 0) gives t=1.
        view = _view(
            [_state(fill_factor=1.0, step=0, window_entry_step=0)],
            flatness_mode="per_walker",
            schedule="1_over_t",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True
        assert plan.switch_to_phase == "1_over_t"

    def test_does_not_fire_when_any_walker_unentered(self) -> None:
        # Both walkers are flat so the halve gate passes; the unentered
        # guard inside the BP-switch branch must block the switch.
        view = _view(
            [
                _state(
                    fill_factor=1.0,
                    step=0,
                    window_entry_step=0,
                    halving_criterion_met=True,
                ),
                _state(
                    fill_factor=1.0,
                    step=0,
                    window_entry_step=None,
                    halving_criterion_met=True,
                ),
            ],
            flatness_mode="per_walker",
            schedule="1_over_t",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True              # gate passed
        assert plan.switch_to_phase is None    # unentered guard fired

    def test_does_not_fire_when_inverse_t_below_post_halve_f(self) -> None:
        # post-halve f = 0.5; t = 1001 gives 1/t ~ 1e-3 < 0.5; no switch.
        view = _view(
            [_state(fill_factor=1.0, step=1000, window_entry_step=0)],
            flatness_mode="per_walker",
            schedule="1_over_t",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True
        assert plan.switch_to_phase is None

    def test_halving_schedule_never_switches(self) -> None:
        view = _view(
            [_state(fill_factor=1.0, step=0, window_entry_step=0)],
            flatness_mode="per_walker",
            schedule="halving",
        )
        plan = decide_block_actions(view)
        assert plan.halve is True
        assert plan.switch_to_phase is None


class TestSummedHistogramFlatZeroCount:
    """Pin the production pooled-gate's zero-count semantics.

    The pooled flatness gate treats a present-but-unvisited bin
    (count 0) as not-flat. Tests call the production function
    directly so a refactor of the gate cannot silently change this
    contract.
    """

    def test_zero_count_bin_blocks_flatness(self) -> None:
        snapshot = _state(histogram={0: 0, 1: 1000})
        assert not _summed_histogram_halving_criterion_met(
            [snapshot], flatness_limit=0.7, schedule="halving"
        )

    def test_no_zero_entry_is_flat(self) -> None:
        snapshot = _state(histogram={1: 1000})
        assert _summed_histogram_halving_criterion_met(
            [snapshot], flatness_limit=0.7, schedule="halving"
        )


class TestSummedHistogramFlatAllZero:
    """An all-zero combined histogram does not pass the flatness gate.

    Otherwise ``mean(counts) = 0`` would make ``limit = 0`` and
    ``all(counts >= 0)`` would be vacuously true.
    """

    def test_single_zero_bin_does_not_flatten(self) -> None:
        snapshot = _state(histogram={0: 0})
        assert not _summed_histogram_halving_criterion_met(
            [snapshot], flatness_limit=0.8, schedule="halving"
        )

    def test_all_zero_histogram_with_multiple_bins_does_not_flatten(
        self,
    ) -> None:
        snapshot = _state(histogram={0: 0, 1: 0, 2: 0})
        assert not _summed_histogram_halving_criterion_met(
            [snapshot], flatness_limit=0.8, schedule="halving"
        )


class TestWOneCollapsesModes:
    def test_pooled_and_per_walker_give_identical_plan_for_w1(self) -> None:
        state = _state(histogram={0: 100, 1: 100}, halving_criterion_met=True)
        v_pooled = _view([state], flatness_mode="pooled")
        v_per = _view([state], flatness_mode="per_walker")
        assert decide_block_actions(v_pooled) == decide_block_actions(v_per)


class TestScheduleAwarePooledGate:
    """The pooled halving gate honours the schedule: BP min(H) > 0
    under '1_over_t', WL flatness under 'halving'."""

    def test_bp_criterion_under_one_over_t(self) -> None:
        snapshots = [
            _state(histogram={0: 1, 1: 100}, halving_criterion_met=False),
        ]
        result = _summed_histogram_halving_criterion_met(
            snapshots, flatness_limit=0.8, schedule="1_over_t"
        )
        assert result is True

    def test_wl_criterion_under_halving(self) -> None:
        snapshots = [
            _state(histogram={0: 1, 1: 100}, halving_criterion_met=False),
        ]
        result = _summed_histogram_halving_criterion_met(
            snapshots, flatness_limit=0.8, schedule="halving"
        )
        assert result is False

    def test_decide_halves_under_one_over_t_low_flatness(self) -> None:
        """A pooled histogram satisfying BP (min > 0) but failing WL
        flatness still produces a halve plan. Demonstrates the bug fix."""
        states = [
            _state(
                histogram={0: 1, 1: 100},
                fill_factor=1.0,
                step=1000,
                window_entry_step=0,
                halving_criterion_met=False,
            ),
        ]
        view = _view(
            states,
            flatness_mode="pooled",
            schedule="1_over_t",
            flatness_limit=0.8,
        )
        plan = decide_block_actions(view)
        assert plan.halve is True

    def test_decide_does_not_halve_under_halving_low_flatness(self) -> None:
        """A pooled histogram with min/mean below flatness_limit must
        NOT halve under 'halving'. Regression guard."""
        states = [
            _state(
                histogram={0: 1, 1: 100},
                fill_factor=1.0,
                step=1000,
                window_entry_step=0,
                halving_criterion_met=False,
            ),
        ]
        view = _view(
            states,
            flatness_mode="pooled",
            schedule="halving",
            flatness_limit=0.8,
        )
        plan = decide_block_actions(view)
        assert plan.halve is False


def test_compute_filled_bins_single_histogram_counts_positives():
    h = {0: 3, 1: 0, 2: 5}
    assert _compute_filled_bins([h], "pooled") == 2
    assert _compute_filled_bins([h], "per_walker") == 2


def test_compute_filled_bins_pooled_is_union_of_positives():
    a = {0: 1, 1: 0, 2: 0}
    b = {0: 0, 1: 2, 2: 0}
    # bins 0 and 1 positive in at least one walker -> 2
    assert _compute_filled_bins([a, b], "pooled") == 2


def test_compute_filled_bins_per_walker_is_intersection_of_positives():
    a = {0: 1, 1: 4, 2: 0}
    b = {0: 3, 1: 0, 2: 7}
    # only bin 0 positive in every walker -> 1
    assert _compute_filled_bins([a, b], "per_walker") == 1


def test_compute_filled_bins_per_walker_disjoint_keys_is_conservative():
    # walker b never seeded bin 2; intersection excludes it even though
    # every walker's own histogram is fully positive.
    a = {0: 1, 1: 1, 2: 1}
    b = {0: 1, 1: 1}
    assert _compute_filled_bins([a, b], "per_walker") == 2


def test_compute_filled_bins_empty_inputs_return_zero():
    assert _compute_filled_bins([], "pooled") == 0
    assert _compute_filled_bins([{}], "per_walker") == 0


def test_compute_per_walker_breakdown_reports_triples():
    a = {0: 2, 1: 0}        # filled 1, known 2, flat_min 0/1 = 0.0
    b = {0: 4, 1: 4}        # filled 2, known 2, flat_min 1.0
    out = _compute_per_walker_breakdown([a, b])
    assert out == [
        {"filled": 1, "known": 2, "flat_min": 0.0},
        {"filled": 2, "known": 2, "flat_min": 1.0},
    ]


def test_compute_per_walker_breakdown_empty_histogram_flat_min_none():
    out = _compute_per_walker_breakdown([{}])
    assert out == [{"filled": 0, "known": 0, "flat_min": None}]


def test_resolve_bins_filled_picks_candidate_by_mode():
    d = {"bins_filled_pooled": 7, "bins_filled_per_walker": 4}
    _resolve_bins_filled(d, "pooled")
    assert d["bins_filled"] == 7
    assert "bins_filled_pooled" not in d and "bins_filled_per_walker" not in d


def test_resolve_bins_filled_per_walker_picks_intersection():
    d = {"bins_filled_pooled": 7, "bins_filled_per_walker": 4}
    _resolve_bins_filled(d, "per_walker")
    assert d["bins_filled"] == 4


def test_resolve_bins_filled_no_candidates_is_noop():
    d = {"bins_filled": 5}
    _resolve_bins_filled(d, "per_walker")
    assert d == {"bins_filled": 5}


def test_pooled_filled_parity_tracks_one_over_t_gate():
    """``bins_filled == bins_known`` iff the pooled 1/t halving gate fires.

    Ties the reported coverage to the criterion the coordinator
    actually consults (``_summed_histogram_halving_criterion_met``),
    so the printer cannot show ``fill/known`` reaching parity while
    the gate refuses to halve (or vice versa) without this test
    failing.
    """
    from mchammer_pt.wl_coordinator import (
        WalkerPostBlockState,
        _summed_histogram_halving_criterion_met,
    )

    def _snap(histogram: dict[int, int]) -> WalkerPostBlockState:
        return WalkerPostBlockState(
            halving_criterion_met=False,
            fill_factor=1.0,
            entropy={},
            step=10,
            window_entry_step=0,
            histogram=histogram,
            reached_energy_window=True,
            current_energy=0.0,
        )

    cases = [
        [{0: 1, 1: 2, 2: 3}],          # all positive -> gate fires
        [{0: 1, 1: 0, 2: 3}],          # a zero -> gate refuses
        [{0: 1, 1: 0}, {0: 0, 1: 4}],  # union covers both -> gate fires
        [{0: 1, 1: 0}, {0: 2, 1: 0}],  # bin 1 zero in both -> refuses
    ]
    for histograms in cases:
        known = set().union(*[set(h) for h in histograms])
        at_parity = _compute_filled_bins(histograms, "pooled") == len(known)
        gate_fires = _summed_histogram_halving_criterion_met(
            [_snap(h) for h in histograms], 0.8, "1_over_t"
        )
        assert at_parity == gate_fires


def test_compute_recency_flatness_single_is_min_over_mean():
    from mchammer_pt.wl_coordinator import _compute_recency_flatness
    w = {0: 2.0, 1: 4.0}            # min/mean = 2/3
    assert _compute_recency_flatness([w], "pooled") == 2.0 / 3.0
    assert _compute_recency_flatness([w], "per_walker") == 2.0 / 3.0


def test_compute_recency_flatness_pooled_sums_weights():
    from mchammer_pt.wl_coordinator import _compute_recency_flatness
    a = {0: 1.0, 1: 0.0}
    b = {0: 0.0, 1: 1.0}
    # summed = {0:1,1:1} -> min/mean = 1.0
    assert _compute_recency_flatness([a, b], "pooled") == 1.0


def test_compute_recency_flatness_per_walker_is_min_over_walkers():
    from mchammer_pt.wl_coordinator import _compute_recency_flatness
    a = {0: 4.0, 1: 4.0}           # min/mean = 1.0
    b = {0: 1.0, 1: 3.0}           # min/mean = 1/2
    assert _compute_recency_flatness([a, b], "per_walker") == 0.5


def test_compute_recency_flatness_per_walker_drops_unusable_walkers():
    from mchammer_pt.wl_coordinator import _compute_recency_flatness
    # One walker has no usable weights (zero-mean -> None) and is dropped;
    # the result is the surviving walker's min/mean, not poisoned by None.
    empty = {0: 0.0, 1: 0.0}       # _min_over_mean -> None
    usable = {0: 1.0, 1: 3.0}      # min/mean = 1/2
    assert _compute_recency_flatness([empty, usable], "per_walker") == 0.5


def test_compute_recency_flatness_empty_or_zero_mean_is_none():
    from mchammer_pt.wl_coordinator import _compute_recency_flatness
    assert _compute_recency_flatness([], "pooled") is None
    assert _compute_recency_flatness([{}], "pooled") is None
    assert _compute_recency_flatness([{0: 0.0, 1: 0.0}], "pooled") is None


def test_resolve_recency_flatness_picks_by_mode():
    from mchammer_pt.wl_coordinator import _resolve_recency_flatness
    d = {"recency_flatness_pooled": 0.9, "recency_flatness_per_walker": 0.4}
    _resolve_recency_flatness(d, "per_walker")
    assert d["recency_flatness"] == 0.4
    assert "recency_flatness_pooled" not in d
    assert "recency_flatness_per_walker" not in d


def test_resolve_recency_flatness_no_candidates_is_noop():
    from mchammer_pt.wl_coordinator import _resolve_recency_flatness
    d = {"recency_flatness": 0.7}
    _resolve_recency_flatness(d, "pooled")
    assert d == {"recency_flatness": 0.7}


class TestOneOverTGateValidators:
    def test_one_over_t_gate_accepts_valid(self) -> None:
        _validate_one_over_t_gate("visit_once")
        _validate_one_over_t_gate("flatness")

    def test_one_over_t_gate_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="one_over_t_gate"):
            _validate_one_over_t_gate("flat")

    def test_bp_stall_multiple_accepts_positive(self) -> None:
        assert _validate_bp_stall_multiple(4) == 4.0
        assert _validate_bp_stall_multiple(2.5) == 2.5
        assert _validate_bp_stall_multiple(np.float64(3.0)) == 3.0

    def test_bp_stall_multiple_rejects_nonpositive_and_nonfinite(self) -> None:
        for bad in (0, -1.0, float("inf"), float("nan"), np.float64("nan"), True):
            with pytest.raises(ValueError, match="bp_stall_multiple"):
                _validate_bp_stall_multiple(bad)


class TestSlotViewDefaults:
    def test_new_fields_default_to_today(self) -> None:
        view = SlotView(
            walker_states=(),
            phase="halving",
            flatness_mode="pooled",
            merge_cadence="at_halve",
            schedule="1_over_t",
            flatness_limit=0.8,
        )
        assert view.one_over_t_gate == "visit_once"
        assert view.bp_stall_multiple == 4.0
        assert view.last_halve_step is None
        assert view.first_halve_duration is None
