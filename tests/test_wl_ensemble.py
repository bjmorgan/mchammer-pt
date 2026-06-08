"""Unit tests for CoordinatedWangLandauEnsemble."""

from __future__ import annotations

import numpy as np
import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _make_ensemble(**kwargs):
    """Construct a CoordinatedWangLandauEnsemble on the toy CE fixture."""
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce, atoms = make_wl_ce(), make_wl_atoms()
    return CoordinatedWangLandauEnsemble(
        structure=atoms,
        calculator=ClusterExpansionCalculator(atoms, ce),
        energy_spacing=0.1,
        energy_limit_left=None,
        energy_limit_right=None,
        random_seed=0,
        dc_filename=None,
        **kwargs,
    )


def test_update_entropy_does_not_halve_on_synthetic_flat_histogram():
    """A synthetic flat histogram does not trigger halving in the subclass."""
    e = _make_ensemble(flatness_check_interval=10)
    # Pre-seed a flat histogram and entropy so that mchammer's check
    # would normally fire halving on the next interval boundary.
    e._histogram = {0: 1000, 1: 1000, 2: 1000}
    e._entropy = {0: 1.0, 1: 1.0, 2: 1.0}
    e._reached_energy_window = True
    e._step = 100  # multiple of interval

    f_before = e._fill_factor
    history_len_before = len(e._fill_factor_history)

    # Call _update_entropy directly with a current bin; this simulates
    # what mchammer would do on a step.
    e._update_entropy(0)

    assert e._fill_factor == pytest.approx(f_before)
    assert len(e._fill_factor_history) == history_len_before
    # Bin updates survive: the called bin increments, others remain
    # unchanged (upstream halving would zero them).
    assert e._histogram[0] == 1001
    assert e._histogram[1] == 1000
    assert e._histogram[2] == 1000


def test_update_entropy_reshifts_entropy_periodically():
    """Periodic min-shift fires at flatness_check_interval boundaries."""
    e = _make_ensemble(flatness_check_interval=10)
    e._reached_energy_window = True
    e._entropy = {0: 5.0, 1: 7.0, 2: 9.0}
    e._histogram = {0: 0, 1: 0, 2: 0}
    e._step = 10  # boundary

    e._update_entropy(0)
    # After update: entropy[0] += 1.0 (current f), then min-shift.
    # entropy = {0: 6.0, 1: 7.0, 2: 9.0}; min = 6.0; shifted to
    # {0: 0.0, 1: 1.0, 2: 3.0}.
    assert e._entropy[0] == pytest.approx(0.0)
    assert e._entropy[1] == pytest.approx(1.0)
    assert e._entropy[2] == pytest.approx(3.0)


def test_update_entropy_one_over_t_phase_tracks_inverse_t():
    """In 1/t phase, fill_factor tracks 1/(step - window_entry + 1)."""
    e = _make_ensemble(
        schedule="1_over_t", flatness_check_interval=1_000_000
    )
    e._reached_energy_window = True
    e._phase = "1_over_t"
    e._window_entry_step = 10
    e._step = 110
    e._fill_factor = 0.5  # whatever; should be overwritten

    e._update_entropy(0)
    # t = step - entry + 1 = 110 - 10 + 1 = 101
    assert e._fill_factor == pytest.approx(1.0 / 101)


def test_update_entropy_adds_bin_to_visited_bins_when_in_window():
    """`_update_entropy` adds the current bin to `_visited_bins` only
    once the walker has reached the window. Pre-window transient bins
    don't pollute the set.
    """
    e = _make_ensemble(flatness_check_interval=10)
    # Pre-window: not added.
    e._reached_energy_window = False
    initial = set(e._visited_bins)
    e._update_entropy(99)
    assert e._visited_bins == initial

    # In-window: added.
    e._reached_energy_window = True
    e._update_entropy(42)
    assert 42 in e._visited_bins


def test_recency_visits_per_bin_validated_positive():
    """A non-positive recency window is rejected at construction."""
    with pytest.raises(ValueError):
        _make_ensemble(recency_visits_per_bin=0)
    with pytest.raises(ValueError):
        _make_ensemble(recency_visits_per_bin=-5)


def test_recency_visits_per_bin_rejects_non_integer():
    """A non-integer recency window is rejected, not silently truncated."""
    with pytest.raises(ValueError, match="positive integer"):
        _make_ensemble(recency_visits_per_bin=2.5)
    # An integer-valued float is accepted.
    e = _make_ensemble(recency_visits_per_bin=1000.0)
    assert e._recency_visits_per_bin == 1000
    assert isinstance(e._recency_visits_per_bin, int)


def test_recency_visits_per_bin_rejects_bool():
    """A bool recency window is rejected, not coerced to 0 or 1."""
    with pytest.raises(ValueError, match="positive integer"):
        _make_ensemble(recency_visits_per_bin=True)
    with pytest.raises(ValueError, match="positive integer"):
        _make_ensemble(recency_visits_per_bin=False)


def test_recency_visits_per_bin_rejects_non_finite():
    """Non-finite recency windows fail with the consistent error message."""
    with pytest.raises(ValueError, match="positive integer"):
        _make_ensemble(recency_visits_per_bin=float("inf"))
    with pytest.raises(ValueError, match="positive integer"):
        _make_ensemble(recency_visits_per_bin=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        _make_ensemble(recency_visits_per_bin=np.float64("inf"))


def test_recency_effective_weights_keys_match_known_bins():
    """Weights are keyed by known bins and start at zero."""
    e = _make_ensemble()
    e._reached_energy_window = True
    e._histogram = {0: 0, 1: 0, 2: 0}
    w = e.recency_effective_weights()
    assert set(w) == {0, 1, 2}
    assert all(v == 0.0 for v in w.values())


def test_recency_weight_increments_on_visit_and_decays():
    """A fresh visit reads 1.0; an older visit has decayed below it."""
    e = _make_ensemble(recency_visits_per_bin=1000)
    e._reached_energy_window = True
    e._histogram = {0: 0, 1: 0}
    e._record_recency_visit(0, step=0)
    e._record_recency_visit(1, step=10_000)
    w = e.recency_effective_weights(step=10_000)
    assert w[1] == 1.0
    assert 0.0 < w[0] < 1.0
    assert w[0] < w[1]


def test_recency_uniform_visits_give_flat_weights():
    """Round-robin visits drive the per-bin weights towards uniform."""
    e = _make_ensemble(recency_visits_per_bin=10)
    e._reached_energy_window = True
    e._histogram = {0: 0, 1: 0, 2: 0}
    for step in range(3000):
        e._record_recency_visit(step % 3, step=step)
    w = e.recency_effective_weights(step=2999)
    vals = np.array(list(w.values()))
    assert vals.min() / vals.mean() > 0.8


def test_validate_dos_snapshot_ratio_accepts_none_and_above_one():
    """None disables; any finite ratio > 1.0 is accepted and returned as float."""
    from mchammer_pt.wl_ensemble import _validate_dos_snapshot_ratio

    assert _validate_dos_snapshot_ratio(None) is None
    assert _validate_dos_snapshot_ratio(2.0) == 2.0
    got = _validate_dos_snapshot_ratio(10**0.5)  # sqrt(10), a coarser ladder
    assert isinstance(got, float)
    assert got == pytest.approx(10**0.5)


def test_validate_dos_snapshot_ratio_rejects_le_one_and_non_finite():
    """A ratio of 1 (snapshot every step), <1, non-finite, or bool is rejected."""
    from mchammer_pt.wl_ensemble import _validate_dos_snapshot_ratio

    for bad in (1.0, 0.5, 0.0, -2.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="None or a finite float"):
            _validate_dos_snapshot_ratio(bad)


def test_ensemble_initialises_empty_snapshot_store():
    """A fresh ensemble has empty snapshot dicts and no recorded rung."""
    e = _make_ensemble(dos_snapshot_ratio=2.0)
    assert e._entropy_snapshots == {}
    assert e._fill_factor_snapshots == {}
    assert e._max_snapshot_rung is None


def test_ensemble_dos_snapshot_ratio_defaults_to_two():
    """The default ladder is factor-2 (a snapshot each time f halves)."""
    e = _make_ensemble()
    assert e._dos_snapshot_ratio == 2.0


def test_ensemble_dos_snapshot_ratio_none_disables():
    """None is stored verbatim to mean 'disabled'."""
    e = _make_ensemble(dos_snapshot_ratio=None)
    assert e._dos_snapshot_ratio is None


def test_ensemble_rejects_bad_dos_snapshot_ratio():
    """A ratio <= 1.0 is rejected at construction."""
    with pytest.raises(ValueError, match="None or a finite float"):
        _make_ensemble(dos_snapshot_ratio=1.0)


def _drive_one_over_t(e, steps):
    """Call `_update_entropy(0)` once per entry in `steps`, setting `_step`.

    Sets up a forced 1/t phase with window entry at step 0, so
    ``f = 1/(step + 1)`` on each call (matching the live 1/t update).
    """
    e._reached_energy_window = True
    e._phase = "1_over_t"
    e._window_entry_step = 0
    e._entropy = {0: 1.0}
    for step in steps:
        e._step = step
        e._update_entropy(0)


def test_one_over_t_records_snapshots_on_factor_two_ladder():
    """Snapshots land on the f = 2^-k ladder; one per rung, baseline first."""
    e = _make_ensemble(
        schedule="1_over_t",
        flatness_check_interval=1_000_000,
        dos_snapshot_ratio=2.0,
    )
    # f = 1/(step+1): rungs at f = 1, 1/2, 1/4, 1/8 -> steps 0, 1, 3, 7.
    _drive_one_over_t(e, range(8))
    assert set(e._fill_factor_snapshots) == {0, 1, 3, 7}
    assert e._fill_factor_snapshots[0] == pytest.approx(1.0)
    assert e._fill_factor_snapshots[1] == pytest.approx(1.0 / 2)
    assert e._fill_factor_snapshots[3] == pytest.approx(1.0 / 4)
    assert e._fill_factor_snapshots[7] == pytest.approx(1.0 / 8)
    # Each snapshot captured the live entropy at that step.
    assert set(e._entropy_snapshots) == {0, 1, 3, 7}
    assert e._entropy_snapshots[7] == e._entropy
    # The step-1 snapshot holds its own intermediate entropy (2.5), not a
    # live reference that would later read the final value.
    assert e._entropy_snapshots[1] == pytest.approx({0: 2.5})


def test_coarser_ratio_records_fewer_snapshots():
    """A larger ratio lays down fewer rungs over the same f-range."""
    e = _make_ensemble(
        schedule="1_over_t",
        flatness_check_interval=1_000_000,
        dos_snapshot_ratio=4.0,
    )
    # ratio 4: rungs at f = 1, 1/4, 1/16 -> steps 0, 3, 15.
    _drive_one_over_t(e, range(16))
    assert set(e._fill_factor_snapshots) == {0, 3, 15}


def test_no_snapshots_in_halving_phase():
    """The trigger only fires in the 1/t phase."""
    e = _make_ensemble(
        flatness_check_interval=1_000_000, dos_snapshot_ratio=2.0
    )
    e._reached_energy_window = True
    e._phase = "halving"
    e._entropy = {0: 1.0}
    for step in range(8):
        e._step = step
        e._update_entropy(0)
    assert e._fill_factor_snapshots == {}
    assert e._entropy_snapshots == {}


def test_dos_snapshot_ratio_none_records_nothing():
    """Disabling the ladder leaves the store empty even in the 1/t phase."""
    e = _make_ensemble(
        schedule="1_over_t",
        flatness_check_interval=1_000_000,
        dos_snapshot_ratio=None,
    )
    _drive_one_over_t(e, range(8))
    assert e._fill_factor_snapshots == {}
    assert e._entropy_snapshots == {}


def test_snapshotting_does_not_alter_entropy_or_halving_history():
    """Snapshotting is a pure side observation: with the ladder on vs off,
    the entropy, histogram, fill factor, and halving history are identical."""
    def run(ratio):
        e = _make_ensemble(
            schedule="1_over_t",
            flatness_check_interval=1_000_000,
            dos_snapshot_ratio=ratio,
        )
        _drive_one_over_t(e, range(64))
        return e

    on = run(2.0)
    off = run(None)
    assert on._entropy == off._entropy
    assert on._histogram == off._histogram
    assert on._fill_factor == pytest.approx(off._fill_factor)
    assert on._fill_factor_history == off._fill_factor_history
    assert on._entropy_history == off._entropy_history
    # Only the snapshot store differs.
    assert on._fill_factor_snapshots != {}
    assert off._fill_factor_snapshots == {}


def test_recency_lazy_decay_matches_eager_reference():
    """Decay-on-read reproduces an eager per-step decay reference exactly.

    ``recency_effective_weights`` decays each bin's weight from its last
    recorded step to the read step in one shot. That lazy update must
    equal an eager loop that decays every bin on every step. The
    equivalence holds because ``_recency_alpha`` is constant across the
    sequence: ``_record_recency_visit`` never mutates ``_histogram`` (it
    only touches ``_recent_weight`` / ``_recent_last_step``), so the
    known-bin count stays at 3 throughout and ``alpha = 1/(10*3)``.
    """
    e = _make_ensemble(recency_visits_per_bin=10)
    e._reached_energy_window = True
    e._histogram = {0: 0, 1: 0, 2: 0}
    visits = [(0, 0), (1, 1), (0, 5), (2, 9), (1, 12)]  # (bin, step)
    for b, step in visits:
        e._record_recency_visit(b, step=step)
    read_step = 20
    got = e.recency_effective_weights(step=read_step)
    # Eager reference: alpha is fixed (histogram fully known = 3 bins here).
    alpha = 1.0 / (10 * 3)
    expected = {0: 0.0, 1: 0.0, 2: 0.0}
    last: dict[int, int | None] = {0: None, 1: None, 2: None}
    for b, step in visits:
        if last[b] is not None:
            expected[b] *= (1 - alpha) ** (step - last[b])
        expected[b] += 1.0
        last[b] = step
    for b in expected:
        if last[b] is not None:
            expected[b] *= (1 - alpha) ** (read_step - last[b])
    assert np.allclose(
        [got[b] for b in (0, 1, 2)], [expected[b] for b in (0, 1, 2)]
    )


def test_rung_tracker_rebuild_on_resume_continues_without_duplicate_or_skip():
    """After resume, the rebuilt rung tracker neither re-records an
    already-captured rung nor skips the next one.

    Pins the production scenario: a long 1/t run is checkpointed and
    resumed mid-regime. ``_max_snapshot_rung`` is not persisted; it is
    rebuilt from the restored ``_fill_factor_snapshots`` and must leave
    the ladder seamless across the resume boundary.
    """
    # Pre-resume: record rungs 0, 1, 2, 3 (steps 0, 1, 3, 7 on the
    # f = 1/(step + 1) ladder).
    pre = _make_ensemble(
        schedule="1_over_t",
        flatness_check_interval=1_000_000,
        dos_snapshot_ratio=2.0,
    )
    _drive_one_over_t(pre, range(8))
    assert set(pre._fill_factor_snapshots) == {0, 1, 3, 7}
    assert pre._max_snapshot_rung == 3

    # Simulate resume: a fresh ensemble inherits the persisted store and
    # rebuilds the in-memory tracker from it (no rung tracker is persisted).
    resumed = _make_ensemble(
        schedule="1_over_t",
        flatness_check_interval=1_000_000,
        dos_snapshot_ratio=2.0,
    )
    resumed._fill_factor_snapshots = dict(pre._fill_factor_snapshots)
    resumed._entropy_snapshots = {
        step: dict(entropy) for step, entropy in pre._entropy_snapshots.items()
    }
    resumed._rebuild_max_snapshot_rung()
    assert resumed._max_snapshot_rung == 3

    # Continue in the 1/t regime from step 8 (entry still at step 0, so
    # f = 1/(step + 1)). Steps 8..14 sit in rung 3 (already recorded); step
    # 15 crosses into rung 4 (f = 1/16) and must record exactly once.
    resumed._reached_energy_window = True
    resumed._phase = "1_over_t"
    resumed._window_entry_step = 0
    resumed._entropy = {0: 1.0}
    for step in range(8, 16):
        resumed._step = step
        resumed._update_entropy(0)

    # The already-captured rungs are intact and exactly one new snapshot
    # was added, at the next rung -- no duplicate, no skip.
    assert set(resumed._fill_factor_snapshots) == {0, 1, 3, 7, 15}
    assert resumed._fill_factor_snapshots[15] == pytest.approx(1.0 / 16)
    assert resumed._max_snapshot_rung == 4
