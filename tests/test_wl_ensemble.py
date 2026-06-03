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
