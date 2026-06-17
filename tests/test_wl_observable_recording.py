"""Tests for per-bin observable recording wired into CoordinatedWangLandauEnsemble.

Covers:
- attach_observable_recorder stores a recorder keyed by tag
- duplicate-tag attach raises ValueError
- recorder accumulates per-bin moments during a frozen_g run
- recorded microcanonical mean <E>(bin) lies within the bin's energy range
- recording respects the observer's interval
- recording does not occur before the walker reaches its energy window
"""

from __future__ import annotations

from typing import Any

import pytest
from mchammer.observers.base_observer import BaseObserver

from tests._wl_fixtures import make_wl_ce, make_wl_ensemble

# ---------------------------------------------------------------------------
# Minimal test observers
# ---------------------------------------------------------------------------


class _EnergyObserver(BaseObserver):
    """Returns the total cluster-expansion energy of the current structure.

    Holds a reference to the cluster expansion and calls
    ``ce.predict(structure) * len(structure)`` to match the total energy
    the WL ensemble uses for binning (``ClusterExpansionCalculator``
    returns per-atom energy via ``ce.predict``; the ensemble multiplies
    by ``n_atoms`` via ``calculate_total``).
    """

    def __init__(self, ce: Any, interval: int = 1, tag: str = "energy") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)
        self._ce = ce

    def get_observable(self, structure: Any) -> float:
        return float(self._ce.predict(structure)) * len(structure)


class _ConstantObserver(BaseObserver):
    """Returns a fixed constant regardless of configuration."""

    def __init__(
        self, value: float = 42.0, interval: int = 1, tag: str = "const"
    ) -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)
        self._value = value

    def get_observable(self, structure: Any) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Helper: warm up entropy, then build a frozen ensemble
# ---------------------------------------------------------------------------


def _warm_up_and_freeze(random_seed_warm: int = 10, random_seed_frozen: int = 11):
    """Return a frozen_g ensemble with a warmed-up entropy seeded in.

    Runs 500 steps on a normal ensemble to populate g(E) and f, then
    constructs a frozen_g=True ensemble with that g and f planted in.
    """
    warm = make_wl_ensemble(
        flatness_check_interval=1_000_000,
        random_seed=random_seed_warm,
    )
    warm.run(500)
    g0 = dict(warm._entropy)
    f0 = warm._fill_factor
    assert len(g0) > 1, "warm-up must reach more than one physical bin"

    frozen = make_wl_ensemble(
        frozen_g=True,
        flatness_check_interval=1_000_000,
        random_seed=random_seed_frozen,
    )
    frozen._entropy = g0
    frozen._fill_factor = f0
    return frozen


# ---------------------------------------------------------------------------
# attach_observable_recorder: basic registration
# ---------------------------------------------------------------------------


def test_attach_stores_recorder_keyed_by_tag():
    """attach_observable_recorder stores a recorder keyed by observer.tag."""
    e = make_wl_ensemble()
    obs = _ConstantObserver(tag="myobs")
    e.attach_observable_recorder(obs)
    assert "myobs" in e._recorders
    assert e._recorders["myobs"].tag == "myobs"


def test_attach_duplicate_tag_raises():
    """attach_observable_recorder raises ValueError for duplicate tag."""
    e = make_wl_ensemble()
    obs1 = _ConstantObserver(tag="dup")
    obs2 = _ConstantObserver(tag="dup")
    e.attach_observable_recorder(obs1)
    with pytest.raises(ValueError, match="dup"):
        e.attach_observable_recorder(obs2)


def test_attach_two_observers_with_distinct_tags():
    """Two observers with different tags are both registered."""
    e = make_wl_ensemble()
    e.attach_observable_recorder(_ConstantObserver(tag="a"))
    e.attach_observable_recorder(_ConstantObserver(tag="b"))
    assert "a" in e._recorders
    assert "b" in e._recorders


# ---------------------------------------------------------------------------
# Analytic <E>(bin) anchor test
# ---------------------------------------------------------------------------


def test_energy_mean_lies_within_bin_range():
    """Microcanonical <E> recorded per bin must lie within that bin's energy range.

    Attach an observer that returns the CE energy (the same value the
    ensemble uses for binning). For each populated bin b, the recorded
    mean sum/count must satisfy:

        b * spacing - spacing/2 <= mean <= b * spacing + spacing/2

    This is the robust, FP-safe assertion: it confirms (a) the observer
    was evaluated, (b) the result was binned by the *current* energy bin,
    without requiring exact equality between an independently recomputed
    energy and the ensemble's cached bin index.
    """
    ce = make_wl_ce()
    frozen = _warm_up_and_freeze(random_seed_warm=20, random_seed_frozen=21)

    energy_obs = _EnergyObserver(ce=ce, interval=1, tag="energy")
    frozen.attach_observable_recorder(energy_obs)
    frozen.run(2_000)

    rec = frozen._recorders["energy"]
    state = rec.to_state()
    spacing = 0.1  # matches make_wl_ensemble(energy_spacing=0.1)

    populated_bins = state["bins"]
    assert len(populated_bins) >= 2, (
        f"expected at least 2 populated bins, got {len(populated_bins)}"
    )

    for b, cnt, s in zip(state["bins"], state["count"], state["sum"], strict=False):
        assert cnt > 0
        mean = s[0] / cnt
        lo = b * spacing - spacing / 2
        hi = b * spacing + spacing / 2
        assert lo - 1e-9 <= mean <= hi + 1e-9, (
            f"bin {b}: mean energy {mean:.6f} not in [{lo:.4f}, {hi:.4f}]"
        )


# ---------------------------------------------------------------------------
# Interval gating: total count matches steps / interval
# ---------------------------------------------------------------------------


def test_recording_respects_interval():
    """Total recorded count across bins equals in-window steps / interval (approx).

    Attach an observer with interval=k and run N_steps steps in a frozen
    ensemble that is already in-window (g planted so that entry is quick).
    The total count across all bins must equal floor(in_window_steps / k).

    We cannot know in advance how many steps are in-window vs. out-of-window,
    so we use a frozen ensemble that has already passed window entry
    (``_reached_energy_window`` set to True in setup) and count the
    in-window steps directly.
    """
    frozen = _warm_up_and_freeze(random_seed_warm=30, random_seed_frozen=31)
    # Force the walker to be already in-window before we run, so that
    # all N_steps contribute to in-window recording.
    frozen._reached_energy_window = True

    k = 7
    obs = _ConstantObserver(interval=k, tag="c")
    frozen.attach_observable_recorder(obs)

    N_steps = 700
    frozen.run(N_steps)

    rec = frozen._recorders["c"]
    state = rec.to_state()
    total_count = sum(state["count"])

    # Steps 0..N_steps-1; count = number of steps where step % k == 0.
    # The step counter starts at whatever value it had after warm-up,
    # so we compute directly from the recorder: each in-window step
    # where step % k == 0 fires once.  We can bound total_count:
    # floor(N_steps / k) - 1 <= total_count <= ceil(N_steps / k) + 1
    # (the ±1 accounts for the start offset).
    lo = N_steps // k - 1
    hi = N_steps // k + 1
    assert lo <= total_count <= hi, (
        f"total count {total_count} not in [{lo}, {hi}] "
        f"for N_steps={N_steps}, interval={k}"
    )


# ---------------------------------------------------------------------------
# Out-of-window: recording only starts after window entry
# ---------------------------------------------------------------------------


def test_recording_only_populates_in_window_bins():
    """Bins populated by the recorder are all in-window bins.

    Attach an energy observer to a NORMAL (non-frozen) ensemble, run
    enough steps to trigger window entry, then check that every
    populated bin satisfies ``_inside_energy_window``.  Since the
    ensemble starts outside the window and steps in, if recording
    happened before window entry the recorder would have accumulated
    bins that are NOT in the window.
    """
    e = make_wl_ensemble(
        frozen_g=False,
        flatness_check_interval=1_000_000,
        random_seed=40,
    )
    ce = make_wl_ce()
    energy_obs = _EnergyObserver(ce=ce, interval=1, tag="energy")
    e.attach_observable_recorder(energy_obs)

    e.run(1_000)

    rec = e._recorders["energy"]
    state = rec.to_state()

    # If no bins are populated (walker never reached window), skip — the
    # test infrastructure is fine, there is just nothing to assert.
    if not state["bins"]:
        pytest.skip("walker did not reach the energy window in 1 000 steps")

    for b in state["bins"]:
        assert e._inside_energy_window(b), (
            f"recorder populated out-of-window bin {b}; "
            "recording must be gated on _reached_energy_window"
        )
