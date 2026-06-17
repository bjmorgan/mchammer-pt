"""Tests for frozen_g mode on CoordinatedWangLandauEnsemble.

In frozen_g mode the density of states g(E) is held fixed: no DOS
updates during the run, while the WL acceptance criterion continues to
use the frozen g so the walk stays flat-in-energy.
"""

from __future__ import annotations

import pytest

from tests._wl_fixtures import make_wl_ensemble as _make_ensemble


def test_frozen_g_accepted_as_kwarg():
    """frozen_g=True is accepted without error at construction."""
    e = _make_ensemble(frozen_g=True)
    assert e._frozen_g is True


def test_frozen_g_defaults_to_false():
    """frozen_g defaults to False (existing behaviour unchanged)."""
    e = _make_ensemble()
    assert e._frozen_g is False


def test_frozen_g_entropy_and_fill_factor_unchanged_after_run():
    """Core passivity contract: g and f are bit-identical after a frozen run.

    1. Run a real MC warm-up (via :func:`_warm_up_entropy`) to populate
       ``_entropy`` and ``_fill_factor`` across multiple physical energy
       bins (so the acceptance criterion has a non-trivial g to use during
       the frozen run).
    2. Snapshot ``g0`` and ``f0``.
    3. Build a second ensemble with ``frozen_g=True`` carrying the same
       ``g0`` and ``f0``, then run 2 000 steps.
    4. Assert ``_entropy == g0`` (bit-identical dict) and
       ``_fill_factor == pytest.approx(f0)``.
    """
    # --- real MC warm-up to populate g and f across physical bins ---
    g0, f0 = _warm_up_entropy(random_seed=1)
    assert len(g0) > 1, "warm-up must populate more than one physical bin"

    # --- frozen run ---
    frozen = _make_ensemble(
        frozen_g=True,
        flatness_check_interval=1_000_000,
        random_seed=2,
    )
    # Plant the converged g and f into the frozen ensemble so it has
    # something non-trivial to accept/reject against.
    frozen._entropy = dict(g0)
    frozen._fill_factor = f0
    # Plant a non-trivial histogram too: frozen_g must leave it untouched
    # (the per-step histogram increment is inside the frozen_g guard).
    frozen._histogram = {b: 5 for b in g0}
    histogram_before = dict(frozen._histogram)

    frozen.run(2_000)

    # g must be bit-identical (not just approximately equal).
    assert frozen._entropy == g0, (
        "frozen_g=True must not mutate _entropy during run"
    )
    # f must be unchanged (no 1/t recompute, no halving).
    assert frozen._fill_factor == pytest.approx(f0), (
        "frozen_g=True must not mutate _fill_factor during run"
    )
    # the histogram must be unchanged (its increment is frozen too).
    assert frozen._histogram == histogram_before, (
        "frozen_g=True must not mutate _histogram during run"
    )


def _warm_up_entropy(random_seed: int) -> tuple[dict[int, float], float]:
    """Run a short MC warm-up and return the resulting (entropy, fill_factor).

    Uses a real :meth:`run` call so the populated bins are the physical
    energy bins the toy CE visits, not synthetic indices. The returned
    snapshot is a copy safe to plant into a second ensemble.
    """
    warm = _make_ensemble(flatness_check_interval=1_000_000, random_seed=random_seed)
    warm.run(500)
    return dict(warm._entropy), warm._fill_factor


def test_frozen_g_walk_explores_physical_bins():
    """The frozen_g walk moves between physical energy bins using the planted g.

    The acceptance criterion uses the frozen entropy seeded from a genuine
    MC warm-up, so the bins visited during the frozen run are present in
    ``g0``. If ``g0`` contained only synthetic bins absent from the live
    CE's energy landscape, every acceptance log-ratio would be zero (free
    random walk) and the test would not exercise the frozen-g path at all.
    """
    g0, f0 = _warm_up_entropy(random_seed=3)
    assert len(g0) > 1, "warm-up must reach more than one physical bin"

    frozen = _make_ensemble(
        frozen_g=True,
        flatness_check_interval=1_000_000,
        random_seed=4,
    )
    frozen._entropy = dict(g0)
    frozen._fill_factor = f0
    # Clear visited_bins so we only count the frozen run.
    frozen._visited_bins = set()

    frozen.run(2_000)

    assert len(frozen._visited_bins) > 1, (
        "frozen_g walk must visit more than one physical energy bin "
        f"(visited: {frozen._visited_bins})"
    )
    # The walk must actually use the planted g: the bins visited during the
    # frozen run must overlap with those in g0. If ``g0`` were empty or on
    # synthetic bins absent from the CE's energy landscape, the acceptance
    # log-ratio would be 0 for every move (free random walk) and this
    # overlap would be absent.
    assert frozen._visited_bins & set(g0), (
        "frozen run visited no bins from the planted g; "
        "acceptance criterion is not using the frozen entropy"
    )
