"""Tests for frozen_g mode on CoordinatedWangLandauEnsemble.

In frozen_g mode the density of states g(E) is held fixed: no DOS
updates during the run, while the WL acceptance criterion continues to
use the frozen g so the walk stays flat-in-energy.
"""

from __future__ import annotations

import pytest

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _make_ensemble(**kwargs):
    """Construct a CoordinatedWangLandauEnsemble on the toy CE fixture.

    Reuses the same construction path as ``test_wl_ensemble.py`` so
    there is one canonical fixture for the WL unit tests.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce, atoms = make_wl_ce(), make_wl_atoms()
    kwargs.setdefault("random_seed", 0)
    return CoordinatedWangLandauEnsemble(
        structure=atoms,
        calculator=ClusterExpansionCalculator(atoms, ce),
        energy_spacing=0.1,
        energy_limit_left=None,
        energy_limit_right=None,
        dc_filename=None,
        **kwargs,
    )


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

    1. Run a short halving-phase warm-up to populate ``_entropy`` and
       ``_fill_factor`` (so the acceptance criterion has a non-trivial g
       to use during the frozen run).
    2. Snapshot ``g0`` and ``f0``.
    3. Build a second ensemble with ``frozen_g=True`` carrying the same
       ``g0`` and ``f0``, then run 2 000 steps.
    4. Assert ``_entropy == g0`` (bit-identical dict) and
       ``_fill_factor == pytest.approx(f0)``.
    """
    # --- warm-up run to populate g and f ---
    warm = _make_ensemble(flatness_check_interval=1_000_000, random_seed=1)
    warm._reached_energy_window = True
    # Drive 200 direct _update_entropy calls so _entropy is well-populated
    # without touching the acceptance machinery.
    for step in range(200):
        warm._step = step
        warm._update_entropy(0)

    # Snapshot: copy so the warm-up ensemble cannot alias our reference.
    g0 = dict(warm._entropy)
    f0 = warm._fill_factor
    assert len(g0) >= 1, "warm-up must have at least one bin"

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

    frozen.run(2_000)

    # g must be bit-identical (not just approximately equal).
    assert frozen._entropy == g0, (
        "frozen_g=True must not mutate _entropy during run"
    )
    # f must be unchanged (no 1/t recompute, no halving).
    assert frozen._fill_factor == pytest.approx(f0), (
        "frozen_g=True must not mutate _fill_factor during run"
    )


def test_frozen_g_walk_explores_multiple_bins():
    """The walk still moves between energy bins in frozen_g mode.

    The acceptance criterion continues to use the frozen g, so the walk
    remains flat-in-energy and is not stuck in one bin.  Collect all bins
    visited via ``_visited_bins`` (which ``_update_entropy`` populates
    from ``_reached_energy_window``); assert more than one distinct bin
    was reached.
    """
    # Warm up to get a populated g.
    warm = _make_ensemble(flatness_check_interval=1_000_000, random_seed=3)
    warm._reached_energy_window = True
    for step in range(500):
        warm._step = step
        warm._update_entropy(step % 3)

    g0 = dict(warm._entropy)
    f0 = warm._fill_factor

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
        "frozen_g walk must visit more than one energy bin "
        f"(visited: {frozen._visited_bins})"
    )
