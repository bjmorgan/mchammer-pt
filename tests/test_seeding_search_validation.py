"""Fast input- and boundary-validation tests for seed_window_configs.

These exercise the validation that runs before any worker pool is
spawned, so they stay in the fast suite.
"""

from __future__ import annotations

import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool

import pytest
from mchammer_moves import PairSwap

from mchammer_pt import seed_window_configs
from mchammer_pt.seeding.search import _validate_inputs, _walk_result
from tests._wl_fixtures import make_wl_atoms, make_wl_ce

_MOVES = [(PairSwap(sublattice_index=0), 1.0)]


def test_walk_result_maps_broken_pool_to_runtimeerror():
    # A worker killed by the OS breaks the pool; the orchestrator must
    # surface a clear RuntimeError rather than hang or leak the raw class.
    future = cf.Future()
    future.set_exception(BrokenProcessPool("worker died"))
    with pytest.raises(RuntimeError, match="died unexpectedly"):
        _walk_result(future)


def test_walk_result_maps_worker_exception_to_runtimeerror():
    future = cf.Future()
    future.set_exception(ValueError("boom"))
    with pytest.raises(RuntimeError, match="worker raised"):
        _walk_result(future)


def test_walk_result_passes_through_success():
    future = cf.Future()
    future.set_result((2, None))
    assert _walk_result(future) == (2, None)


def test_rejects_empty_windows():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_inputs([], [], _MOVES, 10)


def test_rejects_lo_not_less_than_hi():
    with pytest.raises(ValueError, match="strictly less"):
        _validate_inputs([(1.0, 1.0)], [1], _MOVES, 10)


def test_rejects_counts_length_mismatch():
    with pytest.raises(ValueError, match="one count per window"):
        _validate_inputs([(0.0, 1.0), (1.0, 2.0)], [1], _MOVES, 10)


def test_rejects_count_below_one():
    with pytest.raises(ValueError, match=">= 1"):
        _validate_inputs([(0.0, 1.0)], [0], _MOVES, 10)


def test_rejects_max_walks_below_max_count():
    with pytest.raises(ValueError, match="max_walks_per_window"):
        _validate_inputs([(0.0, 1.0)], [5], _MOVES, 4)


def test_rejects_empty_moves():
    with pytest.raises(ValueError, match="moves must be non-empty"):
        _validate_inputs([(0.0, 1.0)], [1], [], 10)


def test_random_fill_wrong_length_is_caught_at_boundary():
    ce = make_wl_ce()
    anchor = make_wl_atoms(n_au=16)

    def bad_fill(seed):
        # Twice the sites of the anchor -> incompatible lattice.
        return make_wl_atoms(n_au=16).repeat((2, 1, 1))

    with pytest.raises(ValueError, match="random_fill"):
        seed_window_configs(
            cluster_expansion=ce,
            moves=_MOVES,
            windows=[(-50.0, 0.0), (-5.0, 50.0)],
            counts=[1, 1],
            energy_spacing=0.5,
            bottom_anchor=anchor,
            random_fill=bad_fill,
            random_seed=0,
        )


def test_random_fill_wrong_type_is_caught_at_boundary():
    ce = make_wl_ce()
    anchor = make_wl_atoms(n_au=16)

    def bad_fill(seed):
        return "not atoms"

    with pytest.raises(ValueError, match="ase.Atoms"):
        seed_window_configs(
            cluster_expansion=ce,
            moves=_MOVES,
            windows=[(-50.0, 0.0), (-5.0, 50.0)],
            counts=[1, 1],
            energy_spacing=0.5,
            bottom_anchor=anchor,
            random_fill=bad_fill,
            random_seed=0,
        )


def test_random_fill_reordered_atoms_is_caught_at_boundary():
    # A fill with the anchor's exact cell and positions but a different
    # atom ordering passes type and length checks, yet its occupation
    # vector would be read against the wrong sites -> silent wrong energy.
    # The index-by-index position check must reject it.
    ce = make_wl_ce()
    anchor = make_wl_atoms(n_au=16)

    def reordering_fill(seed):
        return anchor[list(reversed(range(len(anchor))))]

    with pytest.raises(ValueError, match="positions|ordering"):
        seed_window_configs(
            cluster_expansion=ce,
            moves=_MOVES,
            windows=[(-50.0, 0.0), (-5.0, 50.0)],
            counts=[1, 1],
            energy_spacing=0.5,
            bottom_anchor=anchor,
            random_fill=reordering_fill,
            random_seed=0,
        )
