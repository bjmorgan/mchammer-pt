"""In-process tests for the confined-walk primitive against the toy CE."""

from __future__ import annotations

from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import PairSwap

from mchammer_pt.seeding.walk import _in_band, confined_walk
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def test_in_band_handles_unbounded_edges():
    # An unbounded edge is treated as satisfied on that side.
    assert _in_band(5.0, None, 10.0) is True
    assert _in_band(5.0, 0.0, None) is True
    assert _in_band(5.0, None, None) is True
    assert _in_band(5.0, 0.0, 10.0) is True
    assert _in_band(-1.0, 0.0, 10.0) is False
    assert _in_band(11.0, 0.0, 10.0) is False


def _setup():
    ce = make_wl_ce()
    atoms = make_wl_atoms(n_au=8)  # 32-site 2x2x2 fcc cubic
    calc = ClusterExpansionCalculator(atoms, ce)
    e0 = float(calc.calculate_total(occupations=atoms.numbers))
    moves = [(PairSwap(sublattice_index=0), 1.0)]
    return atoms, calc, moves, e0


def test_walk_reaches_window_from_outside():
    atoms, calc, moves, e0 = _setup()
    lo, hi = e0 + 1.5, e0 + 3.0  # a band above the start energy
    result = confined_walk(
        atoms.copy(), calc, moves,
        lo=lo, hi=hi, energy_spacing=0.25,
        window_search_penalty=2.0, n_steps=20000, seed=1,
    )
    assert result is not None
    e = float(calc.calculate_total(occupations=result.numbers))
    assert lo <= e <= hi


def test_walk_returns_none_for_unreachable_window():
    atoms, calc, moves, _ = _setup()
    # A window far above any achievable energy: never entered.
    result = confined_walk(
        atoms.copy(), calc, moves,
        lo=1.0e6, hi=1.0e6 + 1.0, energy_spacing=0.25,
        window_search_penalty=2.0, n_steps=2000, seed=1,
    )
    assert result is None
