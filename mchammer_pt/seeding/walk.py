"""Confined-walk primitive: one walk yields at most one in-band seed.

Wraps ``mchammer_moves.CustomWangLandauEnsemble``'s built-in window
search. While the configuration is outside ``[lo, hi]`` the ensemble
accepts moves that reduce the distance to the band (penalty walk);
once inside it confines to the window. The WL bookkeeping (entropy,
histogram, fill factor) is incidental here: the ensemble is used purely
as a windowed walker.
"""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import CustomWangLandauEnsemble, Move


def _in_band(e: float, lo: float | None, hi: float | None) -> bool:
    """Whether energy ``e`` lies in ``[lo, hi]`` (``None`` = unbounded)."""
    return (lo is None or e >= lo) and (hi is None or e <= hi)


def confined_walk(
    start: Atoms,
    calculator: ClusterExpansionCalculator,
    moves: Sequence[tuple[Move, float]],
    *,
    lo: float | None,
    hi: float | None,
    energy_spacing: float,
    window_search_penalty: float,
    n_steps: int,
    seed: int,
) -> Atoms | None:
    """Drive ``start`` into ``[lo, hi]`` and return an in-band config.

    Builds a ``CustomWangLandauEnsemble`` for the window, runs
    ``n_steps`` trial steps, and returns the final structure if its
    energy is in ``[lo, hi]``; otherwise returns ``None`` (the budget
    was exhausted before the band was reached).

    Args:
        start: starting structure (may be outside the window).
        calculator: cluster-expansion calculator bound to the lattice.
        moves: move set as ``(Move, weight)`` pairs.
        lo: lower window edge, or ``None`` for unbounded.
        hi: upper window edge, or ``None`` for unbounded.
        energy_spacing: WL energy-grid bin size.
        window_search_penalty: into-window bias coefficient.
        n_steps: trial-step budget for this walk.
        seed: RNG seed for this walk's ensemble.

    Returns:
        The final ``Atoms`` if in-band, else ``None``.
    """
    ens = CustomWangLandauEnsemble(
        structure=start,
        calculator=calculator,
        energy_spacing=energy_spacing,
        moves=list(moves),
        energy_limit_left=lo,
        energy_limit_right=hi,
        window_search_penalty=window_search_penalty,
        random_seed=int(seed),
        dc_filename=None,
    )
    ens.run(int(n_steps))
    if _in_band(float(ens._potential), lo, hi):
        return ens.structure
    return None
