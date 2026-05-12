"""End-to-end correctness check: REWL recovers the analytic 2D Ising DOS.

Marked `slow` and not run by default. Run manually before any
production REWL use:

    pytest tests/integration/test_rewl_2d_ising.py -v -m slow
"""

from __future__ import annotations

import pytest
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]


@pytest.mark.slow
def test_rewl_recovers_analytic_2d_ising_dos():
    """4x4 2D Ising via REWL: stitched ln g(E) matches analytic within tolerance."""
    pytest.xfail(
        "REWL stitching helper and per-window starting configurations "
        "not yet implemented. Once the stitching helper lands and a "
        "configuration-generator gives each window an in-window starter, "
        "delete this xfail and the body below becomes the real correctness "
        "gate: stitched ln g(E) compared against the analytic 4x4 2D "
        "Ising DOS."
    )

    # Sketch of the eventual correctness gate. Not currently reachable.
    from mchammer_pt.wl import WangLandauParallelTempering

    primitive = Atoms(
        "Au", positions=[[0, 0, 0]], cell=[1, 1, 10], pbc=True
    )
    cs = ClusterSpace(
        structure=primitive, cutoffs=[1.1],
        chemical_symbols=["Ag", "Au"],
    )
    ce = ClusterExpansion(cluster_space=cs, parameters=[0, 0, 2])
    structure = primitive.repeat((4, 4, 1))
    for k in range(8):
        structure[k].symbol = "Ag"

    # NOTE: passing four copies of `structure` is wrong here — every
    # replica's starting energy is 16, which lies outside three of the
    # four windows. The eventual real test needs per-window starting
    # configurations whose energies fall in their assigned windows.
    pt = WangLandauParallelTempering.from_bin_count(
        cluster_expansion=ce,
        atoms=[structure.copy() for _ in range(4)],
        n_bins=4,
        energy_spacing=1.0,
        minimum_energy=-32.0,
        maximum_energy=32.0,
        overlap=4,
        block_size=len(structure) * 1000,
        random_seed=42,
    )
    pt.run(n_cycles=500)
