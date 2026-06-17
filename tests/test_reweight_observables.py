"""Tests for canonical reweighting of microcanonical observable moments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mchammer_pt.analysis.dos import reweight_canonical_from_dos
from mchammer_pt.analysis.observables import reweight_observables

T = np.array([300.0, 600.0])


def _moments(
    energies: list[float],
    counts: list[int],
    mean: object,
    sq_mean: object,
    quartic_mean: object,
    name: str = "m",
) -> pd.DataFrame:
    """Build a one-tag, one-scalar moments DataFrame.

    ``mean``/``sq_mean``/``quartic_mean`` may be scalars (broadcast over
    all bins) or per-bin arrays; they are the intended ``<O^n>(E)`` and
    are multiplied by ``count`` to form the stored sums.
    """
    c = np.asarray(counts, dtype=float)
    return pd.DataFrame(
        {
            "energy": energies,
            "count": c.astype(int),
            f"{name}_sum": c * np.asarray(mean, dtype=float),
            f"{name}_sum2": c * np.asarray(sq_mean, dtype=float),
            f"{name}_sum4": c * np.asarray(quartic_mean, dtype=float),
        }
    )


def test_constant_observable_mean_sq_and_binder():
    """A constant observable O = c gives <O>(T)=c, <O^2>(T)=c^2, U=2/3."""
    energies = [0.0, 1.0, 2.0]
    dos = pd.DataFrame({"energy": energies, "entropy": [0.0, 0.0, 0.0]})
    c = 2.0
    mom = _moments(energies, [10, 10, 10], mean=c, sq_mean=c**2, quartic_mean=c**4)

    out = reweight_observables(mom, dos, T)

    assert np.allclose(out["m_mean"], c)
    assert np.allclose(out["m_sq_mean"], c**2)
    assert np.allclose(out["m_binder"], 2.0 / 3.0)
    assert np.allclose(out["coverage"], 1.0)


def test_O_equals_E_matches_canonical_energy():
    """With O(E)=E and full coverage, <O>(T) equals the canonical <E>(T)."""
    energies = np.array([-1.0, 0.0, 1.0])
    dos = pd.DataFrame({"energy": energies, "entropy": [0.0, 0.0, 0.0]})
    mom = _moments(
        list(energies),
        [10, 10, 10],
        mean=energies,
        sq_mean=energies**2,
        quartic_mean=energies**4,
    )

    out = reweight_observables(mom, dos, T)
    ref = reweight_canonical_from_dos(dos, T)

    assert np.allclose(out["m_mean"], ref["E_mean"].to_numpy())


def test_binder_bimodal_is_two_thirds():
    """A bimodal p(O|E) at +/-m has U = 1 - m^4/(3 m^4) = 2/3."""
    energies = [0.0, 1.0, 2.0]
    dos = pd.DataFrame({"energy": energies, "entropy": [0.0, 0.0, 0.0]})
    m = 3.0
    # <O>=0, <O^2>=m^2, <O^4>=m^4 (two delta peaks at +/-m).
    mom = _moments(energies, [20, 20, 20], mean=0.0, sq_mean=m**2, quartic_mean=m**4)

    out = reweight_observables(mom, dos, T)

    assert np.allclose(out["m_mean"], 0.0)
    assert np.allclose(out["m_sq_mean"], m**2)
    assert np.allclose(out["m_binder"], 2.0 / 3.0)


def test_binder_gaussian_is_zero():
    """A Gaussian p(O|E) has <O^4> = 3 <O^2>^2, so U = 0 (contrast to bimodal)."""
    energies = [0.0, 1.0]
    dos = pd.DataFrame({"energy": energies, "entropy": [0.0, 0.0]})
    var = 4.0
    mom = _moments(
        energies, [50, 50], mean=0.0, sq_mean=var, quartic_mean=3.0 * var**2
    )

    out = reweight_observables(mom, dos, T)

    assert np.allclose(out["m_binder"], 0.0, atol=1e-12)


def test_two_scalars_reweighted_independently():
    """An S=2 observer produces independent per-scalar canonical columns."""
    energies = [0.0, 1.0]
    dos = pd.DataFrame({"energy": energies, "entropy": [0.0, 0.0]})
    mom = pd.DataFrame(
        {
            "energy": energies,
            "count": [10, 10],
            "a_sum": [20.0, 20.0],
            "a_sum2": [40.0, 40.0],
            "a_sum4": [160.0, 160.0],
            "b_sum": [50.0, 50.0],
            "b_sum2": [250.0, 250.0],
            "b_sum4": [6250.0, 6250.0],
        }
    )

    out = reweight_observables(mom, dos, T)

    assert np.allclose(out["a_mean"], 2.0)
    assert np.allclose(out["a_sq_mean"], 4.0)
    assert np.allclose(out["a_binder"], 2.0 / 3.0)
    assert np.allclose(out["b_mean"], 5.0)
    assert np.allclose(out["b_sq_mean"], 25.0)


def test_coverage_warning_for_unsampled_high_g_bin():
    """A finite-g bin with no observable samples that carries weight warns."""
    # Bin 1 has huge ln g but is never sampled; it dominates the partition
    # sum at 300 K, so coverage collapses.
    dos = pd.DataFrame({"energy": [0.0, 1.0], "entropy": [0.0, 100.0]})
    mom = _moments([0.0], [10], mean=1.0, sq_mean=1.0, quartic_mean=1.0)

    with pytest.warns(UserWarning, match="coverage"):
        out = reweight_observables(mom, dos, np.array([300.0]))
    assert float(out["coverage"].iloc[0]) < 0.99


def test_rejects_nonpositive_temperature():
    dos = pd.DataFrame({"energy": [0.0, 1.0], "entropy": [0.0, 0.0]})
    mom = _moments([0.0, 1.0], [10, 10], 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        reweight_observables(mom, dos, np.array([0.0, 300.0]))


def test_empty_inputs_raise():
    dos = pd.DataFrame({"energy": [0.0, 1.0], "entropy": [0.0, 0.0]})
    mom = _moments([0.0, 1.0], [10, 10], 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="moments has no rows"):
        reweight_observables(mom.iloc[0:0], dos, T)
    with pytest.raises(ValueError, match="dos has no rows"):
        reweight_observables(mom, dos.iloc[0:0], T)


def test_no_overlapping_bins_raises():
    """Moments bins disjoint from the finite-g DOS bins cannot be reweighted."""
    dos = pd.DataFrame({"energy": [0.0, 1.0], "entropy": [0.0, 0.0]})
    mom = _moments([5.0], [10], 1.0, 1.0, 1.0)  # bin 5 absent from the DOS
    with pytest.raises(ValueError, match="no overlapping"):
        reweight_observables(mom, dos, T)
