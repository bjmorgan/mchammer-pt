"""Detailed-balance test: empirical Boltzmann distribution at fixed T.

Pins that a `CanonicalEnsemble` subclass passed via ``ensemble_cls=``
samples the analytic Boltzmann distribution of the bundled fixture
to within statistical tolerance, plus a discriminating-power check
against an ensemble that deliberately violates detailed balance.

The fixture and assertion logic live in `mchammer_pt.testing` as a
public utility so downstream packages can pin their own custom
`CanonicalEnsemble` subclasses against the same anchor without
reimplementing the analytic-Boltzmann math.
"""

from __future__ import annotations

import pytest
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from mchammer_pt.testing import assert_boltzmann_sampling


@pytest.mark.parametrize("ensemble_cls", [CanonicalEnsemble])
def test_ensemble_samples_correct_boltzmann_distribution(
    ensemble_cls: type[CanonicalEnsemble],
) -> None:
    """Empirical class population matches analytic Boltzmann to 4σ."""
    assert_boltzmann_sampling(ensemble_cls)


def test_detailed_balance_test_detects_broken_acceptance() -> None:
    """Sanity: the test fails when the ensemble accepts every move.

    `HighAcceptanceCanonicalEnsemble` overrides `_acceptance_condition`
    to always return True. On this fixture the stationary distribution
    under always-accept is uniform over the six microstates (4:2 by
    class multiplicity, because every state has the same number of
    (Cu, Au) site-pairs and the proposal kernel is therefore doubly
    stochastic — this is a property of the symmetric 2-Cu/2-Au
    fixture, not of always-accept in general). The analytic Boltzmann
    distribution at T = 1000 K with ΔE ≈ 3 kT is far from uniform; the
    detailed-balance check must detect the deviation.
    """
    from tests._ensemble_fixtures import HighAcceptanceCanonicalEnsemble

    with pytest.raises(AssertionError, match="empirical"):
        assert_boltzmann_sampling(HighAcceptanceCanonicalEnsemble)
