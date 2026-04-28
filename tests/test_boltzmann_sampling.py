"""Boltzmann-sampling test: empirical distribution at fixed T.

Pins that a `CanonicalEnsemble` subclass passed via ``ensemble_cls=``
samples the analytic Boltzmann distribution of the bundled fixture to
within statistical tolerance, plus a discriminating-power check
against an ensemble that is stationary at the wrong distribution
(`HighAcceptanceCanonicalEnsemble`).

The fixture and assertion logic live in `mchammer_pt.testing` as a
public utility so downstream packages providing custom
`CanonicalEnsemble` subclasses can pin against the same anchor
without reimplementing the analytic-Boltzmann math.
"""

from __future__ import annotations

from typing import Any

import pytest
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from mchammer_pt.testing import FIXTURE_CHAIN_INDICES, assert_boltzmann_sampling


def test_fixture_chain_indices_pin() -> None:
    """Pin the public fixture geometry constant.

    The constant has no internal consumer in this module, so silent
    drift would not fail any other test. This pin makes any change
    explicit.
    """
    assert FIXTURE_CHAIN_INDICES == ((0, 1, 2, 3),)
    # Cross-consistency: the chain spans contiguous site indices in
    # geometric order, no overlaps, no gaps.
    seen: set[int] = set()
    for chain in FIXTURE_CHAIN_INDICES:
        for site in chain:
            assert site not in seen, "FIXTURE_CHAIN_INDICES must not overlap"
            seen.add(site)
    assert seen == set(range(len(seen)))


@pytest.mark.parametrize("ensemble_cls", [CanonicalEnsemble])
def test_ensemble_samples_correct_boltzmann_distribution(
    ensemble_cls: type[CanonicalEnsemble],
) -> None:
    """Empirical class population matches analytic Boltzmann to 4σ."""
    assert_boltzmann_sampling(ensemble_cls)


def test_boltzmann_check_detects_broken_acceptance() -> None:
    """Sanity: the test fails when the ensemble accepts every move.

    `HighAcceptanceCanonicalEnsemble` overrides `_acceptance_condition`
    to always return True. Its stationary distribution under the
    canonical-swap proposal kernel on this 2-Cu/2-Au fixture is
    uniform over the six microstates (4:2 by class multiplicity),
    because the proposal kernel is doubly stochastic — every state
    has the same number of (Cu, Au) site-pairs. Detailed balance
    therefore holds against the wrong stationary distribution; what
    fails is the *match to the target Boltzmann distribution*. The
    analytic Boltzmann distribution at T = 1000 K with ΔE ≈ 3 kT is
    far from uniform; the Boltzmann-sampling assertion must detect
    the deviation.
    """
    from tests._ensemble_fixtures import HighAcceptanceCanonicalEnsemble

    with pytest.raises(AssertionError, match="empirical"):
        assert_boltzmann_sampling(HighAcceptanceCanonicalEnsemble)


def test_assert_boltzmann_sampling_forwards_ensemble_kwargs() -> None:
    """ensemble_kwargs reach the constructed ensemble.

    Pins the kwarg pass-through that `FIXTURE_CHAIN_INDICES` consumers
    rely on. A regression that dropped the kwarg or conflated
    None-vs-{} would silently leave downstream chain-aware moves
    constructed without their chain definitions; this test surfaces
    such a regression as a missing sentinel.
    """
    captured: list[str] = []

    class _StopAfterCapture(Exception):
        pass

    class _KwargRecorder(CanonicalEnsemble):
        def __init__(self, *, sentinel: str, **kwargs: Any) -> None:
            captured.append(sentinel)
            raise _StopAfterCapture()

    with pytest.raises(_StopAfterCapture):
        assert_boltzmann_sampling(
            _KwargRecorder,
            ensemble_kwargs={"sentinel": "abc123"},
        )
    assert captured == ["abc123"]
