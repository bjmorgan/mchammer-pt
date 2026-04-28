"""Public testing utilities for verifying `CanonicalEnsemble` subclasses.

Downstream packages providing custom `CanonicalEnsemble` subclasses
(custom Monte Carlo moves, alternative acceptance criteria, etc.) can
use `assert_boltzmann_sampling` to anchor their detailed-balance
correctness against the same analytic Boltzmann fixture mchammer-pt's
own test suite uses.

The fixture is a four-site one-dimensional chain (orthorhombic cell
with periodic length 4 along x, isolated along y and z) with two Cu
and two Au and an NN-only pair ECI. The six microstates of (2 Cu,
2 Au) split into four "clustered" (1 CuCu + 2 CuAu + 1 AuAu NN
bonds) and two "alternating" (0 CuCu + 4 CuAu + 0 AuAu)
configurations, giving two distinct CE energies. The pair ECI is
calibrated so the energy gap is approximately 3 kT at
`FIXTURE_TEMPERATURE`, producing analytic class populations of
roughly 0.98 (clustered) and 0.02 (alternating). With 10 000 samples
at the default 4σ binomial tolerance, the test discriminates the
correct distribution from any kernel that perturbs class populations
by more than approximately 0.6%.

For ensembles whose construction depends on fixture geometry
(e.g. a custom move taking chain definitions), the
`FIXTURE_CHAIN_INDICES` constant exposes the four-site chain in
geometric order.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from .replica import Replica

FIXTURE_N_SITES: int = 4
"""Number of sites in the bundled detailed-balance fixture."""

FIXTURE_CHAIN_INDICES: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3),)
"""Site indices of the chains in the bundled fixture, in geometric order.

The fixture is a single one-dimensional chain of four sites; downstream
custom moves that take chain definitions can use this constant to
construct fixture-aware kwargs. For example::

    from mchammer_pt.testing import (
        assert_boltzmann_sampling,
        FIXTURE_CHAIN_INDICES,
    )

    assert_boltzmann_sampling(
        MyChainEnsemble,
        ensemble_kwargs={"chains": [list(c) for c in FIXTURE_CHAIN_INDICES]},
    )
"""

FIXTURE_TEMPERATURE: float = 1000.0
"""Temperature (K) at which the fixture is sampled and analytic-checked."""

_KB_EV_PER_K = 8.617333262145e-5
_TARGET_GAP_KT = 3.0


def _build_chain_ce_and_atoms() -> tuple[ClusterExpansion, Atoms]:
    """Build a four-site 1D-chain CE calibrated to ΔE ≈ 3 kT at fixture T.

    Cell length is 4 Å along x with isolated y, z. NN cutoff 1.5 Å
    captures only x-direction nearest neighbours; the second-NN
    distance along x is 2.0 Å, outside the cutoff.
    """
    primitive = Atoms(
        "Cu",
        positions=[(0.0, 0.0, 0.0)],
        cell=[(1.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)],
        pbc=True,
    )
    cs = ClusterSpace(
        structure=primitive,
        cutoffs=[1.5],
        chemical_symbols=["Cu", "Au"],
    )
    parameters = np.zeros(len(cs), dtype=float)
    # Calibrate the pair ECI so the gap ΔE = E_alt - E_clust is ~3 kT
    # at FIXTURE_TEMPERATURE. Energy is linear in the pair ECI so we
    # compute the gap once with a placeholder value, then scale.
    parameters[-1] = 1.0
    ce_probe = ClusterExpansion(cluster_space=cs, parameters=parameters)
    atoms: Atoms = primitive.repeat((FIXTURE_N_SITES, 1, 1))  # type: ignore[no-untyped-call]
    atoms.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])  # type: ignore[no-untyped-call]
    e_clust_unit = float(ce_probe.predict(atoms)) * len(atoms)
    atoms.set_chemical_symbols(["Cu", "Au", "Cu", "Au"])  # type: ignore[no-untyped-call]
    e_alt_unit = float(ce_probe.predict(atoms)) * len(atoms)
    gap_unit = e_alt_unit - e_clust_unit
    if abs(gap_unit) < 1e-12:
        raise RuntimeError(
            "ECI probe produced a degenerate energy gap; check ClusterSpace"
        )
    target_gap = _TARGET_GAP_KT * _KB_EV_PER_K * FIXTURE_TEMPERATURE
    parameters[-1] = target_gap / gap_unit
    ce = ClusterExpansion(cluster_space=cs, parameters=parameters)
    atoms.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])  # type: ignore[no-untyped-call]
    return ce, atoms


def _enumerate_two_cu_microstates() -> list[list[str]]:
    """All six (2 Cu, 2 Au) symbol assignments on the 4 fixture sites."""
    configs: list[list[str]] = []
    for cu_indices in itertools.combinations(range(FIXTURE_N_SITES), 2):
        symbols = ["Au"] * FIXTURE_N_SITES
        for i in cu_indices:
            symbols[i] = "Cu"
        configs.append(symbols)
    return configs


def _classify_by_energy(
    ce: ClusterExpansion,
    atoms: Atoms,
    configs: list[list[str]],
) -> dict[float, int]:
    """Group microstates by total CE energy; return {energy: multiplicity}.

    Operates on a copy of `atoms` so the caller's symbol assignment
    is preserved.
    """
    work: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
    multiplicities: dict[float, int] = {}
    for symbols in configs:
        work.set_chemical_symbols(symbols)  # type: ignore[no-untyped-call]
        e_total = float(ce.predict(work)) * len(work)
        for e_existing in list(multiplicities):
            if abs(e_total - e_existing) < 1e-9:
                multiplicities[e_existing] += 1
                break
        else:
            multiplicities[e_total] = 1
    return multiplicities


def _analytic_class_probabilities(
    multiplicities: dict[float, int], temperature: float
) -> dict[float, float]:
    """Boltzmann probability per energy class, P(E) ∝ g(E) exp(-βE)."""
    beta = 1.0 / (_KB_EV_PER_K * temperature)
    weights = {e: g * np.exp(-e * beta) for e, g in multiplicities.items()}
    Z = sum(weights.values())
    return {e: float(w / Z) for e, w in weights.items()}


def assert_boltzmann_sampling(
    ensemble_cls: type[CanonicalEnsemble],
    ensemble_kwargs: Mapping[str, Any] | None = None,
    *,
    n_samples: int = 10_000,
    sample_interval: int = 50,
    burn_in: int = 5_000,
    seed: int = 0,
    sigma_tolerance: float = 4.0,
) -> None:
    """Assert that `ensemble_cls` samples the bundled Boltzmann fixture.

    Fixture: a four-site one-dimensional chain (orthorhombic cell with
    periodic length 4 along x, isolated along y and z) with two Cu and
    two Au and an NN-only pair ECI. The six microstates of (2 Cu,
    2 Au) split into four "clustered" (1 CuCu + 2 CuAu + 1 AuAu NN
    bonds) and two "alternating" (0 CuCu + 4 CuAu + 0 AuAu)
    configurations, giving two distinct CE energies. The pair ECI is
    calibrated so the energy gap is approximately 3 kT at
    `FIXTURE_TEMPERATURE`, producing analytic class populations of
    roughly 0.98 (clustered) and 0.02 (alternating) — far enough from
    a uniform 4:2 stationary distribution that broken acceptance is
    detected at default tolerance.

    The function constructs a `Replica` with `ensemble_cls` and
    `ensemble_kwargs`, advances it for `burn_in` trial steps, then
    collects `n_samples` samples at `sample_interval` step intervals.
    Compares empirical class proportions to analytic Boltzmann
    probabilities; raises AssertionError if any class deviates by
    more than `sigma_tolerance` standard errors of the binomial.

    Args:
        ensemble_cls: A `CanonicalEnsemble` or subclass.
        ensemble_kwargs: Extra keyword arguments forwarded to
            ``ensemble_cls(...)``. Cannot include `structure`,
            `calculator`, `temperature`, or `random_seed` (set by
            `Replica`). For ensembles whose construction depends on
            fixture geometry, use the `FIXTURE_CHAIN_INDICES` constant.
        n_samples: Number of samples to collect after burn-in.
        sample_interval: MC trial steps between samples.
        burn_in: MC trial steps to advance before sampling.
        seed: Random seed for the replica's RNG stream.
        sigma_tolerance: Maximum binomial-σ deviation per class
            before the assertion fires.

    Raises:
        AssertionError: If any class's empirical population deviates
            from the analytic Boltzmann probability by more than
            `sigma_tolerance × σ`, or if the fixture itself produces
            an unexpected number of energy classes.
    """
    ce, atoms = _build_chain_ce_and_atoms()
    multiplicities = _classify_by_energy(ce, atoms, _enumerate_two_cu_microstates())
    assert len(multiplicities) == 2, (
        f"Expected exactly 2 distinct energy classes for the 4-site "
        f"chain with NN-only pair ECI; got {len(multiplicities)}"
    )
    p_analytic = _analytic_class_probabilities(multiplicities, FIXTURE_TEMPERATURE)
    class_energies = sorted(multiplicities.keys())

    rep = Replica(
        cluster_expansion=ce,
        atoms=atoms,
        temperature=FIXTURE_TEMPERATURE,
        random_seed=seed,
        ensemble_cls=ensemble_cls,
        ensemble_kwargs=ensemble_kwargs,
    )
    rep.advance(n_steps=burn_in)
    counts = dict.fromkeys(class_energies, 0)
    for _ in range(n_samples):
        rep.advance(n_steps=sample_interval)
        e = rep.current_energy()
        for e_class in class_energies:
            if abs(e - e_class) < 1e-6:
                counts[e_class] += 1
                break
        else:
            raise AssertionError(
                f"Sample energy {e!r} matches no enumerated class "
                f"{class_energies!r}"
            )

    for e_class, count in counts.items():
        p_emp = count / n_samples
        p_an = p_analytic[e_class]
        sigma = np.sqrt(p_an * (1.0 - p_an) / n_samples)
        delta = abs(p_emp - p_an)
        assert delta < sigma_tolerance * sigma, (
            f"Class E={e_class:.4f} eV (multiplicity "
            f"{multiplicities[e_class]}): empirical {p_emp:.4f} vs "
            f"analytic {p_an:.4f}, |Δ|={delta:.4f} > "
            f"{sigma_tolerance}σ={sigma_tolerance * sigma:.4f}"
        )
