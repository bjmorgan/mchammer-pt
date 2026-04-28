"""Boltzmann-sampling test: empirical distribution at fixed T.

Pins that a `CanonicalEnsemble` subclass passed via ``ensemble_cls=``
samples the analytic Boltzmann distribution at a chosen temperature,
to within statistical tolerance.

The test system is a four-site one-dimensional chain (orthorhombic
cell with periodic length 4 along x, isolated along y and z) with
two Cu and two Au and an NN-only pair ECI. The six microstates of
(2 Cu, 2 Au) split into four "clustered" (1 CuCu + 2 CuAu + 1 AuAu
NN bonds) and two "alternating" (0 CuCu + 4 CuAu + 0 AuAu)
configurations, giving two distinct CE energies. The pair ECI is
calibrated so the energy gap is approximately 3 kT at T = 1000 K,
producing analytic class populations of roughly 0.98 (clustered)
and 0.02 (alternating) — far from the 4:2 uniform ratio a broken
acceptance criterion would produce on this fixture.

Parametrise the main test over ``ensemble_cls`` so future custom
moves (e.g. row-translation subclasses) join the list as a one-line
addition. The companion test confirms the check discriminates: a
deliberately broken ensemble (`HighAcceptanceCanonicalEnsemble`,
which accepts every move regardless of energy) must fail the same
assertion.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ase import Atoms
from icet import ClusterExpansion, ClusterSpace  # type: ignore[import-untyped]
from mchammer.ensembles import CanonicalEnsemble  # type: ignore[import-untyped]

from mchammer_pt.replica import Replica

_T_KELVIN = 1000.0
_KB_EV_PER_K = 8.617333262145e-5
_TARGET_GAP_KT = 3.0


def _build_chain_ce_and_atoms() -> tuple[ClusterExpansion, Atoms]:
    """Build a four-site 1D-chain CE calibrated to ΔE ≈ 3 kT at T_KELVIN.

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
    # cs has zerolet, singlet, NN-pair (3 ECIs by default).
    parameters = np.zeros(len(cs), dtype=float)
    # Calibrate the pair ECI so the energy gap ΔE = E_alt - E_clust is
    # ~3 kT at T_KELVIN. Energy is linear in the pair ECI so we compute
    # the gap once with a placeholder value, then scale.
    parameters[-1] = 1.0
    ce_probe = ClusterExpansion(cluster_space=cs, parameters=parameters)
    atoms = primitive.repeat((4, 1, 1))
    atoms.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])
    e_clust_unit = float(ce_probe.predict(atoms)) * len(atoms)
    atoms.set_chemical_symbols(["Cu", "Au", "Cu", "Au"])
    e_alt_unit = float(ce_probe.predict(atoms)) * len(atoms)
    gap_unit = e_alt_unit - e_clust_unit
    if abs(gap_unit) < 1e-12:
        raise RuntimeError(
            "ECI probe produced a degenerate energy gap; check ClusterSpace"
        )
    target_gap = _TARGET_GAP_KT * _KB_EV_PER_K * _T_KELVIN
    parameters[-1] = target_gap / gap_unit
    ce = ClusterExpansion(cluster_space=cs, parameters=parameters)
    # Reset atoms to a canonical clustered configuration.
    atoms.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])
    return ce, atoms


def _enumerate_two_cu_microstates() -> list[list[str]]:
    """All six (2 Cu, 2 Au) symbol assignments on the 4 sites."""
    configs: list[list[str]] = []
    for cu_indices in itertools.combinations(range(4), 2):
        symbols = ["Au"] * 4
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
    work = atoms.copy()
    multiplicities: dict[float, int] = {}
    for symbols in configs:
        work.set_chemical_symbols(symbols)
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


def _run_and_assert_boltzmann(
    ensemble_cls: type[CanonicalEnsemble],
    *,
    n_samples: int = 10_000,
    sample_interval: int = 50,
    burn_in: int = 5_000,
    seed: int = 0,
    sigma_tolerance: float = 4.0,
) -> None:
    """Run a Replica trajectory and assert empirical ≈ analytic Boltzmann.

    Raises `AssertionError` if any class population deviates from the
    analytic Boltzmann probability by more than ``sigma_tolerance``
    standard errors of the binomial.
    """
    ce, atoms = _build_chain_ce_and_atoms()
    multiplicities = _classify_by_energy(ce, atoms, _enumerate_two_cu_microstates())
    assert len(multiplicities) == 2, (
        f"Expected exactly 2 distinct energy classes for the 4-site "
        f"chain with NN-only pair ECI; got {len(multiplicities)}"
    )
    p_analytic = _analytic_class_probabilities(multiplicities, _T_KELVIN)
    class_energies = sorted(multiplicities.keys())

    rep = Replica(
        cluster_expansion=ce,
        atoms=atoms,
        temperature=_T_KELVIN,
        random_seed=seed,
        ensemble_cls=ensemble_cls,
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


@pytest.mark.parametrize("ensemble_cls", [CanonicalEnsemble])
def test_ensemble_samples_correct_boltzmann_distribution(
    ensemble_cls: type[CanonicalEnsemble],
) -> None:
    """Empirical class population matches analytic Boltzmann to 4σ."""
    _run_and_assert_boltzmann(ensemble_cls)


def test_boltzmann_check_detects_broken_acceptance() -> None:
    """Sanity: the test fails when the ensemble accepts every move.

    `HighAcceptanceCanonicalEnsemble` overrides `_acceptance_condition`
    to always return True. On this fixture the stationary distribution
    under always-accept is uniform over the six microstates (4:2 by
    class multiplicity, because every state has the same number of
    (Cu, Au) site-pairs and the proposal kernel is therefore doubly
    stochastic — this is a property of the symmetric 2-Cu/2-Au
    fixture, not of always-accept in general). The analytic Boltzmann
    distribution at T = 1000 K with ΔE ≈ 3 kT is far from uniform; the
    Boltzmann-sampling assertion must detect the deviation.
    """
    from tests._ensemble_fixtures import HighAcceptanceCanonicalEnsemble

    with pytest.raises(AssertionError, match="empirical"):
        _run_and_assert_boltzmann(HighAcceptanceCanonicalEnsemble)
