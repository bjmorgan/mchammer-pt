"""Unit tests for CanonicalBuilder, WLBuilder, and AtomsSpec."""

from __future__ import annotations

import pickle

import numpy as np

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def test_atoms_spec_round_trip_preserves_atoms():
    """AtomsSpec.from_atoms -> to_atoms preserves the four payload fields."""
    from mchammer_pt.parallel._builder import AtomsSpec

    atoms = make_wl_atoms()
    spec = AtomsSpec.from_atoms(atoms)
    restored = spec.to_atoms()
    assert np.array_equal(restored.numbers, atoms.numbers)
    assert np.array_equal(restored.positions, atoms.positions)
    assert np.array_equal(restored.cell.array, atoms.cell.array)
    assert np.array_equal(restored.pbc, atoms.pbc)


def test_atoms_spec_arrays_are_read_only():
    """from_atoms locks the arrays; mutating in-place raises ValueError."""
    import pytest

    from mchammer_pt.parallel._builder import AtomsSpec

    atoms = make_wl_atoms()
    spec = AtomsSpec.from_atoms(atoms)
    for name in ("numbers", "positions", "cell", "pbc"):
        arr = getattr(spec, name)
        with pytest.raises(ValueError, match="read-only|writeable"):
            arr[0] = arr[0]


def test_atoms_spec_does_not_alias_input_atoms():
    """from_atoms breaks aliasing: mutating the input atoms does not change the spec."""
    from mchammer_pt.parallel._builder import AtomsSpec

    atoms = make_wl_atoms()
    spec = AtomsSpec.from_atoms(atoms)
    original_numbers = spec.numbers.copy()
    # Mutate the input atoms; the spec must remain unchanged.
    atoms.numbers[0] = 1 if atoms.numbers[0] != 1 else 2
    assert np.array_equal(spec.numbers, original_numbers)


def test_canonical_builder_build_returns_replica(tmp_path):
    """CanonicalBuilder.build() constructs a Replica with the configured fields."""
    from mchammer.ensembles import CanonicalEnsemble

    from mchammer_pt.parallel._builder import AtomsSpec, CanonicalBuilder
    from mchammer_pt.replica import Replica

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    builder = CanonicalBuilder(
        ce_path=str(ce_path),
        atoms=AtomsSpec.from_atoms(atoms),
        temperature=300.0,
        seed=42,
        ensemble_cls=CanonicalEnsemble,
        ensemble_kwargs={},
    )
    replica = builder.build()
    assert isinstance(replica, Replica)
    assert replica.temperature == 300.0


def test_canonical_builder_propagates_seed(tmp_path):
    """Same seed via the builder yields the same RNG state on the replica."""
    from mchammer.ensembles import CanonicalEnsemble

    from mchammer_pt.parallel._builder import AtomsSpec, CanonicalBuilder

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    def _build(seed: int):
        builder = CanonicalBuilder(
            ce_path=str(ce_path),
            atoms=AtomsSpec.from_atoms(atoms),
            temperature=300.0,
            seed=seed,
            ensemble_cls=CanonicalEnsemble,
            ensemble_kwargs={},
        )
        return builder.build()

    r1 = _build(42)
    r2 = _build(42)
    r3 = _build(99)
    assert r1._rng_state == r2._rng_state
    assert r1._rng_state != r3._rng_state


def test_wl_builder_build_returns_wl_replica(tmp_path):
    """WLBuilder.build() constructs a WangLandauReplica with the configured window."""
    from mchammer_pt.parallel._builder import AtomsSpec, WLBuilder
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble
    from mchammer_pt.wl_replica import WangLandauReplica

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    from mchammer.calculators import ClusterExpansionCalculator
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )

    builder = WLBuilder(
        ce_path=str(ce_path),
        atoms=AtomsSpec.from_atoms(atoms),
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={},
        recency_visits_per_bin=1000,
    )
    replica = builder.build()
    assert isinstance(replica, WangLandauReplica)
    assert replica.energy_window == (e0 - 100.0, e0 + 100.0)
    assert replica.energy_spacing == 0.1


def test_canonical_builder_picklable(tmp_path):
    """CanonicalBuilder round-trips through pickle (required for spawn)."""
    from mchammer.ensembles import CanonicalEnsemble

    from mchammer_pt.parallel._builder import AtomsSpec, CanonicalBuilder

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    builder = CanonicalBuilder(
        ce_path=str(ce_path),
        atoms=AtomsSpec.from_atoms(atoms),
        temperature=300.0,
        seed=42,
        ensemble_cls=CanonicalEnsemble,
        ensemble_kwargs={},
    )
    restored = pickle.loads(pickle.dumps(builder))
    assert restored.ce_path == builder.ce_path
    assert restored.temperature == builder.temperature
    assert restored.seed == builder.seed
    assert restored.ensemble_cls is builder.ensemble_cls
    assert restored.ensemble_kwargs == builder.ensemble_kwargs
    assert np.array_equal(restored.atoms.numbers, builder.atoms.numbers)
    assert np.array_equal(restored.atoms.positions, builder.atoms.positions)
    assert np.array_equal(restored.atoms.cell, builder.atoms.cell)
    assert np.array_equal(restored.atoms.pbc, builder.atoms.pbc)


def test_wl_builder_picklable(tmp_path):
    """WLBuilder round-trips through pickle (required for spawn)."""
    from mchammer_pt.parallel._builder import AtomsSpec, WLBuilder
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    builder = WLBuilder(
        ce_path=str(ce_path),
        atoms=AtomsSpec.from_atoms(atoms),
        energy_spacing=0.1,
        energy_limit_left=-100.0,
        energy_limit_right=100.0,
        seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={},
        recency_visits_per_bin=1000,
    )
    restored = pickle.loads(pickle.dumps(builder))
    assert restored.ce_path == builder.ce_path
    assert restored.energy_spacing == builder.energy_spacing
    assert restored.energy_limit_left == builder.energy_limit_left
    assert restored.energy_limit_right == builder.energy_limit_right
    assert restored.seed == builder.seed
    assert restored.ensemble_cls is builder.ensemble_cls
    assert restored.ensemble_kwargs == builder.ensemble_kwargs
    assert np.array_equal(restored.atoms.numbers, builder.atoms.numbers)
    assert np.array_equal(restored.atoms.positions, builder.atoms.positions)
    assert np.array_equal(restored.atoms.cell, builder.atoms.cell)
    assert np.array_equal(restored.atoms.pbc, builder.atoms.pbc)
