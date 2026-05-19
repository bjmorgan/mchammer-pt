"""Unit tests for CanonicalBuilder and WLBuilder."""

from __future__ import annotations

import pickle

import numpy as np

from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _atoms_dict_for(atoms):
    return {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": np.asarray(atoms.positions, dtype=np.float64),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64),
        "pbc": np.asarray(atoms.pbc, dtype=bool),
    }


def test_canonical_builder_build_returns_replica(tmp_path):
    """CanonicalBuilder.build() constructs a Replica with the configured fields."""
    from mchammer.ensembles import CanonicalEnsemble

    from mchammer_pt.parallel._builder import CanonicalBuilder
    from mchammer_pt.replica import Replica

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    builder = CanonicalBuilder(
        ce_path=str(ce_path),
        atoms_dict=_atoms_dict_for(atoms),
        temperature=300.0,
        seed=42,
        ensemble_cls=CanonicalEnsemble,
        ensemble_kwargs={},
    )
    replica = builder.build()
    assert isinstance(replica, Replica)
    assert replica.temperature == 300.0


def test_wl_builder_build_returns_wl_replica(tmp_path):
    """WLBuilder.build() constructs a WangLandauReplica with the configured window."""
    from mchammer_pt.parallel._builder import WLBuilder
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
        atoms_dict=_atoms_dict_for(atoms),
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={},
    )
    replica = builder.build()
    assert isinstance(replica, WangLandauReplica)
    assert replica.energy_window == (e0 - 100.0, e0 + 100.0)
    assert replica.energy_spacing == 0.1


def test_canonical_builder_picklable(tmp_path):
    """CanonicalBuilder round-trips through pickle (required for spawn)."""
    from mchammer.ensembles import CanonicalEnsemble

    from mchammer_pt.parallel._builder import CanonicalBuilder

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    builder = CanonicalBuilder(
        ce_path=str(ce_path),
        atoms_dict=_atoms_dict_for(atoms),
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
    for k in ("numbers", "positions", "cell", "pbc"):
        assert np.array_equal(
            restored.atoms_dict[k], builder.atoms_dict[k]
        )


def test_wl_builder_picklable(tmp_path):
    """WLBuilder round-trips through pickle (required for spawn)."""
    from mchammer_pt.parallel._builder import WLBuilder
    from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    ce_path = tmp_path / "ce.dat"
    ce.write(str(ce_path))

    builder = WLBuilder(
        ce_path=str(ce_path),
        atoms_dict=_atoms_dict_for(atoms),
        energy_spacing=0.1,
        energy_limit_left=-100.0,
        energy_limit_right=100.0,
        seed=42,
        ensemble_cls=CoordinatedWangLandauEnsemble,
        ensemble_kwargs={},
    )
    restored = pickle.loads(pickle.dumps(builder))
    assert restored.ce_path == builder.ce_path
    assert restored.energy_spacing == builder.energy_spacing
    assert restored.energy_limit_left == builder.energy_limit_left
    assert restored.energy_limit_right == builder.energy_limit_right
    assert restored.seed == builder.seed
    assert restored.ensemble_cls is builder.ensemble_cls
    assert restored.ensemble_kwargs == builder.ensemble_kwargs
    for k in ("numbers", "positions", "cell", "pbc"):
        assert np.array_equal(
            restored.atoms_dict[k], builder.atoms_dict[k]
        )
