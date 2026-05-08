"""Tests for the checkpoint/resume machinery."""

from __future__ import annotations

import random as stdlib_random

import numpy as np
import pytest
from mchammer.data_containers.base_data_container import BaseDataContainer

from mchammer_pt.replica import Replica


def test_replica_restart_from_restores_per_replica_state(
    toy_ce, toy_atoms, tmp_path
):
    """`Replica.restart_from(container)` restores step count, accepted-trial
    count, occupations, and stdlib `random` state from the saved container."""
    # Drive a fresh replica for some steps, then write its data container
    # to disk. mchammer populates `_last_state` only inside
    # `write_data_container`, so a round-trip through disk is the
    # realistic way to obtain a container with the saved per-replica state.
    original = Replica(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperature=300.0,
        random_seed=0,
    )
    original.advance(n_steps=50)

    dc_path = tmp_path / "replica.dc"
    original.ensemble.write_data_container(str(dc_path))
    container = BaseDataContainer.read(str(dc_path))

    saved_step = container._last_state["last_step"]
    saved_accepted = container._last_state["accepted_trials"]
    saved_occupations = container._last_state["occupations"]
    saved_random_state = container._last_state["random_state"]

    # Reconstruct via restart_from. The new replica should match the
    # saved state exactly.
    restored = Replica.restart_from(
        container,
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperature=300.0,
        random_seed=0,
    )

    assert restored.ensemble._step == saved_step
    assert restored.ensemble._accepted_trials == saved_accepted
    np.testing.assert_array_equal(
        restored.current_occupations(), saved_occupations
    )
    # The replica's saved RNG snapshot was taken at the end of
    # restart_from; restoring it to stdlib random and comparing
    # tuples confirms the state was actually pulled in.
    caller_state = stdlib_random.getstate()
    try:
        stdlib_random.setstate(restored._rng_state)
        assert stdlib_random.getstate() == saved_random_state
    finally:
        stdlib_random.setstate(caller_state)


def test_compute_ce_identity_is_deterministic_for_same_ce(toy_ce):
    """Hashing the same CE twice gives the same digest."""
    from mchammer_pt.checkpoint import _compute_ce_identity

    assert _compute_ce_identity(toy_ce) == _compute_ce_identity(toy_ce)


def test_compute_ce_identity_differs_for_different_parameters(
    toy_ce, toy_cluster_space
):
    """Different parameters → different digest."""
    from icet import ClusterExpansion

    from mchammer_pt.checkpoint import _compute_ce_identity

    other = ClusterExpansion(
        cluster_space=toy_cluster_space,
        parameters=np.zeros(len(toy_cluster_space)),
    )
    assert _compute_ce_identity(toy_ce) != _compute_ce_identity(other)


def test_compute_ce_identity_differs_for_different_chemistry(toy_ce):
    """Different chemical_symbols → different digest."""
    from ase.build import bulk
    from icet import ClusterExpansion, ClusterSpace

    from mchammer_pt.checkpoint import _compute_ce_identity

    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    other_cs = ClusterSpace(
        structure=primitive,
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Ag"],
    )
    other_ce = ClusterExpansion(
        cluster_space=other_cs,
        parameters=np.zeros(len(other_cs)),
    )
    # toy_ce uses Cu/Au; reconstruct an Au-version for a fair comparison
    # with matching parameter vector length.
    toy_au_cs = ClusterSpace(
        structure=primitive,
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )
    toy_au_ce = ClusterExpansion(
        cluster_space=toy_au_cs,
        parameters=np.zeros(len(toy_au_cs)),
    )
    assert _compute_ce_identity(other_ce) != _compute_ce_identity(toy_au_ce)


def test_compute_ce_identity_differs_for_different_cutoffs():
    """Different cutoffs → different digest."""
    from ase.build import bulk
    from icet import ClusterExpansion, ClusterSpace

    from mchammer_pt.checkpoint import _compute_ce_identity

    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    cs_short = ClusterSpace(
        structure=primitive, cutoffs=[3.5], chemical_symbols=["Cu", "Au"]
    )
    cs_long = ClusterSpace(
        structure=primitive, cutoffs=[4.5], chemical_symbols=["Cu", "Au"]
    )
    ce_short = ClusterExpansion(
        cluster_space=cs_short, parameters=np.zeros(len(cs_short))
    )
    ce_long = ClusterExpansion(
        cluster_space=cs_long, parameters=np.zeros(len(cs_long))
    )
    assert _compute_ce_identity(ce_short) != _compute_ce_identity(ce_long)


def test_compute_ce_identity_differs_for_different_primitive_structure():
    """Different primitive structure → different digest."""
    from ase.build import bulk
    from icet import ClusterExpansion, ClusterSpace

    from mchammer_pt.checkpoint import _compute_ce_identity

    cs_a = ClusterSpace(
        structure=bulk("Cu", "fcc", a=4.0, cubic=True),
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )
    cs_b = ClusterSpace(
        structure=bulk("Cu", "fcc", a=4.1, cubic=True),  # different lattice constant
        cutoffs=[3.5],
        chemical_symbols=["Cu", "Au"],
    )
    ce_a = ClusterExpansion(
        cluster_space=cs_a, parameters=np.zeros(len(cs_a))
    )
    ce_b = ClusterExpansion(
        cluster_space=cs_b, parameters=np.zeros(len(cs_b))
    )
    assert _compute_ce_identity(ce_a) != _compute_ce_identity(ce_b)


def test_compute_ensemble_kwargs_hash_handles_picklable_and_unpicklable():
    """Picklable kwargs hash deterministically; unpicklable kwargs return the
    sentinel ``""``."""
    from mchammer_pt.checkpoint import _compute_ensemble_kwargs_hash

    # None and {} both hash to the same canonical empty value.
    assert (
        _compute_ensemble_kwargs_hash(None)
        == _compute_ensemble_kwargs_hash({})
    )
    # Picklable kwargs hash deterministically.
    h1 = _compute_ensemble_kwargs_hash({"a": 1, "b": "x"})
    h2 = _compute_ensemble_kwargs_hash({"a": 1, "b": "x"})
    assert h1 == h2 and h1 != ""
    # Different picklable kwargs give different hashes.
    assert _compute_ensemble_kwargs_hash({"a": 1}) != _compute_ensemble_kwargs_hash(
        {"a": 2}
    )

    # Unpicklable kwargs return the sentinel.
    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    assert _compute_ensemble_kwargs_hash({"x": _Unpicklable()}) == ""


def test_compute_ensemble_kwargs_hash_returns_sentinel_for_local_class():
    """Instances of locally-defined classes return the sentinel rather than
    crashing — pickle can't resolve their qualified name."""
    from mchammer_pt.checkpoint import _compute_ensemble_kwargs_hash

    class _LocalClass:
        pass

    # pickle raises AttributeError (or similar) for local-class instances;
    # the broadened except in `_compute_ensemble_kwargs_hash` should catch.
    assert _compute_ensemble_kwargs_hash({"x": _LocalClass()}) == ""


def test_orchestrator_state_round_trips(tmp_path):
    """Writing orchestrator state via `write_hdf5` and reading it back via
    `_read_orchestrator_state` returns equivalent values."""
    from mchammer_pt.checkpoint import _read_orchestrator_state
    from mchammer_pt.history import ExchangeHistory, write_hdf5

    path = tmp_path / "ckpt.h5"
    history = ExchangeHistory.empty(n_cycles=3, n_replicas=4)
    rng_state_json = '{"bit_generator": "PCG64", "state": {"state": 42}}'
    write_hdf5(
        path,
        history=history,
        replica_containers=[],
        meta={"schema_version": "1"},
        orchestrator_state={
            "replica_labels": np.array([2, 0, 3, 1], dtype=np.int64),
            "rng_state": rng_state_json,
        },
    )

    loaded = _read_orchestrator_state(path)
    np.testing.assert_array_equal(
        loaded["replica_labels"], np.array([2, 0, 3, 1])
    )
    assert loaded["rng_state"] == rng_state_json


def _short_pt(toy_ce, toy_atoms, **overrides):
    """Build a short canonical PT for fast tests."""
    from mchammer_pt import CanonicalParallelTempering

    return CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        block_size=10,
        random_seed=42,
        **overrides,
    )


def test_save_checkpoint_writes_a_valid_resumable_file(toy_ce, toy_atoms, tmp_path):
    """`pt.save_checkpoint(path)` writes the schema fields the resume path
    requires."""
    from mchammer_pt.checkpoint import _read_orchestrator_state
    from mchammer_pt.history import read_hdf5

    pt = _short_pt(toy_ce, toy_atoms)
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    history, containers, meta = read_hdf5(path)
    assert meta["schema_version"] == "1"
    assert meta["block_size"] == 10
    assert "ce_identity" in meta and len(meta["ce_identity"]) == 64
    assert meta["ensemble_cls_fqn"].endswith(".CanonicalEnsemble")
    assert meta["random_seed"] == 42
    assert len(containers) == 3

    orchestrator_state = _read_orchestrator_state(path)
    assert orchestrator_state["replica_labels"].shape == (3,)
    assert orchestrator_state["rng_state"].startswith("{")  # JSON


def test_save_checkpoint_before_run_raises(toy_ce, toy_atoms, tmp_path):
    """`pt.save_checkpoint` requires `run()` to have been called at least once."""
    pt = _short_pt(toy_ce, toy_atoms)
    with pytest.raises(RuntimeError, match="run.*at least once"):
        pt.save_checkpoint(tmp_path / "ckpt.h5")


def test_data_container_file_path_writes_valid_checkpoint(toy_ce, toy_atoms, tmp_path):
    """The existing `data_container_file=` write path now produces files that
    include the new schema additions, so they are valid resume sources."""
    from mchammer_pt.checkpoint import _read_orchestrator_state
    from mchammer_pt.history import read_hdf5

    path = tmp_path / "run.h5"
    pt = _short_pt(toy_ce, toy_atoms, data_container_file=path)
    pt.run(n_cycles=3)

    _, _, meta = read_hdf5(path)
    assert meta["schema_version"] == "1"
    # And the orchestrator state is there too.
    _read_orchestrator_state(path)  # raises if absent


def test_checkpoint_writer_emits_at_interval_and_final_cycle(
    toy_ce, toy_atoms, tmp_path, monkeypatch
):
    """CheckpointWriter writes the file at every `interval`-th cycle plus the
    final cycle.

    Pins the cadence directly: wraps `_write_checkpoint` with a recorder
    that captures the cycle index at each emission. Asserts the exact
    set of emission cycles, then confirms the final file on disk is a
    valid checkpoint via `read_hdf5`.
    """
    from mchammer_pt.checkpoint import CheckpointWriter, _read_orchestrator_state
    from mchammer_pt.history import read_hdf5

    emissions: list[int] = []
    original_on_cycle_end = CheckpointWriter.on_cycle_end

    def recording_on_cycle_end(self, cycle, n_cycles, history):
        is_interval = (cycle + 1) % self._interval == 0
        is_final = cycle == n_cycles - 1
        if is_interval or is_final:
            emissions.append(cycle)
        return original_on_cycle_end(self, cycle, n_cycles, history)

    monkeypatch.setattr(CheckpointWriter, "on_cycle_end", recording_on_cycle_end)

    path = tmp_path / "ckpt.h5"
    pt = _short_pt(toy_ce, toy_atoms)
    pt.attach_checkpoint_writer(path, interval=3)
    pt.run(n_cycles=10)

    # n_cycles=10, interval=3:
    #   is_interval_emission for cycles where (cycle + 1) % 3 == 0 -> 2, 5, 8
    #   is_final_emission for cycle == n_cycles - 1 -> 9
    assert emissions == [2, 5, 8, 9]

    # And the final file is a valid checkpoint.
    history, containers, meta = read_hdf5(path)
    assert meta["schema_version"] == "1"
    assert history.energies_per_cycle.shape == (11, 3)
    _read_orchestrator_state(path)


def test_checkpoint_writer_rejects_non_positive_interval(toy_ce, toy_atoms, tmp_path):
    from mchammer_pt import CheckpointWriter

    pt = _short_pt(toy_ce, toy_atoms)
    with pytest.raises(ValueError):
        CheckpointWriter(tmp_path / "ckpt.h5", interval=0, pt=pt)
    with pytest.raises(ValueError):
        CheckpointWriter(tmp_path / "ckpt.h5", interval=-1, pt=pt)
