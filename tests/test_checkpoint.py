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
        meta={"schema_version": "3"},
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
    assert meta["schema_version"] == "5"
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
    from mchammer_pt import CanonicalParallelTempering
    from mchammer_pt.checkpoint import _read_orchestrator_state
    from mchammer_pt.history import read_hdf5

    path = tmp_path / "run.h5"
    pt = _short_pt(toy_ce, toy_atoms, data_container_file=path)
    pt.run(n_cycles=3)

    _, _, meta = read_hdf5(path)
    assert meta["schema_version"] == "5"
    # And the orchestrator state is there too.
    _read_orchestrator_state(path)  # raises if absent

    # The README promises files written via `data_container_file=` are
    # valid resume sources. Pin that end-to-end so a regression where
    # the write path drifts away from `_write_checkpoint` would surface
    # here rather than at a user's next walltime kill.
    pt_b = CanonicalParallelTempering.resume(path, cluster_expansion=toy_ce)
    pt_b.run(n_cycles=1)


def test_checkpoint_writer_emits_at_interval_and_final_cycle(
    toy_ce, toy_atoms, tmp_path, monkeypatch
):
    """CheckpointWriter writes the file at every `interval`-th cycle plus the
    final cycle.

    Pins the cadence by observing production: spies on `_write_checkpoint`
    (the function `CheckpointWriter.on_cycle_end` calls when it decides to
    emit) and records the cycle index at each call. A regression that broke
    the production modulus would invoke `_write_checkpoint` on different
    cycles, flipping this assertion red.
    """
    from mchammer_pt import checkpoint as checkpoint_module
    from mchammer_pt.checkpoint import CheckpointWriter, _read_orchestrator_state
    from mchammer_pt.history import read_hdf5

    # Spy on the actual write function. The spy records that *some* write
    # happened; the wrapping on_cycle_end below records *which cycle* the
    # write was for. Together they observe production's decision rather
    # than re-implementing it.
    write_calls: list[None] = []

    def spy_write_checkpoint(pt_arg, path_arg):
        # Run the real write so the final-file assertion still passes.
        original_write_checkpoint(pt_arg, path_arg)
        write_calls.append(None)

    original_write_checkpoint = checkpoint_module._write_checkpoint
    monkeypatch.setattr(
        checkpoint_module, "_write_checkpoint", spy_write_checkpoint
    )

    # Wrap on_cycle_end to record the cycle when (and only when) production
    # actually triggered a write — i.e. the spy was invoked during the
    # original call.
    emissions: list[int] = []
    original_on_cycle_end = CheckpointWriter.on_cycle_end

    def recording_on_cycle_end(self, cycle, n_cycles, history):
        before = len(write_calls)
        original_on_cycle_end(self, cycle, n_cycles, history)
        if len(write_calls) > before:
            emissions.append(cycle)

    monkeypatch.setattr(CheckpointWriter, "on_cycle_end", recording_on_cycle_end)

    path = tmp_path / "ckpt.h5"
    pt = _short_pt(toy_ce, toy_atoms)
    pt.attach_checkpoint_writer(path, interval=3)
    pt.run(n_cycles=10)

    # n_cycles=10, interval=3 — production's emission decision is
    # `(cycle + 1) % interval == 0` OR `cycle == n_cycles - 1`:
    #   modulus fires at cycles 2, 5, 8
    #   final-cycle guard fires at cycle 9
    # If production's modulus is wrong, this list is wrong.
    assert emissions == [2, 5, 8, 9]

    # And the final file is a valid resumable checkpoint.
    history, _, meta = read_hdf5(path)
    assert meta["schema_version"] == "5"
    assert history.energies_per_cycle.shape == (11, 3)
    _read_orchestrator_state(path)


def test_checkpoint_writer_rejects_non_positive_interval(toy_ce, toy_atoms, tmp_path):
    from mchammer_pt import CheckpointWriter

    pt = _short_pt(toy_ce, toy_atoms)
    with pytest.raises(ValueError):
        CheckpointWriter(tmp_path / "ckpt.h5", interval=0, pt=pt)
    with pytest.raises(ValueError):
        CheckpointWriter(tmp_path / "ckpt.h5", interval=-1, pt=pt)


def test_resume_is_bit_identical(toy_ce, toy_atoms, tmp_path):
    """A:N + B:M, concatenated, equals a single-run-of-(N+M)."""
    from mchammer_pt import CanonicalParallelTempering
    from mchammer_pt.history import ExchangeHistory

    pt_full = _short_pt(toy_ce, toy_atoms)
    history_full = pt_full.run(n_cycles=20)

    pt_a = _short_pt(toy_ce, toy_atoms)
    history_a = pt_a.run(n_cycles=10)
    path = tmp_path / "ckpt.h5"
    pt_a.save_checkpoint(path)

    pt_b = CanonicalParallelTempering.resume(
        path, cluster_expansion=toy_ce
    )
    history_b = pt_b.run(n_cycles=10)

    combined = ExchangeHistory.concatenate(history_a, history_b)
    np.testing.assert_array_equal(
        combined.energies_per_cycle, history_full.energies_per_cycle
    )
    np.testing.assert_array_equal(
        combined.replica_labels_per_cycle,
        history_full.replica_labels_per_cycle,
    )
    np.testing.assert_array_equal(
        combined.swap_attempted, history_full.swap_attempted
    )
    np.testing.assert_array_equal(
        combined.swap_accepted, history_full.swap_accepted
    )


def test_resume_rejects_unknown_schema_version(toy_ce, toy_atoms, tmp_path):
    """A checkpoint with an unrecognised schema_version raises ValueError."""
    import h5py

    from mchammer_pt import CanonicalParallelTempering

    pt = _short_pt(toy_ce, toy_atoms)
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with h5py.File(path, "r+") as f:
        f["meta"].attrs["schema_version"] = "999"

    with pytest.raises(ValueError, match="0.9.0"):
        CanonicalParallelTempering.resume(path, cluster_expansion=toy_ce)


def test_resume_rejects_v3_schema(toy_ce, toy_atoms, tmp_path):
    """v4 readers refuse v3 files with a message pointing at 0.9.0."""
    import h5py

    from mchammer_pt import CanonicalParallelTempering

    pt = _short_pt(toy_ce, toy_atoms)
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with h5py.File(path, "r+") as f:
        f["meta"].attrs["schema_version"] = "3"

    with pytest.raises(ValueError, match="0.9.0"):
        CanonicalParallelTempering.resume(path, cluster_expansion=toy_ce)


def test_data_container_file_works_with_process_pool(toy_ce, toy_atoms, tmp_path):
    """`process_pool(... data_container_file=...).run(...)` writes a valid
    checkpoint. Pins the symmetry guarantee that both pool kinds produce
    files via the same cross-pool snapshot machinery — a regression to
    the prior `_pool.replicas` hard-coding would break ProcessPool here."""
    from mchammer_pt import CanonicalParallelTempering
    from mchammer_pt.checkpoint import _read_orchestrator_state
    from mchammer_pt.history import read_hdf5

    path = tmp_path / "run.h5"
    with CanonicalParallelTempering.process_pool(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        block_size=10,
        random_seed=42,
        data_container_file=path,
    ) as pt:
        pt.run(n_cycles=3)

    history, containers, meta = read_hdf5(path)
    assert meta["schema_version"] == "5"
    assert len(containers) == 3
    _read_orchestrator_state(path)


def test_resume_process_pool_round_trips(toy_ce, toy_atoms, tmp_path):
    """Smoke test: a process-pool run can be checkpointed, resumed under
    `resume_process_pool`, and continued. No bit-identical claim across
    pools — worker scheduling makes that flake — only that resume works
    end-to-end."""
    from mchammer_pt import CanonicalParallelTempering

    path = tmp_path / "ckpt.h5"
    with CanonicalParallelTempering.process_pool(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        block_size=10,
        random_seed=42,
    ) as pt_a:
        pt_a.run(n_cycles=5)
        pt_a.save_checkpoint(path)

    with CanonicalParallelTempering.resume_process_pool(
        path, cluster_expansion=toy_ce
    ) as pt_b:
        history_b = pt_b.run(n_cycles=5)

    assert history_b.energies_per_cycle.shape == (6, 3)
    assert np.all(np.isfinite(history_b.energies_per_cycle))


def test_resume_rejects_mismatched_ce(toy_ce, toy_atoms, toy_cluster_space, tmp_path):
    """Resuming with a different CE raises with a clear message."""
    from icet import ClusterExpansion

    from mchammer_pt import CanonicalParallelTempering

    pt = _short_pt(toy_ce, toy_atoms)
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    other = ClusterExpansion(
        cluster_space=toy_cluster_space,
        parameters=np.zeros(len(toy_cluster_space)),
    )
    with pytest.raises(ValueError, match="CE identity"):
        CanonicalParallelTempering.resume(path, cluster_expansion=other)


def test_resume_rejects_mismatched_ensemble_cls(toy_ce, toy_atoms, tmp_path):
    """Resuming with a different `ensemble_cls` raises with a clear message."""
    from mchammer.ensembles import CanonicalEnsemble

    from mchammer_pt import CanonicalParallelTempering

    pt = _short_pt(toy_ce, toy_atoms)
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    # Subclass with a different fully-qualified name; the cls itself is
    # picklable and the spawn-import guard accepts it, so the only thing
    # the FQN guard catches is the identity mismatch.
    class _OtherEnsemble(CanonicalEnsemble):
        pass

    with pytest.raises(ValueError, match="ensemble_cls FQN mismatch"):
        CanonicalParallelTempering.resume(
            path,
            cluster_expansion=toy_ce,
            ensemble_cls=_OtherEnsemble,
        )


def test_validate_kwargs_hash_warns_on_sentinel():
    """When either side returned the unpicklable-kwargs sentinel ``""``, the
    resume-time guard emits `UserWarning` rather than silently passing.

    Pins the silent-to-loud transition the previous review fix introduced.
    A regression that removed the warning, downgraded it to a less-visible
    category, or gated it on a branch that doesn't fire would restore the
    original silent-physics-divergence failure mode."""
    from mchammer_pt.checkpoint import (
        _compute_ensemble_kwargs_hash,
        _validate_kwargs_hash,
    )

    # Saved-side sentinel (original run had unpicklable kwargs); user
    # supplies kwargs that hash cleanly. Guard cannot enforce identity.
    with pytest.warns(UserWarning, match="kwargs-identity guard"):
        _validate_kwargs_hash(
            "ckpt.h5",
            {"ensemble_kwargs_hash": ""},
            ensemble_kwargs={"user_tag": "first"},
            caller="resume",
        )

    # Supplied-side sentinel (user's kwargs are not stably hashable);
    # checkpoint's hash is real. Same skip → same warning.
    class _Local:
        pass

    saved_hash = _compute_ensemble_kwargs_hash({"user_tag": "first"})
    with pytest.warns(UserWarning, match="kwargs-identity guard"):
        _validate_kwargs_hash(
            "ckpt.h5",
            {"ensemble_kwargs_hash": saved_hash},
            ensemble_kwargs={"thing": _Local()},
            caller="resume",
        )


def test_resume_rejects_mismatched_ensemble_kwargs_hash(toy_ce, toy_atoms, tmp_path):
    """Resuming with materially different `ensemble_kwargs` raises with a clear
    message when both sides hash cleanly."""
    from mchammer_pt import CanonicalParallelTempering

    pt = _short_pt(toy_ce, toy_atoms, ensemble_kwargs={"user_tag": "first"})
    pt.run(n_cycles=3)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with pytest.raises(ValueError, match="ensemble_kwargs hash mismatch"):
        CanonicalParallelTempering.resume(
            path,
            cluster_expansion=toy_ce,
            ensemble_kwargs={"user_tag": "second"},
        )


def test_read_window_groups_returns_one_entry_per_window(tmp_path):
    """Returns list[dict|None] of length N; None for windows whose
    subgroup is absent (W=1 by convention)."""
    import h5py

    from mchammer_pt.checkpoint import _read_window_groups

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["walkers_per_window"] = np.array([1, 2, 1], dtype=np.int32)
        wg = f.create_group("orchestrator/window_groups")
        sub = wg.create_group("1")
        sub.create_dataset("rng_state", data='{"bit_generator": "PCG64"}')
        sub.create_dataset("phase", data="1_over_t")

    out = _read_window_groups(path)
    assert out[0] is None
    assert out[1] == {
        "rng_state": '{"bit_generator": "PCG64"}',
        "phase": "1_over_t",
    }
    assert out[2] is None


def test_write_checkpoint_passes_window_groups_through_to_hdf5(tmp_path):
    """A W=2 PT round-trips through save_checkpoint with
    /orchestrator/window_groups/<g>/ on disk."""
    import h5py
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce = make_wl_ce()
    atoms = make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
        data_container_file=None,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == "5"
        assert "orchestrator/window_groups/0" in f
        assert "orchestrator/window_groups/1" in f
        assert "exchange_idx" not in f["orchestrator/window_groups/0"]


def test_read_window_groups_raises_on_phase_mismatch(tmp_path):
    """_read_window_groups raises ValueError when the on-disk group phase
    disagrees with any of its walkers' _last_state['phase']."""
    import h5py
    from mchammer.calculators import ClusterExpansionCalculator

    from mchammer_pt.checkpoint import _read_window_groups
    from mchammer_pt.history import read_hdf5
    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    def _initial_energy():
        ce = make_wl_ce()
        atoms = make_wl_atoms()
        return float(
            ClusterExpansionCalculator(atoms, ce).calculate_total(
                occupations=atoms.numbers
            )
        )

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=2,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    # Corrupt: flip the group phase on window 0 to a value that
    # disagrees with the walkers' _last_state['phase'].
    with h5py.File(path, "r+") as f:
        del f["orchestrator/window_groups/0/phase"]
        f["orchestrator/window_groups/0"].create_dataset(
            "phase", data="__corrupted__",
        )

    _, containers, _ = read_hdf5(path)
    with pytest.raises(ValueError, match="phase consistency"):
        _read_window_groups(path, containers)


def test_w1_only_checkpoint_omits_window_groups_subgroup(tmp_path):
    """An all-W=1 v4 checkpoint omits /orchestrator/window_groups/ entirely;
    _read_window_groups returns all Nones."""
    import h5py

    from mchammer_pt.checkpoint import _read_window_groups
    from mchammer_pt.wl import WangLandauParallelTempering
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    def _initial_energy():
        from mchammer.calculators import ClusterExpansionCalculator
        return float(
            ClusterExpansionCalculator(make_wl_atoms(), make_wl_ce())
            .calculate_total(occupations=make_wl_atoms().numbers)
        )

    e0 = _initial_energy()
    pt = WangLandauParallelTempering(
        cluster_expansion=make_wl_ce(),
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 50.0), (e0 - 50.0, None)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=0,
        n_walkers_per_window=1,
    )
    pt.run(n_cycles=2)
    path = tmp_path / "ckpt.h5"
    pt.save_checkpoint(path)

    with h5py.File(path, "r") as f:
        # The writer must omit the /orchestrator/window_groups/ subgroup
        # entirely when every window has W=1 (not create-and-leave-empty).
        assert "window_groups" not in f["orchestrator"]
    assert _read_window_groups(path) == [None, None]
