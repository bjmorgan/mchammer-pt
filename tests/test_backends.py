"""Tests for parallel backends (serial + processes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mchammer_pt.parallel.processes import ProcessBackend
from mchammer_pt.parallel.serial import SerialBackend
from mchammer_pt.replica import Replica


def _replicas(toy_ce, toy_atoms) -> list[Replica]:
    return [
        Replica(toy_ce, toy_atoms, temperature=300.0 + 100 * i, random_seed=i)
        for i in range(3)
    ]


def test_serial_backend_advances_all_replicas(toy_ce, toy_atoms):
    backend = SerialBackend()
    replicas = _replicas(toy_ce, toy_atoms)
    energies_before = np.array([r.current_energy() for r in replicas])
    backend.advance_all(replicas, n_steps=200)
    energies_after = np.array([r.current_energy() for r in replicas])
    # With 200 steps on 108 sites at 300-500 K, every replica should drift.
    assert not np.allclose(energies_before, energies_after)
    backend.shutdown()


def test_serial_backend_shutdown_is_idempotent():
    backend = SerialBackend()
    backend.shutdown()
    backend.shutdown()


def test_process_backend_parity_with_serial(tmp_path: Path, toy_ce, toy_atoms):
    # Same seeds + same start -> serial and process backends must produce
    # identical energies after one advance step.
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))

    replicas_serial = _replicas(toy_ce, toy_atoms)
    SerialBackend().advance_all(replicas_serial, n_steps=100)
    energies_serial = np.array([r.current_energy() for r in replicas_serial])
    occ_serial = [r.current_occupations() for r in replicas_serial]

    replicas_processes = _replicas(toy_ce, toy_atoms)
    backend = ProcessBackend(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0, 500.0],
        seeds=[0, 1, 2],
    )
    try:
        backend.advance_all(replicas_processes, n_steps=100)
        # Pull the occupations from the remote workers back into the
        # local replicas so the next block would continue coherently.
        for replica, occ in zip(
            replicas_processes, backend.current_occupations(), strict=True
        ):
            replica.set_occupations(occ)
        energies_processes = np.array(
            [r.current_energy() for r in replicas_processes]
        )
        occ_processes = [r.current_occupations() for r in replicas_processes]
    finally:
        backend.shutdown()

    np.testing.assert_array_equal(energies_serial, energies_processes)
    for occ_s, occ_p in zip(occ_serial, occ_processes, strict=True):
        np.testing.assert_array_equal(occ_s, occ_p)


def test_process_backend_shutdown_cleans_up_temp_ce(tmp_path: Path, toy_ce, toy_atoms):
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))
    backend = ProcessBackend(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 400.0],
        seeds=[0, 1],
    )
    assert backend._workers  # workers are live
    backend.shutdown()
    assert not backend._workers  # all joined / cleared
