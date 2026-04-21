"""Tests for parallel backends (serial + processes)."""
from __future__ import annotations

import numpy as np

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
