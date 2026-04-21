"""Integration tests exercising the full orchestrator."""

from __future__ import annotations

import numpy as np
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble

from mchammer_pt import CanonicalParallelTempering


def test_mean_energy_same_as_single_ensemble_at_same_T(toy_ce, toy_atoms):
    """If every replica is at the same T, exchange is a permutation,
    and the mean energy averaged over all replicas and cycles should
    match a standalone single-ensemble run at that T to within 5 sigma.
    """
    T = 1000.0
    block_size = 100
    n_cycles = 100
    n_replicas = 4
    burn = 20  # burn-in cycles discarded from both runs

    # PT reference: 4 replicas all at T, configurations get shuffled by exchange.
    pt = CanonicalParallelTempering(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[T] * n_replicas,
        block_size=block_size,
        random_seed=1,
    )
    history = pt.run(n_cycles=n_cycles)
    pt_energies = history.energies_per_cycle[burn:].ravel()

    # Single-ensemble reference: run for a comparable number of MC steps.
    atoms_copy = toy_atoms.copy()
    calc = ClusterExpansionCalculator(atoms_copy, toy_ce)
    ens = CanonicalEnsemble(
        structure=atoms_copy,
        calculator=calc,
        temperature=T,
        random_seed=42,
    )
    total_steps = block_size * n_cycles * n_replicas
    ens.run(total_steps)
    # Drop first 20% as burn-in.
    trace = ens.data_container.data["potential"].to_numpy()
    single_energies = trace[len(trace) // 5 :]

    # Compare means.
    mean_pt = pt_energies.mean()
    mean_single = single_energies.mean()
    # Standard error on the PT mean (naive, ignores autocorrelation).
    sem_pt = pt_energies.std() / np.sqrt(pt_energies.size)
    assert abs(mean_pt - mean_single) < 5 * sem_pt


def test_parallel_serial_parity_same_as_serial(toy_ce, toy_atoms, tmp_path):
    """`SerialPool` and `ProcessPool` agree on per-pair swap_attempted.

    Pair-set scheduling is backend-agnostic — it depends only on cycle
    index and n_replicas — so swap attempt counts must match between
    backends for the same (n_cycles, n_replicas). `swap_accepted` and
    `energies_per_cycle` are not compared because the two backends
    seed their per-replica RNG streams independently and therefore
    produce different trajectories.
    """
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))

    kwargs = dict(
        cluster_expansion=toy_ce,
        atoms=toy_atoms,
        temperatures=[300.0, 600.0, 1200.0],
        block_size=50,
        random_seed=0,
    )
    pt_serial = CanonicalParallelTempering(**kwargs)
    h_serial = pt_serial.run(n_cycles=3)

    from mchammer_pt.parallel.processes import ProcessPool

    pool = ProcessPool(
        ce_path=ce_path,
        initial_atoms=toy_atoms,
        temperatures=[300.0, 600.0, 1200.0],
        seeds=[
            int(np.random.SeedSequence(0).spawn(4)[i].generate_state(1)[0])
            for i in range(3)
        ],
    )
    try:
        pt_processes = CanonicalParallelTempering(**kwargs, pool=pool)
        h_processes = pt_processes.run(n_cycles=3)
    finally:
        pool.shutdown()

    np.testing.assert_array_equal(h_serial.swap_attempted, h_processes.swap_attempted)
