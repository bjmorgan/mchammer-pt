"""Custom ExchangeCallback example.

Records per-pair swap probabilities over time, then prints the
cumulative running acceptance rate per pair at the end of the run.

Run from the repo root:

    python examples/02_custom_callback.py
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from icet import ClusterExpansion, ClusterSpace

from mchammer_pt import CanonicalParallelTempering


class AcceptanceTrace:
    """Records (cycle, pair_index, accepted) tuples for later analysis."""

    def __init__(self) -> None:
        self.events: list[tuple[int, int, bool]] = []

    def on_exchange(
        self,
        cycle: int,
        pair_index: int,
        accepted: bool,
        log_prob_ratio: float,
    ) -> None:
        self.events.append((cycle, pair_index, bool(accepted)))


def build_toy_ce() -> ClusterExpansion:
    primitive = bulk("Cu", "fcc", a=4.0, cubic=True)
    cs = ClusterSpace(structure=primitive, cutoffs=[3.5], chemical_symbols=["Cu", "Au"])
    rng = np.random.default_rng(0)
    params = rng.normal(scale=0.05, size=len(cs))
    params[0] = -1.0
    return ClusterExpansion(cluster_space=cs, parameters=params)


def main() -> None:
    ce = build_toy_ce()
    atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(1)
    au_indices = rng.choice(len(atoms), size=len(atoms) // 2, replace=False)
    symbols = np.array(atoms.get_chemical_symbols())
    symbols[au_indices] = "Au"
    atoms.set_chemical_symbols(symbols.tolist())

    pt = CanonicalParallelTempering(
        cluster_expansion=ce,
        atoms=atoms,
        temperatures=[200.0, 400.0, 800.0],
        block_size=100,
        random_seed=0,
    )
    trace = AcceptanceTrace()
    pt.attach_callback(trace)
    pt.run(n_cycles=80)

    # Running acceptance rate per pair.
    per_pair: dict[int, list[int]] = {0: [], 1: []}
    for _, pair, accepted in trace.events:
        per_pair[pair].append(int(accepted))
    for pair, seq in per_pair.items():
        rate = np.cumsum(seq) / np.arange(1, len(seq) + 1)
        print(
            f"pair {pair}: final cumulative acceptance = "
            f"{rate[-1]:.2%} over {len(seq)} attempts"
        )


if __name__ == "__main__":
    main()
