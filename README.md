# mchammer-pt

Parallel-tempering orchestrator for [`mchammer`](https://icet.materialsmodeling.org/)
canonical Monte Carlo simulations with `icet` cluster expansions.

## Why

`mchammer`'s canonical ensemble samples at a single temperature. Large
supercells with competing ordered basins can trap the chain in local
minima; a single-temperature chain may never visit the other basin.
Parallel tempering runs `N` replicas at different temperatures and
periodically proposes configuration swaps between adjacent replicas,
so a high-temperature chain can cross barriers and deliver escape
paths to the colder chains.

## Features

- `CanonicalParallelTempering` — canonical-ensemble PT with an
  arbitrary temperature ladder.
- Serial and multiprocessing backends, swappable via a single
  constructor argument.
- Per-replica `mchammer.BaseObserver` attachment, pass-through — use
  your existing mchammer observers unchanged.
- HDF5 output bundling one `mchammer.DataContainer` per replica plus
  a compact `ExchangeHistory` of per-pair swap statistics and
  replica-label trajectories.
- Round-trip count and integrated-autocorrelation-time diagnostics
  as pure functions over the run output.
- `ExchangeCallback` protocol for PT-level events (with `ExchangePrinter`
  and `SwapRateTracker` built-ins).

## Install

    pip install -e .

Requires Python 3.11+, `icet`, and a working MC environment.
Optional dev tooling: `pip install -e '.[dev]'` adds `pytest`,
`mypy`, `ruff`.

## Quickstart

```python
from ase.build import bulk
from icet import ClusterExpansion
from mchammer_pt import CanonicalParallelTempering

ce = ClusterExpansion.read("my_ce.ce")
atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((4, 4, 4))
# ... decorate atoms with the correct composition ...

pt = CanonicalParallelTempering(
    cluster_expansion=ce,
    atoms=atoms,
    temperatures=[100, 200, 350, 550, 800, 1200, 1800, 2700],
    block_size=1000,
    random_seed=0,
    data_container_file="pt.h5",
)
pt.run(n_cycles=200)

# Diagnostics.
from mchammer_pt import (
    round_trip_counts,
    swap_acceptance_rates,
    energy_autocorrelation_time,
)
print("acceptance:", swap_acceptance_rates(pt.history))
print("round-trips:", round_trip_counts(pt.history.replica_labels_per_cycle))
for r in range(len(pt.replicas)):
    tau = energy_autocorrelation_time(pt.history.energies_per_cycle[:, r])
    print(f"replica {r}: tau = {tau:.1f} cycles")
```

For the multiprocessing backend, construct a `ProcessBackend` and
pass it via the `backend=` argument — see
`examples/03_parallel_workers.py`.

## Examples

- `examples/01_basic_canonical.py` — self-contained run on a toy Cu/Au CE.
- `examples/02_custom_callback.py` — writing your own `ExchangeCallback`.
- `examples/03_parallel_workers.py` — PT with the `ProcessBackend`.

## Status

`mchammer-pt` is `0.x` pre-release. The public API may shift before
`1.0`. Currently ships canonical-ensemble PT only; semi-grand-canonical
and variance-constrained SGC are planned for a future minor version.

## Upstream

If this idea turns out to be useful, the preferred home for the
functionality is inside [`mchammer`](https://icet.materialsmodeling.org/)
itself. Questions about that direction should go to Paul Erhart.

## License

MIT.
