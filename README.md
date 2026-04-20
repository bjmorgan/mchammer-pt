# mchammer-pt

Parallel-tempering orchestrator for [`mchammer`](https://icet.materialsmodeling.org/)
canonical Monte Carlo simulations with `icet` cluster expansions.

`mchammer`'s canonical ensemble samples at a single temperature; large
supercells with competing ordered basins can trap the chain in local
minima. `mchammer-pt` runs N `CanonicalEnsemble` replicas at different
temperatures in parallel, periodically proposing configuration swaps
between adjacent-temperature replicas via the standard parallel
tempering exchange criterion. Configurations can thereby traverse
kinetic barriers by way of the high-temperature chains.

## Install

    pip install -e .

Requires Python 3.11+ and a working `icet` install.

## Quickstart

```python
from ase.build import bulk
from icet import ClusterExpansion
from mchammer_pt import CanonicalParallelTempering

ce = ClusterExpansion.read("my_ce.ce")
atoms = bulk("Cu", "fcc", a=4.0, cubic=True).repeat((4, 4, 4))
# ... decorate atoms with the correct initial composition ...

pt = CanonicalParallelTempering(
    cluster_expansion=ce,
    atoms=atoms,
    temperatures=[100, 200, 350, 550, 800, 1200, 1800, 2700],
    block_size=1000,
    random_seed=0,
    parallel="serial",             # or "processes"
    data_container_file="pt.h5",
)
pt.run(n_cycles=200)
print(pt.swap_acceptance_rates)
```

See `examples/` for more.

## License

MIT.
