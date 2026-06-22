# mchammer-pt

Replica-exchange orchestrators for [`mchammer`](https://icet.materialsmodeling.org/)
Monte Carlo with `icet` cluster expansions: canonical-ensemble
parallel tempering across a temperature ladder, and replica-exchange
Wang-Landau (REWL) across an energy-window ladder, with optional
multiple walkers per window.

For an architecture overview, see [docs/architecture.md](docs/architecture.md).

## Why

`mchammer`'s canonical ensemble samples at a single temperature. Large
supercells with competing ordered basins can trap the chain in local
minima; a single-temperature chain may never visit the other basin.
Parallel tempering runs `N` replicas at different temperatures and
periodically proposes configuration swaps between adjacent replicas,
so a high-temperature chain can cross barriers and deliver escape
paths to the colder chains.

`mchammer`'s single-walker Wang-Landau samples a fixed energy window,
but on rugged density-of-states landscapes the walker spends long
stretches near the window edges and the fill-factor schedule stalls
before the histogram is flat. REWL splits the target energy range
into overlapping windows and proposes configuration swaps between
adjacent windows weighted by their within-window density-of-states
ratio, so each walker mixes faster inside its own window and the
combined run converges in wall-clock time that scales with window
width rather than total energy range. Each window optionally runs
multiple walkers in lockstep that share the flatness gate and merge
their entropy estimates, further reducing the random-walk variance
that drives Wang-Landau's per-window convergence cost.

## Features

- `CanonicalParallelTempering` — canonical-ensemble PT with an
  arbitrary temperature ladder.
- `WangLandauParallelTempering` — replica-exchange Wang-Landau
  (REWL) on top of icet's `WangLandauEnsemble`. Each window owns a
  fixed energy range; adjacent windows attempt configuration swaps
  with a within-window density-of-states ratio for acceptance.
  `n_walkers_per_window` (scalar or per-window sequence) runs
  multiple WL walkers inside the same window, sharing the flatness
  gate and merging entropies across the group — straightforward to
  configure here, not exposed by raw icet. To use the
  Belardinelli-Pereyra 1/t schedule, pass
  `ensemble_kwargs={'schedule': '1_over_t'}`; the default
  `schedule='halving'` gives the standard WL fill-factor scheme.
  Serial and process-parallel backends as for the canonical
  orchestrator; checkpoint/resume into either pool kind.
- `mchammer_pt.analysis.dos.stitch_entropy` and
  `reweight_canonical_from_dos` post-process REWL output: stitch the
  per-window ln g(E) curves into a single density of states (working
  in log space, with bin-index matching that survives ULP-level
  energy drift between windows), then evaluate canonical
  thermodynamic observables on a user-supplied temperature grid.
- `mchammer-pt-stitch` and `mchammer-pt-reweight` console scripts
  expose the same pipeline from the command line, reading either an
  mchammer-pt checkpoint HDF5 or `WangLandauDataContainer` files
  directly.
- Serial and multiprocessing backends, swappable via a single
  constructor argument.
- Custom Monte Carlo moves: pass any `mchammer.CanonicalEnsemble`
  subclass via `ensemble_cls=`, with extra constructor arguments
  forwarded via `ensemble_kwargs=`. Custom `_do_trial_step` overrides
  ride the PT machinery without subclassing `Replica`.
- Per-replica `mchammer.BaseObserver` attachment on both serial and
  process-parallel pools, with each replica receiving its own
  observer copy. Three attach paths cover the spectrum: pass an
  observer instance for the common case (`attach_observer`), a class
  plus constructor arguments when picklable (`attach_observer_class`),
  or a top-level factory that constructs the observer inside each
  worker — required for observers like `ClusterCountObserver` whose
  constructors take icet `ClusterSpace` objects that do not pickle
  (`attach_observer_factory`). The factory reloads the
  `ClusterExpansion` from disk via
  `ClusterExpansion.read(replica.cluster_expansion_path)`;
  `ProcessPool` auto-populates the path on every worker.
- HDF5 output bundling one `mchammer.BaseDataContainer` per replica plus
  a compact `ExchangeHistory` of per-pair swap statistics and
  replica-label trajectories.
- Round-trip count and integrated-autocorrelation-time diagnostics
  as pure functions over the run output.
- `ExchangeCallback` protocol for PT-level events (with `ExchangePrinter`
  and `SwapRateTracker` built-ins).
- `CycleCallback` protocol for per-cycle hooks, with `ProgressPrinter`
  built-in for periodic stderr progress lines on long runs (cycle,
  percent, elapsed, ETA, swap-acceptance rates).
- `CheckpointWriter` cycle callback and
  `CanonicalParallelTempering.resume(...)` for crash-safe long runs
  and bit-identical continuation across `pt.run()` calls (after
  `ExchangeHistory.concatenate`). Same payload also written by
  `pt.save_checkpoint(path)` and via the existing
  `data_container_file=` constructor kwarg.
- `mchammer_pt.testing.assert_boltzmann_sampling` — public utility for
  pinning the empirical stationary distribution of a custom
  `CanonicalEnsemble` subclass against an analytic Boltzmann fixture.
  Downstream packages providing custom moves can use this to pin
  stationarity correctness against the same anchor as mchammer-pt's
  own test suite.

## Install

    pip install mchammer-pt

Requires Python 3.11+ and `icet>=3.2` (installed automatically from
PyPI).

The `seeding` and `contrib` features additionally require
`mchammer-moves`; pull it in with the `custom-moves` extra:

    pip install 'mchammer-pt[custom-moves]'

For development, clone the repository and install editable with the dev
tooling (`pytest`, `mypy`, `ruff`):

    pip install -e '.[dev]'

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

# Optional: live progress on stderr for long runs.
from mchammer_pt import ProgressPrinter
pt.attach_cycle_callback(ProgressPrinter(interval=100))

pt.run(n_cycles=200)

# Diagnostics.
from mchammer_pt import (
    round_trip_counts,
    swap_acceptance_rates,
    energy_autocorrelation_time,
)
print("acceptance:", swap_acceptance_rates(pt.history))
print("round-trips:", round_trip_counts(pt.history.replica_labels_per_cycle))
# The single-argument form above is for one walker per rung (canonical PT
# and single-walker REWL). For multi-walker REWL pass the window mapping,
# which the history carries (so it works on a run read back from disk):
#   round_trip_counts(pt.history.replica_labels_per_cycle,
#                     pt.history.window_of_position)
for r in range(len(pt.pool)):
    tau = energy_autocorrelation_time(pt.history.energies_per_cycle[:, r])
    print(f"replica {r}: tau = {tau:.1f} cycles")
```

For multiprocess parallelism, use the `process_pool` classmethod:

```python
with CanonicalParallelTempering.process_pool(
    cluster_expansion=ce,
    atoms=atoms,
    temperatures=[200, 400, 800, 1600],
    block_size=1000,
    random_seed=0,
) as pt:
    pt.run(n_cycles=200)
```

The factory handles seed spawning, writing the CE to a managed temp
directory, and constructing a `ProcessPool` at the same ladder as
the orchestrator. See `examples/03_parallel_workers.py`.

Observer attachment is supported on both `SerialPool` and `ProcessPool`.
See the Features list above for the three attach paths and when to use each.

For custom Monte Carlo moves, subclass `mchammer.CanonicalEnsemble`
and pass via `ensemble_cls=`:

```python
from mchammer.ensembles import CanonicalEnsemble

class MyMove(CanonicalEnsemble):
    def _do_trial_step(self) -> int:
        # ... your custom move ...
        return super()._do_trial_step()

with CanonicalParallelTempering.process_pool(
    cluster_expansion=ce,
    atoms=atoms,
    temperatures=[200, 400, 800, 1600],
    block_size=1000,
    random_seed=0,
    ensemble_cls=MyMove,
) as pt:
    pt.run(n_cycles=200)
```

Spawn workers re-import the class by fully qualified name, so define
the subclass in a `.py` module file rather than a Jupyter cell. See
`examples/05_custom_ensemble.py` for a complete worked example.

### Wang-Landau parallel tempering

For Wang-Landau parallel tempering, build per-window starting
configurations whose energies lie inside their assigned windows,
then drive `WangLandauParallelTempering.from_bin_count` (or pass
explicit `windows=` for non-uniform splits):

```python
from mchammer_pt import WangLandauParallelTempering

# `per_window_atoms` is a list[Atoms], one per window, with each
# entry's energy in the corresponding window. Generating these
# is the user's responsibility — typically a short pilot MC run.
pt = WangLandauParallelTempering.from_bin_count(
    cluster_expansion=ce,
    atoms=per_window_atoms,
    n_bins=4,
    energy_spacing=1.0,
    minimum_energy=-32.0,
    maximum_energy=32.0,
    overlap=4,
    block_size=len(per_window_atoms[0]) * 1000,
    random_seed=0,
)
pt.run(n_cycles=500)
```

`pt.run(...)` exits early once every replica reports `converged`.
`WangLandauParallelTempering.process_pool(...)` spawns one OS
process per replica. `save_checkpoint(path)` / `resume(path, ...)`
/ `resume_process_pool(path, ...)` mirror the canonical surface.
Observers attach the same way as on the canonical pool (via
`pt.attach_observer(...)` or, for the class and factory paths,
directly on `pt.pool`); each replica's recorded observable
trajectory ends up in its `WangLandauDataContainer`, ready for
icet's `get_average_observables_wl` against the stitched ln g(E).

Stitch the per-window ln g(E) curves into a single density of
states, then reweight onto a canonical temperature grid:

```python
from mchammer_pt.analysis.dos import (
    reweight_canonical_from_dos,
    stitch_entropy,
)

per_window = [r.get_entropy() for r in pt.results()]
stitched, errors = stitch_entropy(per_window, energy_spacing=1.0)
canonical = reweight_canonical_from_dos(
    stitched, temperatures=[100, 200, 400, 800, 1600],
)
```

The same pipeline is available from the command line via the
`mchammer-pt-stitch` and `mchammer-pt-reweight` console scripts, which
read either an mchammer-pt checkpoint HDF5 or
`WangLandauDataContainer` files directly. Pass `--multi-run` with two
or more checkpoints to merge independent seeds of the same system into
one consensus DOS (each window is merged across runs before
stitching). For production runs on a new system, plan
to validate the recovered DOS against ground truth (e.g. by
brute-force enumeration on a small case, or against an analytic
result) before trusting downstream thermodynamic averages.

## Examples

- `examples/01_basic_canonical.py` — self-contained run on a toy Cu/Au CE.
- `examples/02_custom_callback.py` — writing your own `ExchangeCallback`.
- `examples/03_parallel_workers.py` — PT with the `ProcessPool`.
- `examples/04_equilibrium_sampling.py` – discarding the initial burn-in period for equilibrium sampling.
- `examples/05_custom_ensemble.py` — PT with a custom
  `CanonicalEnsemble` subclass.
- `examples/06_progress_monitoring.py` — live progress on stderr for
  long runs via `ProgressPrinter`.
- `examples/07_resume.py` — checkpoint and resume a PT run, with
  bit-identical continuation.
- `examples/08_rewl.py` — replica-exchange Wang-Landau on a 4x4
  2D Ising model, with per-window seeding and DOS stitching.
- `examples/09_dos_postprocessing.py` — stitching REWL output into
  a single ln g(E) and reweighting onto a canonical temperature
  grid via `mchammer_pt.analysis.dos`.

## License

MIT.
