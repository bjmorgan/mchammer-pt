# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ProcessPool` now satisfies `ObservablePool`: observers can be
  attached to process-parallel runs without falling back to
  `SerialPool`. Closes the long-standing parity gap that forced
  users to choose between observers and parallelism.
- `ObservablePool.attach_observer_class(cls, /, *args, replicas, **kwargs)`
  — escape hatch for observers whose instances do not pickle but
  whose constructor arguments do. Constructs `cls(*args, **kwargs)`
  once per selected replica inside that replica's process. The
  parent runs an eager dry-run construction so bad arguments raise
  at the call site rather than from a worker.
- `ObservablePool.attach_observer_factory(factory)` — for observers
  whose constructors take inputs that do not pickle (notably icet
  `ClusterSpace` and `ClusterExpansion`). The factory is a top-level
  callable that runs inside each worker with that worker's `Replica`
  as its argument; it reaches icet objects via
  `replica.ensemble.calculator.cluster_expansion`. icet objects
  never cross the process boundary.

### Changed

- `ObservablePool.attach_observer` parameter renamed from `indices=`
  to `replicas=`. The new name is semantically clearer and avoids
  collision with constructor kwargs forwarded through
  `attach_observer_class(**kwargs)`. The same rename was applied to
  `BaseParallelTempering.attach_observer` for consistency.
- `SerialPool.attach_observer` now gives each selected replica its
  own deserialised observer copy via a `pickle` round-trip, instead
  of registering the same Python instance on every replica. Stateful
  observers (counters, accumulators, private RNGs) no longer share
  state across replicas. Stateless observers — the typical case —
  see no observable change.
- `ProcessPool` raises `RuntimeError("pool is shut down")` from
  every public method called after `shutdown()`. Previously these
  silently no-opped or raised an opaque `IndexError`.

### Internal

- `_check_ensemble_cls_importable` (process-pool spawn-import guard)
  generalised to `_check_importable(obj, *, kind)` and moved to
  `mchammer_pt/parallel/_imports.py`. Now accepts both classes and
  callables; reused for `ensemble_cls`, the class argument to
  `attach_observer_class`, and the callable argument to
  `attach_observer_factory`.

## [0.2.0] - 2026-04-28

### Added

- `mchammer_pt.testing.assert_boltzmann_sampling` — public utility
  for pinning the empirical stationary distribution of a
  `CanonicalEnsemble` subclass against an analytic Boltzmann fixture
  (4-site 1D chain, NN-only pair ECI, ΔE ≈ 3 kT at the test
  temperature). Downstream packages providing custom ensembles can
  pin their stationarity correctness against the same anchor as
  mchammer-pt's own test suite. Exposes `FIXTURE_CHAIN_INDICES` for
  consumers whose `ensemble_kwargs` depend on the fixture's chain
  geometry.
- `Replica`, `CanonicalParallelTempering`, and
  `CanonicalParallelTempering.process_pool` accept new keyword-only
  parameters `ensemble_cls` (a `CanonicalEnsemble` subclass; defaults
  to `CanonicalEnsemble`) and `ensemble_kwargs` (extra keyword
  arguments forwarded to the ensemble constructor). Lets callers
  run parallel tempering with custom Monte Carlo moves implemented
  as `CanonicalEnsemble` subclasses without duplicating `Replica`'s
  body. Resolves #6.
- `ProcessPool` gained the same two parameters and forwards them to
  each worker. Workers must be able to import the supplied class by
  fully qualified module name (i.e. defined in a module file).
- `Replica` rejects `ensemble_kwargs` containing any of `structure`,
  `calculator`, `temperature`, or `random_seed` (set by `Replica`
  itself). `CanonicalParallelTempering` rejects `pool=` combined
  with non-default `ensemble_cls` / `ensemble_kwargs`.

## [0.1.0] - 2026-04-21

### Added

- `CanonicalParallelTempering` — canonical-ensemble PT orchestrator
  over an arbitrary temperature ladder. Constructor takes a
  cluster expansion, starting atoms, temperatures, block size, and
  random seed; `run(n_cycles)` returns an `ExchangeHistory`.
- `CanonicalParallelTempering.process_pool(...)` classmethod for
  process-parallel runs; owns seed spawning, CE tempdir lifecycle,
  and pool construction so the pool and orchestrator cannot disagree
  on the temperature ladder. Usable as a context manager.
- `BaseParallelTempering` abstract orchestrator for future ensemble
  types; subclasses override `_log_prob_ratio(i, j)`.
- `ReplicaPool` protocol with `SerialPool` and `ProcessPool`
  implementations. `ObservablePool` sub-protocol adds
  `attach_observer`; satisfied by `SerialPool` only.
  `ProcessPool` uses persistent worker processes with a narrow
  command protocol and structured error forwarding.
- `ExchangeCallback` protocol for per-exchange hooks, plus
  `SwapRateTracker` and `ExchangePrinter` built-ins.
- `ExchangeHistory` dataclass capturing per-cycle energies,
  replica-label trajectories, and per-pair swap counts. Written as
  an atomic HDF5 bundle alongside one native
  `mchammer.BaseDataContainer` per replica; `read_hdf5` validates
  the schema on load.
- Diagnostics: `round_trip_counts`, `swap_acceptance_rates`,
  `energy_autocorrelation_time` (Sokal-window estimator with a
  warning when the window does not close).
- Context-manager support on `ProcessPool` and
  `BaseParallelTempering` for exception-safe worker shutdown.
- Per-replica RNG isolation around `mchammer`'s global-`random`
  Monte Carlo driver so co-tenant replicas evolve independently and
  constructing or advancing a replica has no observable side effect
  on the caller's `random` state.
- Non-finite log-probability ratios surface as `RuntimeError` with
  cycle, pair, and energy context instead of flowing through
  `metropolis_accept` silently.
- Three worked examples (basic canonical run, custom callback,
  process-parallel run) against a synthetic Cu/Au cluster
  expansion; no external files required.
- Test suite covering protocol conformance, RNG isolation,
  exchange correctness, HDF5 atomicity and schema validation, and
  end-to-end serial/parallel agreement. CI runs pytest + mypy
  (strict) + ruff on Python 3.11, 3.12, 3.13, 3.14.
