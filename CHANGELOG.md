# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ObservablePool.get_observers(replica_index) -> dict[str, BaseObserver]`
  — recover worker-side observer state at end-of-run. Returns a
  snapshot dict (keyed by observer tag) of the observers attached
  to a single replica; values are independent copies via `pickle`
  round-trip so mutations on the returned objects do not affect
  the pool's running state. Mid-run retrieval is supported.
  Implemented on both `SerialPool` and `ProcessPool`.
- `Replica.cluster_expansion_path: str | None` keyword-only
  constructor argument and read-only property. Auto-populated on
  every worker spawned by `ProcessPool` from the pool's `ce_path`;
  optional on `SerialPool`. Lets factory-path observers reload a
  fresh `ClusterExpansion` via
  `ClusterExpansion.read(replica.cluster_expansion_path)` without
  hardcoding the path.

### Fixed

- `_check_importable` (the spawn-import guard for `ensemble_cls`,
  observer classes, and observer factories) now accepts callable
  instances of user-defined classes. Previously it required the
  argument itself to expose `__qualname__`, which functions and
  classes have but instances do not; users had to monkey-patch
  `__qualname__` onto the instance to make `attach_observer_factory`
  work. The check now falls through to `type(obj).__qualname__`,
  which is what `pickle` walks anyway.
- `attach_observer_factory` docstrings on both pools previously
  recommended reaching for `replica.ensemble.calculator.cluster_expansion`
  to obtain a `ClusterSpace`. The calculator mutates that
  `ClusterSpace` during runs, so observers built from it produced
  wrong-length cluster vectors at observation time. Docstrings now
  point at `ClusterExpansion.read(replica.cluster_expansion_path)`,
  which always yields an unmutated copy.

## [0.3.0] - 2026-04-29

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
- `ProcessPool` shuts itself down and refuses subsequent operations
  if a worker reports ERR (or its pipe closes) mid-`attach_observer*`.
  Pre-fix the contract was a docstring promise that "the run should
  abort"; now the failure path is the mechanism — pending replies
  on later workers are drained, the pool transitions to shut-down
  state, and the user gets a framed `RuntimeError` carrying the
  worker-side cause. Subsequent calls refuse via the shutdown
  guard.
- Replica selection in every `attach_observer*` call eagerly rejects
  out-of-range indices with `IndexError`, and silently dedupes
  repeated indices (`replicas=[0, 0]` is equivalent to
  `replicas=[0]`).

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
