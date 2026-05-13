# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Single-walker replica-exchange Wang-Landau (REWL) on top of
  icet's ``WangLandauEnsemble``. Each replica owns a fixed energy
  window; adjacent windows attempt configuration swaps between
  cycles using a within-window log-density-of-states ratio for
  acceptance. The base ``WangLandauEnsemble`` from mainline icet is
  the default; pass ``ensemble_cls=OneOverTWangLandauEnsemble``
  explicitly (from icet's patched fork at
  https://gitlab.com/bjmorgan/icet) to use the 1/t schedule.
- ``WangLandauParallelTempering`` orchestrator with
  ``save_checkpoint``/``resume`` (serial pool) and
  ``resume_process_pool``. ``run()`` stops early when every replica
  reports converged. ``from_bin_count`` convenience constructor
  wraps icet's ``get_bins_for_parallel_simulations`` for a uniform
  window split.
- ``WangLandauReplica`` wrapper handle. Validates initial energy is
  in window at construction; ``set_occupations`` refreshes the
  cached ``_potential`` and ``_reached_energy_window`` so swap-
  delivered configurations are bookkept correctly.
- ``WangLandauPool`` protocol plus ``SerialWangLandauPool`` and
  ``ProcessWangLandauPool`` implementations. The process pool
  spawns one OS process per replica with a new ``_wl_worker``
  entry point and two REWL-specific opcodes (``LOG_G_AT``,
  ``CONVERGED``) on top of the canonical-shared set.
- Observer attach for REWL pools. ``SerialWangLandauPool`` and
  ``ProcessWangLandauPool`` now satisfy a new
  ``WangLandauObservablePool`` protocol (mirrors
  ``ObservablePool`` for canonical) with ``attach_observer``,
  ``attach_observer_class``, ``attach_observer_factory``, and
  ``get_observers``. Use this to record per-step observable
  trajectories during a REWL run, then feed the per-replica data
  containers into icet's ``get_average_observables_wl`` for
  thermodynamic averages against the stitched density of states.
- A ``slow``-marked integration test
  (``tests/integration/test_rewl_2d_ising.py``) exercising REWL
  end-to-end on a 4x4 2D Ising fixture. After convergence, the
  stitched ln g(E) curve from ``get_density_of_states_wl`` is
  compared against the exact density of states obtained by
  brute-force enumeration over all 2^16 configurations; the
  residual is constrained in both standard deviation and maximum
  deviation. A sign error or systematic bias in the swap formula
  would push the recovered curve away from the analytic result and
  the test would fail.

### Changed

- Default ``ensemble_cls`` for ``WangLandauReplica`` and
  ``WangLandauParallelTempering`` is now base ``WangLandauEnsemble``
  rather than ``OneOverTWangLandauEnsemble``. The 1/t schedule is
  available only in icet's patched fork at
  https://gitlab.com/bjmorgan/icet; users wanting it now pass
  ``ensemble_cls=OneOverTWangLandauEnsemble`` explicitly. This
  decouples the package from the fork; mainline icet users can
  install and use mchammer-pt directly.
- Lifted ``temperatures`` off the base ``ReplicaPool`` protocol
  into a new ``CanonicalPool`` subprotocol. Existing
  ``SerialPool``/``ProcessPool`` classes continue to satisfy
  ``CanonicalPool`` (and therefore ``ObservablePool``) without
  caller changes. ``CanonicalParallelTempering.__init__`` now
  types its ``pool`` parameter as ``CanonicalPool | None``.
- ``BaseParallelTempering._try_exchange`` now accepts a ``-inf``
  log-probability ratio as a clean swap rejection (legal in REWL
  when one replica's partner energy lies outside its window).
  ``+inf`` and ``NaN`` continue to raise.
- ``WangLandauParallelTempering.cycles_completed`` renamed to
  ``cycles_in_segment`` to reflect that it counts cycles in the
  current ``run()`` call (resets to 0 on each call, including after
  resume), not cumulative cycles across the full trajectory.
- Bumped checkpoint schema to ``"3"``. Checkpoints written by
  v0.7.0 (schema ``"2"``) are not valid resume sources for this
  version; ``resume`` hard-errors with a clear message on the
  version mismatch. The schema change accompanies a refactor that
  moves ladder-specific meta keys (``temperatures`` for canonical
  PT, ``windows`` + ``energy_spacing`` for Wang-Landau PT) behind
  a per-subclass ``_checkpoint_meta()`` hook on
  ``BaseParallelTempering``.

## [0.7.0] - 2026-05-08

### Added

- `CanonicalParallelTempering.save_checkpoint(path)` — atomic
  full-checkpoint write at the call site. Captures everything
  needed to reconstruct the orchestrator at the saved state.
- `CheckpointWriter` `CycleCallback` and the convenience method
  `pt.attach_checkpoint_writer(path, interval=...)` for periodic
  mid-run checkpointing. Writes the same payload as
  `save_checkpoint` every ``interval`` cycles plus the final
  cycle, atomically. A failed write raises out of `on_cycle_end`
  and aborts the run with the partial history preserved.
- `CanonicalParallelTempering.resume(path, *, cluster_expansion)` —
  reconstructs a `SerialPool` orchestrator from a checkpoint and
  returns it ready for further `run()` calls. The bit-identical
  contract holds: a run of `N` cycles, checkpointed, resumed, and
  continued for `M` cycles produces an `ExchangeHistory` that —
  when concatenated with the prior history via
  `ExchangeHistory.concatenate` — equals byte-for-byte the history
  of a single run of `N+M` cycles from the same seed.
- `CanonicalParallelTempering.resume_process_pool(path, *, cluster_expansion)`
  — `ProcessPool` resume. Same identity validation as `resume`, but
  the bit-identical contract holds per-pool-kind only (worker
  scheduling non-determinism prevents cross-pool exact equivalence).
- `Replica.restart_from(container, *, cluster_expansion, atoms, ...)` —
  classmethod that constructs a `Replica` whose ensemble has been
  restored from a saved `BaseDataContainer` via mchammer's
  `_restart_ensemble`, plus the path-dependent
  `ConfigurationManager._sites_by_species` cache that
  `_restart_ensemble` does not itself restore but whose order
  controls `random.choice` outcomes in canonical trial-step
  proposals.
- `Replica.restore_state(container, *, sites_by_species=None)` —
  instance-method counterpart to `restart_from` for mutating an
  existing replica. Used by `ProcessPool` workers to restore state
  in place via the new `RESTORE_STATE` opcode.
- `Replica.snapshot_for_checkpoint() -> dict` — populates the
  live `BaseDataContainer._last_state` with the four fields
  `_restart_ensemble` reads on resume, and returns the additional
  per-replica state the orchestrator embeds in the checkpoint
  (notably `_sites_by_species`). mchammer's own
  `BaseEnsemble.write_data_container` performs the equivalent
  refresh inline; serialising containers directly via the
  checkpoint writer requires us to replicate it.
- `ReplicaPool.snapshot_for_checkpoint() -> list[dict]` protocol
  method — cross-pool snapshot capture. `SerialPool` calls each
  replica locally; `ProcessPool` round-trips a new
  `SNAPSHOT_FOR_CHECKPOINT` opcode to each worker.
- `examples/07_resume.py` — worked example demonstrating
  checkpoint/resume with the bit-identical concatenation pattern.

### Changed

- HDF5 ``meta/`` schema extended with ``schema_version``,
  ``random_seed``, ``ce_identity`` (sha256 of a stable canonical
  form: `to_dataframe().to_csv()` plus `chemical_symbols`,
  `cutoffs`, and the primitive structure — `ClusterExpansion.write`
  is not byte-deterministic and was rejected as a hash source),
  ``ensemble_cls_fqn``, and ``ensemble_kwargs_hash``. ``schema_version``
  bumped to ``"2"``; new top-level ``/orchestrator/`` group carries
  ``replica_labels`` and the JSON-encoded numpy ``bit_generator.state``;
  new top-level ``/sites_by_species/`` group carries one JSON
  dataset per replica.
- The existing ``data_container_file=`` constructor kwarg now
  produces full checkpoints. Files written via that path are
  resumable via `resume`/`resume_process_pool`. No behaviour change
  for users not using resume; the file payload simply includes the
  new schema additions.
- `BaseParallelTempering.run()` now publishes ``self._history`` to
  the in-progress history object before the cycle loop, so cycle
  callbacks (such as `CheckpointWriter`) see the live history rather
  than ``None``. The prior behaviour was that ``self._history``
  remained ``None`` until the run completed; the existing
  partial-history-on-exception contract was preserved via a
  ``try/finally`` that became redundant once the eager assignment
  landed and was removed.

### Notes

Files written by 0.6.0 (no checkpoint payload) or by intermediate
development builds (schema ``"1"``, lacking ``_sites_by_species``)
are not resumable. The schema-version guard hard-errors on any
value other than ``"2"`` with a clear message.

## [0.6.0] - 2026-05-08

### Added

- `CycleCallback` protocol on `BaseParallelTempering` — handlers fired
  once per PT cycle, after that cycle's history rows have been
  written. Mirrors the existing `ExchangeCallback` registration shape:
  `attach_cycle_callback(cb)`. Multiple cycle callbacks compose.
- `ProgressPrinter` built-in `CycleCallback` — emits periodic
  append-only progress lines to stderr during long PT runs, designed
  for monitoring multi-hour non-interactive runs where stderr is
  captured to a log file. Each line carries a wall-clock timestamp,
  cycle/total counter, completion fraction, elapsed and ETA, and
  (optionally) cumulative per-pair swap-acceptance rates. Cadence is
  "every `interval` cycles plus the final cycle"; the elapsed/ETA
  clock resets at the start of each `run()` so reusing one printer
  across multiple runs produces a fresh clock per run. Output stays
  single-line for wide ladders, and the `H:MM:SS` shape is stable
  past 24 hours.
- `examples/06_progress_monitoring.py` — worked example showing
  `ProgressPrinter` attached to a short canonical PT run.

## [0.5.0] - 2026-04-30

### Added

- `CanonicalParallelTempering` and `ProcessPool` accept `atoms` /
  `initial_atoms` as a single `Atoms` (broadcast to every replica,
  the existing behaviour) or a `Sequence[Atoms]` (one per
  temperature, length-validated). Geometry consistency
  (cell/positions/pbc) is validated at construction time.
  Closes #4.
- `BaseParallelTempering.final_configurations() -> list[Atoms]`
  — the current structure at each temperature position,
  reconstructed from a stored template and per-replica occupation
  vectors. Enables seeding a follow-up run from a completed run's
  equilibrated configurations.
- `ExchangeHistory.concatenate(*histories)` — combine sequential
  histories for multi-segment diagnostics. Stacks energies and
  replica labels along the cycle axis (dropping the pre-run
  snapshot from all but the first history), sums swap counts
  element-wise. Validates replica-count consistency.

## [0.4.0] - 2026-04-29

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
