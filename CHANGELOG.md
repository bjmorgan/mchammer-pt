# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  Monte Carlo driver so co-tenant replicas evolve independently.
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
