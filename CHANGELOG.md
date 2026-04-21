# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — prior to first public release

This release supersedes the unreleased v0.1.1 patch. The following are
not backwards-compatible changes but also not releases against prior
versions; they shape v0.1.0 before first publication.

- Replaced `Backend` / `SerialBackend` / `ProcessBackend` with a pair of
  protocols, `ReplicaPool` (core operations) and `ObservablePool`
  (extends with `attach_observer`). Pool implementations are
  `SerialPool` (satisfies `ObservablePool`) and `ProcessPool`
  (satisfies only `ReplicaPool`). Single source of truth for replica
  state: the pool owns it, the orchestrator routes through the pool.
- `CanonicalParallelTempering`: `backend=` kwarg renamed to `pool=`.
- Removed `pt.replicas` attribute; use `pt.pool` (and methods on it
  like `pt.pool.current_energy(i)`, `pt.pool.current_occupations(i)`,
  `pt.pool.data_containers()`) for per-replica access.
- `ProcessPool.attach_observer` is not supported in v0.1 (workers can't
  safely receive pickled `BaseObserver` instances). Expressed in the
  type system via the `ObservablePool` split; calling
  `pt.attach_observer(...)` on a process-based pool raises `TypeError`
  with a clear message pointing at `SerialPool`.
- `CanonicalParallelTempering` now validates temperatures are
  non-decreasing and `block_size >= 1` at construction.
- `BaseParallelTempering.run()` wraps its cycle loop in try/finally so
  `self.history` is assigned even on mid-run exception.
- `_log_prob_ratio` raises `RuntimeError` with cycle/pair/energy
  context if a non-finite log ratio is computed (previously silently
  flowed through `metropolis_accept`).

## [0.1.0] - 2026-04-21

### Added

- `BaseParallelTempering` abstract orchestrator and concrete
  `CanonicalParallelTempering` subclass.
- Serial and multiprocessing-backed replica advance.
- Per-replica `mchammer.BaseObserver` attachment (pass-through).
- `ExchangeCallback` protocol plus `ExchangePrinter` and
  `SwapRateTracker` built-ins.
- `ExchangeHistory` with HDF5 read/write, bundled alongside
  per-replica `mchammer.BaseDataContainer` outputs.
- Round-trip-rate and energy-autocorrelation-time diagnostics.
- Worked examples and CI on Python 3.11 through 3.14.
