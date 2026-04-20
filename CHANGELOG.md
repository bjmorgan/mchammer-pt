# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-21

### Added

- `BaseParallelTempering` abstract orchestrator and concrete
  `CanonicalParallelTempering` subclass.
- Serial and multiprocessing-backed replica advance.
- Per-replica `mchammer.BaseObserver` attachment (pass-through).
- `ExchangeCallback` protocol plus `ExchangeLogger` and
  `SwapRateTracker` built-ins.
- `ExchangeHistory` with HDF5 read/write, bundled alongside
  per-replica `mchammer.DataContainer` outputs.
- Round-trip-rate and energy-autocorrelation-time diagnostics.
- Worked examples and CI on Python 3.11 through 3.14.
