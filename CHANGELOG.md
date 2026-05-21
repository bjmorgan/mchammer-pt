# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ``mchammer_pt.dos`` module with ``stitch_entropy``,
  ``reweight_canonical_from_dos``, and the ``KB_EV`` constant for
  generic Wang-Landau DOS post-processing. Both helpers are also
  re-exported from the package root.
- ``mchammer-pt-reweight`` console script for canonical reweighting
  from a stitched DOS CSV, with optional ``--plot`` output.

## [0.11.0] - 2026-05-21

### Added

- Checkpoint and resume support for REWL windows with
  ``n_walkers_per_window > 1``. ``save_checkpoint``,
  ``CheckpointWriter``, ``attach_checkpoint_writer``, ``resume``
  and ``resume_process_pool`` all now work for multi-walker
  windows on both ``SerialWangLandauPool`` and
  ``ProcessWangLandauPool``.
- ``WangLandauWindowGroup`` gains ``snapshot_for_checkpoint``
  and ``restore_state`` methods plus a public ``exchange_idx``
  property.
- ``UserWarning`` emitted on resume of a checkpoint with any
  window-group of ``W > 1`` walkers, documenting that the
  resumed trajectory is structurally correct but not
  bit-identical to an uninterrupted run because of the
  destructive end-of-run entropy merge.
- Phase-consistency check at read time: a corrupted checkpoint
  whose ``/orchestrator/window_groups/<g>/phase`` disagrees
  with any walker's ``_last_state["phase"]`` raises
  ``ValueError`` before any pool state is restored.
- Corruption-robustness guards on resume: explicit ``ValueError``
  (rather than ``IndexError`` deep in reconstruction) when
  ``walkers_per_window`` is inconsistent with ``windows``, when
  the per-replica container count is inconsistent with
  ``sum(walkers_per_window)``, or when a multi-walker
  ``exchange_idx`` falls outside ``[0, n_walkers)``.

### Changed

- **Breaking: checkpoint on-disk schema bumped from v3 to v4.**
  v4 readers refuse v3 files with a message pointing at
  ``mchammer-pt`` 0.9.0 as the last v3-capable release. v4 adds
  ``walkers_per_window`` to ``/meta`` and an
  ``/orchestrator/window_groups/<g>/`` subgroup carrying
  per-window-group exchange RNG, ``exchange_idx``, and ``phase``
  for windows with ``walkers_per_window[g] > 1``. W = 1 windows
  omit the subgroup, so an all-W=1 file is byte-equal to the v3
  layout aside from the meta key and the schema version string.
- WL pool ``snapshot_for_checkpoint`` returns a structured dict
  ``{"per_walker": [...], "group_state": [...]}`` rather than a
  flat list; ``restore_replica_state`` signature becomes
  ``(containers, per_walker_extras, group_state)``. Canonical
  PT pools are unchanged.
- WL pool ``data_containers()`` now returns a flat per-walker
  list of length ``M = sum(walkers_per_window)`` (was one entry
  per slot).
- Same-pool W = 1 resume retains the bit-identical contract;
  same-pool W > 1 resume is structurally correct but not
  bit-identical (see ``UserWarning`` above).

### Removed

- The ``n_walkers_per_window > 1`` + ``data_container_file``
  rejection guard from ``WangLandauParallelTempering`` and the
  ``process_pool`` classmethod.
- The private ``_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED``
  constant and the tests that pinned it.

## [0.10.1] - 2026-05-21

### Changed

- ``WangLandauReplica`` no longer seeds the placed bin into the
  ensemble's ``_entropy`` dict at construction, ``set_occupations``,
  or ``restore_state`` — only the matching ``_histogram`` seed
  remains. This restores ``_update_entropy``'s min-shift floor to
  the lowest *visited* bin (matching the upstream icet algorithm)
  while keeping the flatness gate aware of placed bins via
  ``_histogram``. The 0.10.0 side effect — newly-visited bins
  deep-trapping the walker until their entropy caught up to the
  un-shifted accumulated value of older bins — is gone.
- ``merge_entropies`` now filters out walkers whose ``_entropy``
  dict is empty (previously a placed-but-not-yet-stepped walker
  carried a single zero-valued entry from the seed and survived
  the filter). In practice ``merge_entropies`` only runs at
  halving or end-of-run, by which point every walker has stepped,
  so the change is observable only in tests that call it directly
  on a freshly-constructed walker.

### Notes

- Runs resumed from 0.10.0 checkpoints carry an
  ``_entropy``-at-the-starting-bin entry of 0.0 forward; the
  min-shift continues to see that bin at 0 for the remainder of
  the resumed run. Fresh runs in 0.10.1 use the corrected
  semantics.

## [0.10.0] - 2026-05-20

### Added

- ``CoordinatedWangLandauEnsemble._visited_bins``: set of bins the
  walker has reached via ``_update_entropy`` since window entry.
  Persisted across ``refresh_last_state`` / ``restore_state``
  round-trips.
- ``bins_visited`` and ``bins_known`` keys on the dict returned by
  ``WangLandauReplica.window_stats`` and
  ``WangLandauWindowGroup.window_stats``. ``bins_visited`` is
  ``len(_visited_bins)`` (or the union across walkers);
  ``bins_known`` is ``len(_histogram)`` (or the union across
  walkers).
- ``WangLandauProgressPrinter`` reports ``bins (vis/known)`` (e.g.
  ``14/15``) in place of the previous ``bins_visited`` column.

### Changed

- ``WangLandauReplica`` seeds the bin the walker is placed at
  (``bin_init`` at construction, ``new_bin`` on ``set_occupations``
  and ``restore_state``) into the underlying ensemble's
  ``_histogram`` (count 0) and ``_entropy`` (value 0.0) via
  ``setdefault``. Placed bins appear in ``_histogram`` and
  contribute to the Wang-Landau flatness gate.
- ``_summed_histogram_flat_from_snapshots`` and
  ``WangLandauReplica.is_flat`` return ``False`` when
  ``mean(counts) <= 0``, so an all-zero histogram is not flat.

## [0.9.0] - 2026-05-19

### Added

- ``flatness_mode`` and ``merge_cadence`` parameters on
  ``WangLandauParallelTempering``, ``from_bin_count``, and
  ``process_pool`` factories. ``flatness_mode="per_walker"`` halves
  when every walker is independently flat (published Vogel et al.
  2013); ``flatness_mode="pooled"`` (default) halves when the
  summed histogram across walkers is flat (pooled bins see ``W×``
  the samples per bin under the same wall-clock budget).
  ``merge_cadence="at_halve"`` (default) merges entropies at each
  collective halve; ``merge_cadence="never"`` skips mid-run merges
  entirely (walker entropies diverge during the run; end-of-run
  finalisation reconciles them). Both apply only in the halving
  phase. The 1/t phase never merges mid-run regardless of cadence.
- ``CoordinatedWangLandauEnsemble``, a subclass of mchammer's
  ``WangLandauEnsemble`` with internal halving suppressed. Halving
  is now driven by the pool-level coordinator after each block.
  ``WangLandauReplica`` defaults ``ensemble_cls`` to this subclass
  and rejects any ``ensemble_cls`` that is not a subclass of it.
- ``WangLandauPool.finalise_for_reporting()`` protocol method,
  with matching implementations on ``SerialWangLandauPool`` and
  ``ProcessWangLandauPool``. Called from
  ``WangLandauParallelTempering.run()`` on every exit path
  (full completion, early convergence, and exceptions including
  ``KeyboardInterrupt``). Merges per-walker entropies into a
  single window estimate via ``merge_entropies``, writes the
  merged dict to every walker's ``_entropy``, and refreshes
  ``_last_state`` so downstream readers (``WindowResult``, data
  containers) see a consistent estimate regardless of which
  walker they sample from. No-op for single-walker windows.
- ``FINALISE_MERGE`` worker opcode for the process pool. Receives
  a merged entropy dict, writes it to ``_entropy``, and refreshes
  ``_last_state`` in one round-trip.
- ``mchammer_pt/wl_coordinator.py`` module owning the collective
  Wang-Landau policy as pure data + a pure function. Carries
  ``WalkerPostBlockState``, ``SlotView``, ``CoordinatorPlan``,
  ``FlatnessMode``, ``MergeCadence``, ``Schedule``, ``Phase``,
  and ``decide_block_actions(SlotView) -> CoordinatorPlan``.
- ``mchammer_pt/parallel/_builder.py`` module with frozen
  dataclasses ``AtomsSpec``, ``CanonicalBuilder``, and
  ``WLBuilder``. Each Builder carries the inputs required to
  construct one replica and exposes a ``build()`` method;
  ``AtomsSpec`` holds the four numpy-array fields required to
  reconstruct an ``ase.Atoms`` across the spawn boundary, deeply
  immutable (arrays copied + marked non-writeable in
  ``from_atoms``).
- ``docs/architecture.md`` — developer-facing overview of the
  REWL runtime (Pool → Slot → Walker → Ensemble model, the
  four-phase ``advance_all`` pipeline shared across backends).
  Linked from the README.

### Changed

- Multi-walker windows now use a collective halving gate with two
  selectable modes (per-walker vs pooled — see above) plus
  collective halving via ``force_halve``. This replaces the
  previous per-walker autonomous halving with force-halving of
  laggards. Halvings fire on the collective gate's verdict
  rather than when the fastest walker individually flattens.
- ``merge_entropies`` rewritten to rebase each walker's entropy by
  the mean over the intersection of bins all walkers visited,
  then average bin-wise over the walkers that visited each bin,
  then shift so ``min(merged) == 0``. The previous naive arithmetic
  mean combined values with different additive constants and
  treated bins visited by only some walkers as zero-valued in the
  others, producing a curve with coverage-boundary artefacts.
  Intersection-mean rebasing preserves the shape across partial
  coverage.
- The collective-halving policy is expressed as one pure function
  ``decide_block_actions(SlotView) -> CoordinatorPlan`` in the new
  ``wl_coordinator`` module. Both ``SerialWangLandauPool.advance_all``
  and ``ProcessWangLandauPool.advance_all`` follow the same
  four-phase shape: advance walkers, collect per-walker snapshots,
  decide (shared), apply (backend-specific). Previously the policy
  was duplicated across a coordinator block on
  ``WangLandauWindowGroup`` (serial) and a separate ``_compute_plan``
  function on ``processes.py`` (process pool); the duplication is
  gone.
- Single-walker windows use bare ``WangLandauReplica`` slots
  directly. The ``WangLandauWindowGroup`` wrap is now applied only
  to windows with ``n_walkers_per_window > 1``. The
  ``WangLandauSlot`` Protocol widened with ``walker_states``,
  ``apply_plan``, ``reroll_exchange_idx``, ``phase``, ``schedule``,
  and ``flatness_limit`` so the bare replica satisfies it. The
  flatness check fires at block boundaries (after every
  ``advance_all``) rather than at the ensemble's
  ``flatness_check_interval`` step boundaries, on both single- and
  multi-walker paths.
- ``flatness_mode`` and ``merge_cadence`` now live on the pool
  rather than on the window group. ``SerialWangLandauPool.__init__``
  gained both as keyword arguments (already present on
  ``ProcessWangLandauPool``). ``WangLandauWindowGroup.__init__`` no
  longer accepts them; the constructor now also validates that all
  walkers in the group share the same ``schedule`` and
  ``flatness_limit`` as well as the existing energy-window /
  spacing checks. ``WangLandauWindowGroup`` now requires
  ``n_walkers >= 2`` and raises if constructed with a single
  walker (use a bare ``WangLandauReplica`` instead).
- Worker construction in ``mchammer_pt/parallel/_worker.py``
  separates build inputs from runtime state. ``BaseWorker.__init__``
  takes a ``Builder`` (``CanonicalBuilder | WLBuilder | None``)
  and a ``reply_sink``; ``_build_replica`` delegates to
  ``self._builder.build()``. The subclass ``__init__`` and
  ``_build_replica`` overrides are gone; ``WangLandauWorker``
  registers its REWL opcodes via a ``_register_extra_handlers``
  hook called from ``BaseWorker.__init__``. ``CanonicalWorker`` is
  now a near-empty subclass with only the test-construction
  ``for_replica`` classmethod. Process-pool entry points
  ``_worker``/``_wl_worker`` take a single ``Builder`` argument
  rather than individual fields.
- Process-pool window state is now exposed through a
  ``ProcessWangLandauWindow`` class with coordinator-facing
  methods (replacing the previous private ``_WindowSlot``
  dataclass). Worker opcodes ``GET_ENTROPY``, ``SET_ENTROPY``,
  ``FORCE_HALVE``, ``SET_PHASE``, and the new ``FINALISE_MERGE``
  carry the collective coordination state. The ``ADVANCE`` ack now
  piggybacks ``is_flat``, ``fill_factor``, ``entropy``, ``step``,
  ``window_entry_step``, ``histogram``, and
  ``reached_energy_window`` so the coordinator can decide pooled
  flatness without further round-trips.
- The WL progress reporter's ``flat_min`` column reports the
  quantity the active halve gate is checking: pooled
  ``min(H_summed)/mean(H_summed)`` under ``flatness_mode="pooled"``,
  the minimum over walkers of ``min(H_k)/mean(H_k)`` under
  ``"per_walker"``. Single-walker windows fall through to the
  pooled computation, which is exact for ``n_walkers == 1``.

### Fixed

- The 1/t phase no longer merges per-walker entropies on every
  block. The 1/t schedule has no flatness gate, so there is no
  natural sync point during the run; mid-block merging mixed
  walker noise unnecessarily. End-of-run merging via
  ``finalise_for_reporting`` reconciles divergent walker
  entropies on every exit path.
- ``WangLandauParallelTempering.resume`` now produces slot objects
  that the pool-level coordinator drives through
  ``decide_block_actions``. Before the fix, resumed runs used bare
  ``WangLandauReplica`` instances that did not participate in the
  coordinator; with the default ``CoordinatedWangLandauEnsemble``
  (halving-suppressed) they silently produced no halving and never
  converged.
- ``_fill_factor_history`` no longer records the Belardinelli-
  Pereyra phase transition at the same step as the halve. The
  dict now records halve events only, restoring symmetry with
  ``_entropy_history`` so downstream analysis that pairs the two
  dicts sees coherent post-halve state at each key.
- ``BaseWorker._build_replica`` raises ``RuntimeError`` (rather
  than ``assert``) when called on a worker constructed with
  ``builder=None`` and no externally-set ``_replica``. ``assert``
  is stripped under ``python -O``; the user-facing failure mode
  would otherwise have been a stripped attribute error on
  ``None.build()``.

### Performance

- Process-pool ``advance_all`` broadcasts ``ADVANCE`` across every
  worker in every window in a single fan-out, then runs the
  coordinator decisions in a separate pure phase, then batches
  follow-up commands (``FORCE_HALVE`` / ``SET_ENTROPY`` /
  ``SET_PHASE``) across slots via ``broadcast_gather`` /
  ``fanout_gather``. Previously ``advance_all`` ran each window's
  MC and follow-up commands serially, costing a factor of
  ``len(slots)`` in wall-clock per block.

### Notes

- Output APIs (``WindowResult.get_entropy``, ``get_histogram``,
  ``WangLandauParallelTempering.results``) are unchanged. Pool
  constructors gained kwargs but the orchestrator-level
  (``WangLandauParallelTempering``) public surface is unchanged.
- Multi-walker checkpointing remains unsupported.

## [0.8.0] - 2026-05-14

### Added

- Single-walker replica-exchange Wang-Landau (REWL) on top of
  icet's ``WangLandauEnsemble``. Each replica owns a fixed energy
  window; adjacent windows attempt configuration swaps between
  cycles using a within-window log-density-of-states ratio for
  acceptance. The base ``WangLandauEnsemble`` from mainline icet is
  the default; pass ``ensemble_kwargs={'schedule': '1_over_t'}``
  to use the Belardinelli-Pereyra 1/t schedule.
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
  entry point and REWL-specific opcodes (``LOG_G_AT``,
  ``CONVERGED``, ``WL_STATS``) on top of the canonical-shared set.
- Observer attach for REWL pools. ``SerialWangLandauPool`` and
  ``ProcessWangLandauPool`` now satisfy a new
  ``WangLandauObservablePool`` protocol (mirrors
  ``ObservablePool`` for canonical) with ``attach_observer``,
  ``attach_observer_class``, ``attach_observer_factory``, and
  ``get_observers``. Use this to record per-step observable
  trajectories during a REWL run, then feed the per-replica data
  containers into icet's ``get_average_observables_wl`` for
  thermodynamic averages against the stitched density of states.
- ``WangLandauProgressPrinter`` built-in ``CycleCallback`` for
  monitoring REWL runs. Emits every ``interval`` cycles: a header
  line (timestamp, cycle, elapsed, ETA, swap rates) followed by a
  compact table with one row per window reporting fill factor,
  number of WL halvings, bins visited, histogram flatness
  (``min(H)/mean(H)`` — compare against your ``flatness_limit``),
  and convergence status. Reads from live ensemble state via a new
  ``pool.per_window_stats()`` method, so values are always current
  regardless of ``ensemble_data_write_interval``.
- A ``slow``-marked integration test
  (``tests/integration/test_rewl_2d_ising.py``) exercising REWL
  end-to-end on a 4x4 2D Ising fixture. After convergence, the
  stitched ln g(E) curve from ``get_density_of_states_wl`` is
  compared against the exact density of states obtained by
  brute-force enumeration over all 2^16 configurations; the
  residual is constrained in both standard deviation and maximum
  deviation.
- ``examples/08_rewl.py`` — self-contained REWL example on the 4x4
  2D Ising model: window construction, per-window seeding, a
  convergence run with ``WangLandauProgressPrinter``, and DOS
  stitching via ``get_density_of_states_wl``.

### Changed

- The 1/t schedule is now selected via
  ``ensemble_kwargs={'schedule': '1_over_t'}`` rather than by
  passing a separate ``ensemble_cls``. This follows icet's
  refactoring of the Belardinelli-Pereyra schedule into the base
  ``WangLandauEnsemble`` class.
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
