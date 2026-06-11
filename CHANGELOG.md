# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0] - 2026-06-11

### Added

- ``one_over_t_entry`` keyword on ``WangLandauParallelTempering`` (also on
  ``from_bin_count`` and ``process_pool``; default ``"window_clock"``):
  selects how a window's fill factor enters the Belardinelli-Pereyra 1/t
  phase at the switch. The default ``"window_clock"`` reproduces today's
  behaviour bit-for-bit (f jumps to ``1/(t since window entry)``).
  ``"f_continuous"`` starts the 1/t clock from the fill factor halving
  actually reached, so f is continuous across the switch; this applies
  uniformly at every switch path (canonical crossing and stall escape,
  coupled and decoupled) and is orthogonal to ``one_over_t_gate``. On the
  stall-escape path this removes the f cliff (factors of 40-3000 observed
  in production data) that froze pre-switch entropy errors into the 1/t
  regime. Selecting ``"f_continuous"`` without
  ``ensemble_kwargs={"schedule": "1_over_t"}`` raises at construction,
  since no switch ever fires under the halving schedule. The policy is
  recorded in checkpoint metadata and adopted from there on resume; the
  per-walker schedule-clock origin round-trips through the checkpoint,
  and checkpoints written before this feature restore and continue with
  the pre-feature behaviour. Restoring per-walker state across entry
  policies (an f-continuous container into a window-clock replica, or a
  1/t-phase window-clock container into an f-continuous replica) raises
  rather than silently switching the f schedule. Threaded through the
  serial and process-pool backends.
- Read-only ``one_over_t_entry`` property on ``SerialWangLandauPool``,
  ``ProcessWangLandauPool``, ``WangLandauWindowGroup``, and the
  ``WangLandauPool`` protocol. As with the policy properties added in
  0.21.0, the pool-held value is the single source of truth when an
  explicit ``pool=`` is passed to ``WangLandauParallelTempering``: the
  orchestrator adopts it and records it in checkpoint metadata, so a
  pool built with f-continuous walkers cannot checkpoint
  ``window_clock``. The serial pool derives the value from its replicas
  and rejects construction from slots with mixed entry policies.

## [0.21.0] - 2026-06-09

### Added

- ``one_over_t_gate`` keyword on ``WangLandauParallelTempering`` (also on
  ``from_bin_count`` and ``process_pool``; default ``"visit_once"``):
  selects the gate that controls when a window halves its fill factor and
  switches to the Belardinelli-Pereyra 1/t schedule. The default
  ``"visit_once"`` reproduces today's 1/t-schedule behaviour (a visit-once
  halving gate coupled to the switch). ``"flatness"`` selects a flatness
  halving gate (reusing ``flatness_limit`` as the threshold) bundled with a
  decoupled, stall-safe 1/t switch evaluated every block. Selecting
  ``"flatness"`` without ``ensemble_kwargs={"schedule": "1_over_t"}`` raises
  at construction, since the gate is inert under the halving schedule.
- ``bp_stall_multiple`` keyword on ``WangLandauParallelTempering`` (also on
  ``from_bin_count`` and ``process_pool``; default ``4.0``): the multiple of
  the first-stage duration after which a stalled window (one that has halved
  at least once but cannot meet the flatness gate) adopts the 1/t schedule.
  Only consulted under ``one_over_t_gate="flatness"``.
- Read-only ``flatness_mode``, ``merge_cadence``, ``one_over_t_gate``, and
  ``bp_stall_multiple`` properties on ``SerialWangLandauPool`` and
  ``ProcessWangLandauPool`` (and the ``WangLandauPool`` protocol). When an
  explicit ``pool=`` is passed to ``WangLandauParallelTempering`` these
  pool-held values are the single source of truth: the orchestrator adopts
  them (and records them in checkpoint metadata) in preference to its own
  policy keyword arguments, which then only build the default pool when
  ``pool`` is ``None``. A caller therefore sets each policy once, on the
  pool, without repeating it to the orchestrator.

## [0.20.0] - 2026-06-08

### Added

- ``dos_snapshot_ratio`` keyword on ``WangLandauParallelTempering`` (also
  on ``from_bin_count`` and ``process_pool``; default ``2.0``, ``None``
  disables): records per-walker ``ln g(E)`` snapshots during the
  Belardinelli-Pereyra 1/t regime on a fill-factor rung ladder
  (``rung(f) = floor(log(1/f) / log(ratio))``; at ``2.0`` a snapshot each
  time ``f`` halves). The snapshots live in a per-walker store kept
  separate from the halving history, so the collective halving count is
  unchanged, and they survive checkpoint and resume with the ladder
  reconstructed from the restored fill factors. The parameter is recorded
  in checkpoint metadata and threaded through the serial and
  process-pool backends.
- ``WindowResult.get_entropy(fill_factor_limit=...)`` now reconstructs the
  density of states at fill factors below the last halving by reading the
  1/t snapshot store (it previously returned ``None`` there, leaving the
  1/t regime invisible), enabling a convergence-versus-run-length study
  across the whole run.

## [0.19.0] - 2026-06-06

### Added

- ``WangLandauParallelTempering`` multi-walker windows now exchange via a
  random matching of walkers at each window boundary, replacing the single
  representative-walker swap, so the per-walker exchange rate no longer
  dilutes as ``n_walkers_per_window`` increases. Exchange acceptance is
  evaluated parent-side from a per-walker (energy, density-of-states)
  cache and accepted swaps are applied in one batched IPC round, so the
  process backend issues no per-exchange round trips. Replica labels are
  position-indexed over the total walker count and carried on
  ``ExchangeHistory``, so multi-walker round-trip counts survive a read
  back from disk. Canonical parallel tempering is single-walker per rung
  and is unchanged: the matching degenerates to the single pair with no
  extra RNG draws, so its exchange stream, labels, and round-trip counts
  are identical.

### Changed

- The checkpoint schema is bumped to 5, dropping the per-window exchange
  index. Single-walker schema-4 checkpoints still load; older
  multi-walker schema-4 checkpoints are rejected with a clear error.

### Removed

- The per-window representative-walker exchange machinery, superseded by
  the walker-matching exchange.

## [0.18.0] - 2026-06-04

### Added

- ``mchammer_pt.seed_window_configs`` and ``SeedSearchParams``: a
  public, material-agnostic search that fills each REWL energy window
  with K distinct, in-band starting configurations, ready for
  ``WangLandauParallelTempering.process_pool(atoms=...,
  n_walkers_per_window=...)``. Each window is anchored to the nearer
  energy end (ground state below, a caller-supplied random fill above)
  and driven into the band by ``CustomWangLandauEnsemble``'s built-in
  window search. The ground state, random fill, and move set are all
  caller-supplied inputs; the search contains no chemistry. Runs a
  spawn ``concurrent.futures.ProcessPoolExecutor`` of independent
  confined walks and raises a clear error naming any window the search
  cannot fill with enough distinct in-band configurations within the
  walk budget (and surfaces a worker that dies, e.g. an out-of-memory
  kill, as a ``RuntimeError`` rather than hanging).

- ``mchammer-pt-stitch`` gains a ``--multi-run`` flag that merges N
  independent run checkpoints (one per run) into a single consensus
  density of states: each energy window is merged across the runs, then
  stitched once with the existing stitcher. A checkpoint passed more than
  once is weighted by its multiplicity. Every run must cover the same
  windows on the same energy grid and yield entropy for every kept
  window; mismatched window bounds or ``energy_spacing``, or a run with no
  data for a kept window, are reported as errors. Per-pair overlap
  standard deviations are written to stderr as a join diagnostic. The DOS
  CSV schema (``energy``, ``entropy``) and single-checkpoint /
  ``--containers`` behaviour are unchanged.

## [0.17.0] - 2026-06-03

### Added

- ``CoexistencePoint.n_self_consistent_iter`` and
  ``CoexistencePoint.self_consistent_converged`` fields, exposing the
  number of self-consistent E_star refinement iterations performed at
  the final brentq evaluation and whether that refinement converged.
- ``smoothing_sigma`` keyword on ``find_phase_split`` (default
  ``0.0``), which applies a Gaussian smoothing of ``ln g(E)`` along
  the energy axis before locating the phi-minima. Opt-in for direct
  callers; the coexistence top-level entry point enables it by
  default.
- ``smoothing_sigma`` (default ``2.0``),
  ``max_self_consistent_iter`` (default ``20``), ``damping``
  (default ``0.5``), and ``self_consistent_tol_K`` (default
  ``1e-3``) keywords on ``equal_area_temperature``, controlling the
  one-shot ``ln g`` smoothing and the ``(Tc, E_star)``
  self-consistency loop respectively.
- ``mchammer-pt-coexistence`` gains ``--smooth-sigma`` (forwarded as
  ``smoothing_sigma``) and ``--no-self-consistent`` (sets
  ``max_self_consistent_iter=0``, freezing ``E_star`` at its initial
  seed value) command-line flags.
- ``find_phase_split`` and ``equal_area_temperature`` accept
  ``ln g(E)`` curves containing ``-inf`` entries (``g = 0``, the
  forbidden energies of a discrete spectrum) as zero-weight bins,
  while rejecting ``NaN`` and ``+inf``. ``equal_area_temperature``
  additionally requires the ``energy`` column to lie on a uniform
  grid. This supports complete-histogram DOS input that records
  unreachable energies explicitly rather than omitting their rows.
- ``WangLandauProgressPrinter`` gains an opt-in ``per_walker_detail``
  argument (``bool | Sequence[int]``) that expands selected
  multi-walker windows into one sub-row per walker, each showing that
  walker's ``filled/known`` coverage and ``flat_min``.
- ``per_window_stats()`` now reports ``bins_filled``: the number of
  bins with a positive count in the current histogram (the union of
  per-walker positive bins under pooled flatness, the intersection
  under per-walker flatness), which resets on each halving.
- ``WangLandauParallelTempering`` and the Wang-Landau pools gain a
  ``recency_visits_per_bin`` parameter (default ``1000``) setting the
  timescale of a per-window recency-flatness diagnostic. Each walker
  keeps an exponentially-weighted estimate of recent visits per energy
  bin (decay constant roughly ``recency_visits_per_bin * N_bins`` MC
  steps), and ``per_window_stats()`` now reports ``recency_flatness``
  (``min / mean`` over known bins, pooled or per-walker by
  ``flatness_mode``) and ``schedule``. The parameter is recorded in the
  checkpoint and adopted from there on resume; the estimate itself is
  not persisted, so the diagnostic re-accumulates after a resume,
  reading ``--`` only until the first recent visit.
- ``WangLandauParallelTempering`` (including ``from_bin_count`` and
  ``process_pool``) and ``ProcessWangLandauPool`` accept per-walker
  initial structures. Each per-window ``atoms`` entry may now be either
  a single ``Atoms`` (broadcast to every walker in that window, as
  before) or a ``Sequence[Atoms]`` of length the window's
  ``n_walkers_per_window`` (one starting structure per walker, in walker
  order); windows may mix the two forms. This lets the walkers in a
  window start from independent or symmetry-paired configurations so
  their merged density of states is not biased toward a single basin in
  quasi-ergodic systems. Each structure's energy must lie inside its
  window, and a bare ``Atoms`` for the whole argument remains rejected.

### Deprecated

- ``CoexistencePoint.n_iterations`` is a backward-compatibility
  ``@property`` alias for ``n_brentq_iterations`` and emits a
  ``DeprecationWarning`` on access. Use ``n_brentq_iterations``
  directly.

### Changed

- The ``WangLandauProgressPrinter`` flatness column is renamed from
  ``flat_min`` to ``flatness`` and is now schedule-aware: it shows the
  cumulative gate flatness ``min(H) / mean(H)`` for a halving-schedule
  run, and the EWMA recency flatness for a ``1/t``
  (Belardinelli-Pereyra) run, whose cumulative histogram does not
  reset. Per-walker detail sub-rows continue to show the gate
  ``flat_min``.
- ``mchammer_pt.analysis.dos.stitch_entropy`` now returns a complete
  histogram on the ``energy_spacing`` grid: every integer-multiple bin
  from the lowest to the highest populated energy is emitted, with
  interior bins that no window reached carried as ``entropy = -inf``
  (``g = 0``) instead of being dropped. The output grid is therefore
  uniform and self-describing, so downstream tools recover the true bin
  width from ``energies[1] - energies[0]`` even across forbidden-energy
  gaps in a discrete spectrum. The window merge, additive shifting, and
  overlap-error reporting are unchanged; the ``-inf`` fills are added
  only for unpopulated interior positions, and no frontier
  extrapolation is done beyond the populated range. The rebase to
  ``min = 0`` uses the minimum over finite entries, so a window that
  already carries ``-inf`` (``g = 0``) bins -- e.g. a re-stitched
  complete histogram -- no longer corrupts the offset.
- ``mchammer-pt-coexistence`` no longer re-validates the DOS grid in
  the CLI. The row-count, finiteness, and uniform-grid checks are
  delegated to ``equal_area_temperature`` (which accepts ``-inf``
  ``g = 0`` bins), removing a duplicated check whose copy wrongly
  rejected the ``-inf`` bins of a complete-histogram DOS.
- ``mchammer_pt.analysis.coexistence.equal_area_temperature`` is
  re-architected around a fixed dividing energy with an outer
  self-consistency loop, replacing the previous design that
  recomputed the phase split at every trial temperature inside the
  root-finder. The new flow smooths ``ln g(E)`` once (controlled by
  ``smoothing_sigma``) to stabilise the topology detection, seeds
  ``E_star`` at the heat-capacity peak inferred from the stitched
  DOS, runs ``scipy.optimize.brentq`` on the equal-area imbalance
  with that ``E_star`` held fixed, then refreshes ``E_star`` at the
  resulting Tc and iterates the ``(Tc, E_star)`` pair under damped
  updates until the temperature change between successive
  iterations falls below ``self_consistent_tol_K``. Detecting the
  split once on a smoothed ``ln g`` — rather than at every trial
  temperature on the raw curve — makes the pipeline robust to
  bin-scale shot-noise dimples in real REWL output, which produce
  spurious local minima of ``phi(E) = beta * E - ln g`` that would
  otherwise be selected as phase peaks and collapse the bracket.
  The auto-bracket helper is replaced by ``_walk_for_sign_change``,
  which integrates the raw fixed-``E_star`` imbalance outward from
  the Cv-peak seed until the sign flips; window-edge exits raise an
  informative diagnostic distinct from the unimodal-midpoint case.
- A user-supplied ``T_bracket`` whose midpoint temperature is
  unimodal under the smoothed ``ln g`` now raises
  ``NotBimodalError`` rather than ``NoBracketError``. The
  diagnostic identifies the root cause (the bracket does not
  enclose a bimodal region) rather than reporting a generic
  bracket failure.
- The inner T root-finder is ``scipy.optimize.brentq``, exploiting
  inverse-quadratic interpolation for faster convergence than
  plain bisection at no API cost. ``scipy`` is declared as an
  explicit dependency; it was already pulled in transitively by
  ``ase`` and ``icet``. ``CoexistencePoint.n_bisection_steps`` is
  renamed to ``CoexistencePoint.n_brentq_iterations`` to reflect
  that the iteration count mixes bisection and
  inverse-quadratic-interpolation steps; the CLI's JSON/CSV output
  column is renamed consistently.
- The REWL progress table's bin column now shows ``bins (fill/known)``
  -- coverage since the last halving -- in place of the monotone
  ``bins (vis/known)``, so the displayed number tracks the halving
  gate rather than pinning at its maximum once a window is spanned.
  ``bins_visited`` is retained in ``per_window_stats()`` as a separate
  monotone statistic.

### Fixed

- ``mchammer_pt.analysis.coexistence.find_phase_split`` previously
  located phase peaks as local maxima of ``ln g(E)``, an assumption
  that holds only for synthetic bimodal-``ln g`` DOS shapes. For any
  real lattice system, ``ln g(E)`` is monotonically increasing in
  energy (combinatorial expansion of configurations), so the helper
  raised ``NotBimodalError`` at every trial temperature and the
  estimator was unusable on real REWL output. Phase peaks are now
  correctly located as the two dominant local minima of
  ``phi(E) = beta * E - ln g(E)`` (peaks of ``P(E | T)``), which is
  temperature-dependent. The reported peak and dividing-energy
  positions are bin centres of the energy grid, and the dividing
  energy is the highest *populated* ``phi`` bin between the peaks, so
  a ``g = 0`` bin in a forbidden-energy gap is never selected as the
  saddle.
- Phase-peak detection (``_two_dominant_peak_indices``) now requires
  both neighbours of a candidate ``phi`` minimum to be populated
  (finite). A populated bin beside a ``g = 0`` gap (``phi = +inf``)
  cleared its gap-side neighbour trivially and could masquerade as a
  phase peak, even though it is only the edge of the populated region
  against the forbidden-energy wall. On a complete-histogram DOS whose
  low-energy spectrum is fragmented by forbidden gaps, this produced a
  spurious low-energy split (two same-side peaks, a negative
  ``barrier_height``, and a collapsed ``T_c``) under automatic Cv-peak
  seeding. Excluding gap-adjacent minima recovers the genuine
  inter-phase split. Two energy bins separated *only* by forbidden
  ``g = 0`` bins (disconnected spectra, no populated barrier between
  them) are correspondingly no longer treated as a coexistence pair.

## [0.16.0] - 2026-05-28

### Fixed

- ``mchammer_pt.__version__`` is now derived from the installed
  package metadata (``importlib.metadata``) rather than a hardcoded
  string, so ``pyproject.toml`` is the single source of truth. The
  previous hardcoded value had drifted from the packaged version.
- Wang-Landau 1/t schedule: decoupled the halving criterion from
  ``flatness_limit``. Under ``schedule='1_over_t'`` the halving gate
  now uses the Belardinelli-Pereyra ``min(H) > 0`` criterion, so the
  BP switch to the 1/t phase reliably fires under any
  ``flatness_limit``. Vanilla ``schedule='halving'`` is unchanged.

### Changed

- Renamed ``is_flat`` to ``halving_criterion_met`` on
  ``WangLandauReplica``, ``WangLandauWindowGroup``, and the
  ``WalkerPostBlockState`` snapshot field, reflecting that the
  criterion is no longer flatness-based under all schedules.

## [0.15.0] - 2026-05-26

### Added

- ``mchammer_pt.analysis.coexistence`` — first-order coexistence-point
  estimator that takes a stitched Wang–Landau DOS and returns the
  equal-area temperature, phase peak locations, dividing energy,
  latent heat, and free-energy barrier height as a
  ``CoexistencePoint`` bundle. Public entry points:
  ``find_phase_split`` (diagnostic shape analysis at a fixed T) and
  ``equal_area_temperature`` (single 1D bisection on T with the
  dividing energy refined at every step). Both re-exported from
  ``mchammer_pt.analysis``.
- ``mchammer_pt.analysis._partition`` — internal count-weighted
  partition helpers (linear fractional apportionment of the boundary
  bin, log-space stability) shared between the coexistence module
  and its tests.
- ``mchammer-pt-coexistence`` console script alongside
  ``mchammer-pt-stitch`` and ``mchammer-pt-reweight``. Reads a
  stitched-DOS CSV and writes a one-row coexistence-point summary
  in JSON or CSV. Validates the input as a uniform, finite-valued,
  numeric grid at the boundary.
- ``mchammer-pt-stitch`` gains three filter flags. ``--windows IDX[,IDX...]``
  keeps only the listed 0-based window indices (energy-sorted ascending).
  ``--emin E_MIN`` and ``--emax E_MAX`` drop bins outside the range from
  each surviving window's entropy DataFrame before stitching, so the
  overlap-alignment step sees only the kept bins. Motivated by the
  equal-area coexistence analysis, where the low-energy windows of a REWL
  run are often the hardest to converge and contribute the noisiest part
  of the stitched DOS.

## [0.14.0] - 2026-05-25

### Changed

- ``WangLandauProgressPrinter`` now reports the current WL phase
  (``halv`` while the flatness gate still drives halving, ``1/t``
  once the Belardinelli-Pereyra switch has fired) as a new column in
  the per-window table. A ``1_over_t`` window stops halving by
  design and the ``fill_factor`` then decays continuously, which
  previously made the table hard to distinguish from a stalled
  halving run; the phase column resolves the ambiguity.
- ``WangLandauReplica.window_stats()``,
  ``WangLandauWindowGroup.window_stats()``, and
  ``WangLandauPool.per_window_stats()`` now include a ``phase`` key
  (``"halving"`` or ``"1_over_t"``) carrying the same value the
  printer consults.

## [0.13.0] - 2026-05-22

### Added

- ``mchammer-pt-stitch`` console script. By default reads one
  mchammer-pt checkpoint HDF5 (the artefact written by
  ``data_container_file=`` / ``save_checkpoint`` /
  ``CheckpointWriter``); with ``--containers`` reads two or more
  ``WangLandauDataContainer`` files directly. Containers are
  grouped by window using each one's ``energy_limit_left`` /
  ``energy_limit_right`` ensemble parameters, walker-merged within
  each window via ``WindowResult.get_entropy()``, and stitched via
  ``stitch_entropy``. Pairs with the existing
  ``mchammer-pt-reweight`` script to give a two-step
  DOS-to-canonical CLI pipeline that handles single- and multi-walker
  REWL output through one code path.

### Changed

- ``mchammer_pt.read_hdf5`` now dispatches per-replica reads on
  ``meta["ensemble_cls_fqn"]``: WL checkpoints route through
  ``WangLandauDataContainer.read`` (which restores int bin keys in
  ``_last_state`` and re-tuples ``_random_state``), while canonical
  and other ensembles continue to use ``BaseDataContainer.read``.
  Previously every consumer of a WL checkpoint had to re-apply
  ``_coerce_wl_last_state_keys_to_int`` itself; now the conversion
  happens at the source.
- ``examples/09_dos_postprocessing.py``: end-to-end Python-API
  pipeline (REWL run -> per-window walker-merge -> stitch -> canonical
  reweight) showing the workflow the two CLIs wrap.

## [0.12.0] - 2026-05-21

### Added

- ``mchammer_pt.analysis.dos`` module with ``stitch_entropy`` and
  ``reweight_canonical_from_dos`` for generic Wang-Landau DOS
  post-processing. Both helpers are re-exported from the package
  root.
- ``mchammer-pt-reweight`` console script for canonical reweighting
  from a stitched DOS CSV, with optional ``--plot`` output (requires
  ``matplotlib``; install with the ``plot`` extra).

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
