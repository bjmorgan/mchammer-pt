"""Orchestration for the REWL window-seeding search.

Runs a spawn ``ProcessPoolExecutor`` of short, independent confined
walks across rounds, dedups per window, and raises if a window cannot
be filled or if a worker process dies. Start configurations are built
in the main process (so the caller's ``random_fill`` factory never
crosses the spawn boundary) and shipped to workers as ``AtomsSpec``.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import numbers
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_moves import Move

from ..parallel._builder import AtomsSpec
from ..parallel._imports import _check_importable
from .anchoring import Anchor, assign_anchors, validate_anchor_override
from .bookkeeping import WindowHarvest
from .params import SeedSearchParams
from .walk import _in_band, confined_walk

# Heartbeat cadence (seconds) for the still-short-window progress line.
_HEARTBEAT_INTERVAL_S = 30.0

# Default search knobs, as a module-level singleton so the public
# signature can reference it without a call in argument defaults.
_DEFAULT_PARAMS = SeedSearchParams()


@dataclass(frozen=True)
class _WalkTask:
    """One confined-walk job dispatched to a worker."""

    window_idx: int
    lo: float | None
    hi: float | None
    start: AtomsSpec
    walk_seed: int


# Per-worker globals, populated by ``_init_worker`` in each spawn child.
_G_CALC: ClusterExpansionCalculator | None = None
_G_MOVES: list[tuple[Move, float]] | None = None
_G_ENERGY_SPACING: float = 0.0
_G_PENALTY: float = 0.0
_G_N_STEPS: int = 0


def _init_worker(
    ce_path: str,
    template: AtomsSpec,
    moves: list[tuple[Move, float]],
    energy_spacing: float,
    window_search_penalty: float,
    n_steps: int,
) -> None:
    """Build the per-worker calculator + constants once per spawn child."""
    global _G_CALC, _G_MOVES, _G_ENERGY_SPACING, _G_PENALTY, _G_N_STEPS
    ce = ClusterExpansion.read(ce_path)
    _G_CALC = ClusterExpansionCalculator(template.to_atoms(), ce)
    _G_MOVES = moves
    _G_ENERGY_SPACING = float(energy_spacing)
    _G_PENALTY = float(window_search_penalty)
    _G_N_STEPS = int(n_steps)


def _run_walk(task: _WalkTask) -> tuple[int, np.ndarray | None]:
    """Run one confined walk in a worker; return ``(window_idx, occ|None)``."""
    assert _G_CALC is not None and _G_MOVES is not None
    result = confined_walk(
        task.start.to_atoms(),
        _G_CALC,
        _G_MOVES,
        lo=task.lo,
        hi=task.hi,
        energy_spacing=_G_ENERGY_SPACING,
        window_search_penalty=_G_PENALTY,
        n_steps=_G_N_STEPS,
        seed=task.walk_seed,
    )
    if result is None:
        return task.window_idx, None
    return task.window_idx, result.numbers.copy()


def _walk_result(
    future: cf.Future[tuple[int, np.ndarray | None]],
) -> tuple[int, np.ndarray | None]:
    """Unwrap a confined-walk future, mapping worker death to RuntimeError.

    A worker killed by the OS (e.g. an out-of-memory kill on a large
    supercell) breaks the pool; ``ProcessPoolExecutor`` then raises
    ``BrokenProcessPool`` rather than hanging, and a worker that raises
    re-surfaces its own exception. Both are wrapped in a ``RuntimeError``
    so the documented contract holds and the cause is named.
    """
    try:
        return future.result()
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "a seed-search worker process died unexpectedly (it may have "
            "been killed by the OS, e.g. an out-of-memory kill on a large "
            "supercell). Try reducing n_workers or the supercell size."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"a seed-search worker raised while running a confined walk: "
            f"{exc!r}"
        ) from exc


def _seed(random_seed: int, *tags: int) -> int:
    """Deterministic, dispatch-order-independent child seed."""
    entropy = [int(random_seed), *[int(t) for t in tags]]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _validate_inputs(
    windows: Sequence[tuple[float | None, float | None]],
    counts: Sequence[int],
    moves: Sequence[tuple[Move, float]],
    max_walks_per_window: int,
    energy_spacing: float,
) -> None:
    if not energy_spacing > 0:
        raise ValueError(f"energy_spacing must be > 0; got {energy_spacing}.")
    if len(windows) < 1:
        raise ValueError("windows must be non-empty.")
    for i, (lo, hi) in enumerate(windows):
        if lo is not None and hi is not None and not (lo < hi):
            raise ValueError(
                f"window {i}: left edge {lo} must be strictly less than "
                f"right edge {hi}."
            )
    if len(counts) != len(windows):
        raise ValueError(
            f"counts has {len(counts)} entries but windows has "
            f"{len(windows)}; supply one count per window."
        )
    for i, c in enumerate(counts):
        if not isinstance(c, numbers.Integral):
            raise ValueError(
                f"counts[{i}] must be an integer; got {type(c).__name__}."
            )
        if c < 1:
            raise ValueError(f"counts[{i}] must be >= 1; got {c}.")
    max_count = max(int(c) for c in counts)
    if max_walks_per_window < max_count:
        raise ValueError(
            f"max_walks_per_window ({max_walks_per_window}) is less than "
            f"the largest per-window count ({max_count}); a window gains at "
            f"most one configuration per round, so such a window could never "
            f"be filled. Increase max_walks_per_window to at least "
            f"{max_count}."
        )
    if len(moves) < 1:
        raise ValueError("moves must be non-empty.")
    for move, _weight in moves:
        # ``Move`` instances are callable (they define ``__call__``);
        # ``_check_importable`` resolves an instance to ``type(obj)``.
        # mypy cannot see ``mchammer_moves``' callable typing.
        _check_importable(move, kind="move")  # type: ignore[arg-type]


def _energy(calc: ClusterExpansionCalculator, atoms: Atoms) -> float:
    return float(calc.calculate_total(occupations=atoms.numbers))


def _validate_fill_output(atoms: object, template: Atoms, seed: int) -> Atoms:
    """Validate one ``random_fill`` output at the caller boundary.

    ``random_fill`` is a user-supplied callable, and every walk reads its
    occupation vector against a single ``ClusterExpansionCalculator``
    bound to ``bottom_anchor``'s fixed index->site mapping. A fill with
    the same cell and the same set of positions but a *different atom
    ordering* (e.g. species-grouped instead of the anchor's site order)
    therefore yields silently wrong energies -- walks target the wrong
    band and windows mis-fill with nothing to surface the cause.

    Guarding here, in the main process where the fill is produced,
    converts that silent wrong-energy failure into an immediate,
    self-explaining error. The configuration is *not* canonicalised:
    reordering a caller's structure could mask a genuine caller bug and
    would defeat the build-once-calculator optimisation, so the contract
    is enforced loudly instead.

    Args:
        atoms: the value ``random_fill`` returned.
        template: the ``bottom_anchor`` whose lattice and atom ordering
            every fill must match.
        seed: the seed ``random_fill`` was called with (named in errors).

    Returns:
        ``atoms`` unchanged, once validated.

    Raises:
        ValueError: if ``atoms`` is not an ``Atoms``, has a different
            site count, cell, periodic boundary conditions, or atom
            ordering (per-index positions) from ``template``.
    """
    contract = (
        f"the structure from random_fill(seed={seed}) must be on the same "
        f"lattice and atom ordering as bottom_anchor, differing only in the "
        f"species at each site"
    )
    if not isinstance(atoms, Atoms):
        raise ValueError(
            f"random_fill(seed={seed}) must return an ase.Atoms; got "
            f"{type(atoms).__name__}."
        )
    if len(atoms) != len(template):
        raise ValueError(
            f"{contract}; it has {len(atoms)} sites but bottom_anchor has "
            f"{len(template)}."
        )
    if not np.allclose(atoms.cell.array, template.cell.array, atol=1e-6):
        raise ValueError(f"{contract}; its cell differs from bottom_anchor's.")
    if not np.array_equal(atoms.pbc, template.pbc):
        raise ValueError(
            f"{contract}; its periodic boundary conditions differ from "
            f"bottom_anchor's."
        )
    if not np.allclose(atoms.positions, template.positions, atol=1e-6):
        raise ValueError(
            f"{contract}; its atom positions differ from bottom_anchor's. A "
            f"reordered structure (same positions, different index order) "
            f"fails this index-by-index check -- the occupation vector would "
            f"otherwise be read against the wrong sites."
        )
    return atoms


def seed_window_configs(
    cluster_expansion: ClusterExpansion,
    moves: Sequence[tuple[Move, float]],
    windows: Sequence[tuple[float | None, float | None]],
    counts: Sequence[int],
    energy_spacing: float,
    bottom_anchor: Atoms,
    random_fill: Callable[[int], Atoms],
    *,
    random_seed: int,
    params: SeedSearchParams = _DEFAULT_PARAMS,
    anchors: Sequence[str] | None = None,
) -> list[list[Atoms]]:
    """Fill each window with ``counts[i]`` distinct, in-band start configs.

    Each window is anchored to whichever energy end is nearer (the
    ground state below, a fresh random fill above) and a directed
    Wang-Landau window search drives configurations into the band.
    ``counts[i]`` independent confined walks per window, cross-deduped,
    produce ``counts[i]`` distinct seeds. When its energy lies within a
    window's band the exact ground state (``bottom_anchor``) is injected
    into that window as one free seed; if it lies below every window the
    injection is skipped and those windows are filled by the search
    alone.

    The search is material-agnostic: the ground state, the random fill,
    and the move set are all caller-supplied. ``random_fill`` is called
    in the main process and its outputs shipped to workers, so it may be
    any callable (closures and lambdas are fine).

    Args:
        cluster_expansion: icet ``ClusterExpansion`` defining the energy.
        moves: move set as ``(Move, weight)`` pairs. Each move class
            must be importable by a spawn worker (no Jupyter-cell or
            function-local move classes).
        windows: per-window ``(lo, hi)`` edges; ``None`` is unbounded.
            A single window is accepted here, but
            ``WangLandauParallelTempering.process_pool`` (the intended
            consumer of the result) requires at least two.
        counts: target number of distinct seeds per window (K).
        energy_spacing: WL energy-grid bin size.
        bottom_anchor: ground-state structure. Start for bottom-anchored
            walks, calculator template, and a guaranteed seed.
        random_fill: ``seed -> Atoms`` producing fresh,
            correct-composition, high-energy structures. Each returned
            structure must be on the same lattice and the same atom
            ordering as ``bottom_anchor`` -- identical cell and
            per-index positions, differing only in the species at each
            site -- because a single calculator bound to ``bottom_anchor``
            reads every occupation vector against that fixed site order.
            This is validated up front.
        random_seed: master seed; per-walk seeds derive from it.
        params: search knobs (see :class:`SeedSearchParams`).
        anchors: optional explicit ``"bottom"``/``"top"`` per window;
            ``None`` auto-derives from energies.

    Returns:
        One list of ``Atoms`` per window, in window order, each of
        length ``counts[i]`` and all distinct and in-band.

    Raises:
        ValueError: on input validation failure, or if ``random_fill``
            returns a structure incompatible with ``bottom_anchor``.
        RuntimeError: if any window cannot be filled to its target after
            ``params.max_walks_per_window`` rounds, or if a seed-search
            worker process raises or dies unexpectedly (e.g. an
            out-of-memory kill).
    """
    _validate_inputs(
        windows, counts, moves, params.max_walks_per_window, energy_spacing
    )
    windows = [tuple(w) for w in windows]  # type: ignore[misc]
    counts = [int(c) for c in counts]
    n_windows = len(windows)
    n_workers = params.n_workers or os.cpu_count() or 1

    template_spec = AtomsSpec.from_atoms(bottom_anchor)
    n_steps = int(params.walk_sweeps) * len(bottom_anchor)

    harvest = WindowHarvest(counts)

    ctx = mp.get_context("spawn")
    tmpdir = tempfile.TemporaryDirectory()
    try:
        ce_path = Path(tmpdir.name) / "cluster_expansion.ce"
        cluster_expansion.write(str(ce_path))

        # Main-process calculator for e_gs / e_top / GS injection.
        main_calc = ClusterExpansionCalculator(
            bottom_anchor.copy(),  # type: ignore[no-untyped-call]
            cluster_expansion,
        )
        e_gs = _energy(main_calc, bottom_anchor)

        # Resolve anchors.
        if anchors is not None:
            anchor_kinds: list[Anchor] = validate_anchor_override(
                anchors, n_windows
            )
        else:
            probe_seed = _seed(random_seed, n_windows, 0, 0)
            top_probe = _validate_fill_output(
                random_fill(probe_seed), bottom_anchor, probe_seed
            )
            e_top = _energy(main_calc, top_probe)
            anchor_kinds = assign_anchors(windows, e_gs, e_top)

        # GS injection: the first window whose band contains e_gs gets the
        # exact ground state as a free seed. If e_gs lies below every
        # window (a ladder that does not reach the true ground state) the
        # injection is skipped -- reported, not silent.
        injected = False
        for i, (lo, hi) in enumerate(windows):
            if _in_band(e_gs, lo, hi):
                harvest.record(i, bottom_anchor.numbers)
                injected = True
                break
        if not injected:
            print(
                f"[seed] ground-state energy {e_gs:.6g} lies outside every "
                f"window band; not injected as a free seed (those windows "
                f"are filled by the search).",
                flush=True,
            )

        print(
            f"[seed] {n_windows} windows, counts={counts}, "
            f"{n_workers} workers (spawn)",
            flush=True,
        )

        with cf.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                str(ce_path),
                template_spec,
                list(moves),
                float(energy_spacing),
                float(params.window_search_penalty),
                n_steps,
            ),
        ) as executor:
            t0 = time.monotonic()
            next_heartbeat = t0 + _HEARTBEAT_INTERVAL_S
            for round_idx in range(params.max_walks_per_window):
                if harvest.all_full():
                    break
                tasks: list[_WalkTask] = []
                for i, (lo, hi) in enumerate(windows):
                    if harvest.is_full(i):
                        continue
                    if anchor_kinds[i] == "bottom":
                        start = bottom_anchor.copy()  # type: ignore[no-untyped-call]
                    else:
                        fill_seed = _seed(random_seed, i, round_idx, 0)
                        start = _validate_fill_output(
                            random_fill(fill_seed), bottom_anchor, fill_seed
                        )
                    tasks.append(
                        _WalkTask(
                            window_idx=i,
                            lo=lo,
                            hi=hi,
                            start=AtomsSpec.from_atoms(start),
                            walk_seed=_seed(random_seed, i, round_idx, 1),
                        )
                    )
                if not tasks:
                    break
                futures = [executor.submit(_run_walk, task) for task in tasks]
                for future in cf.as_completed(futures):
                    window_idx, occ = _walk_result(future)
                    if occ is not None and harvest.record(window_idx, occ):
                        n_filled, _ = harvest.fill_status()
                        print(
                            f"  window {window_idx}: "
                            f"{len(harvest.configs(window_idx))}/"
                            f"{counts[window_idx]} "
                            f"({n_filled}/{n_windows} windows filled)",
                            flush=True,
                        )
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        n_filled, short = harvest.fill_status()
                        print(
                            f"  [seed] {now - t0:.0f}s, "
                            f"{n_filled}/{n_windows} filled; short: {short}",
                            flush=True,
                        )
                        next_heartbeat = now + _HEARTBEAT_INTERVAL_S

        if not harvest.all_full():
            _, short = harvest.fill_status()
            raise RuntimeError(
                f"seed search could not fill all windows after "
                f"{params.max_walks_per_window} rounds; short windows "
                f"(index: found/target): {short}"
            )

        out: list[list[Atoms]] = []
        for i in range(n_windows):
            per_window: list[Atoms] = []
            for occ in harvest.configs(i):
                atoms = bottom_anchor.copy()  # type: ignore[no-untyped-call]
                atoms.numbers = occ
                per_window.append(atoms)
            out.append(per_window)
        print(
            f"[seed] complete: all {n_windows} windows filled in "
            f"{time.monotonic() - t0:.0f}s",
            flush=True,
        )
        return out
    finally:
        tmpdir.cleanup()
