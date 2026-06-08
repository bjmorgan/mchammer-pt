"""Per-window Wang-Landau replica handle.

Sibling of `mchammer_pt.replica.Replica`. Wraps a single
`icet.mchammer.ensembles.WangLandauEnsemble` for use inside the REWL
orchestrator. To use the Belardinelli-Pereyra 1/t schedule, pass
``ensemble_kwargs={'schedule': '1_over_t'}``; the default
``schedule='halving'`` gives the standard WL fill-factor scheme.
"""

from __future__ import annotations

import copy
import os
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import (
    ClusterExpansionCalculator,
)
from mchammer.data_containers.base_data_container import (
    BaseDataContainer,
)
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)
from mchammer.ensembles import WangLandauEnsemble
from mchammer.observers.base_observer import (
    BaseObserver,
)

from .wl_coordinator import (
    CoordinatorPlan,
    Phase,
    Schedule,
    WalkerPostBlockState,
    _min_over_mean,
)
from .wl_ensemble import CoordinatedWangLandauEnsemble

_RESERVED_ENSEMBLE_KWARGS: frozenset[str] = frozenset(
    {
        "structure",
        "calculator",
        "energy_spacing",
        "energy_limit_left",
        "energy_limit_right",
        "random_seed",
        "recency_visits_per_bin",
        "dos_snapshot_ratio",
        "dc_filename",
    }
)

# `_last_state` fields whose dict keys are integer bin indices.
# JSON round-trips coerce these to strings; the conversion has to be
# reversed before mchammer's `_restart_ensemble` reads them. Matches
# the set `WangLandauDataContainer.read` converts upstream.
_WL_INT_KEY_FIELDS: frozenset[str] = frozenset(
    {
        "histogram",
        "entropy",
        "fill_factor_history",
        "entropy_history",
        "fill_factor_snapshots",
        "entropy_snapshots",
    }
)


def _coerce_wl_last_state_keys_to_int(last_state: dict[str, Any]) -> None:
    """Convert string dict keys back to ints in WL `_last_state` fields.

    Mirrors the conversion `WangLandauDataContainer.read` applies
    inline; used on the WL resume path when the container was
    deserialised via `BaseDataContainer.read` (which does not know
    about WL).
    """
    for tag in _WL_INT_KEY_FIELDS:
        if tag not in last_state:
            continue
        value = last_state[tag]
        if not value:
            continue
        first_key = next(iter(value))
        if isinstance(first_key, int):
            continue  # already int-keyed
        try:
            converted: dict[int, Any] = {}
            for key, val in value.items():
                if isinstance(val, dict):
                    val = {int(k): v for k, v in val.items()}
                converted[int(key)] = val
        except ValueError as exc:
            raise ValueError(
                f"WL `_last_state` field {tag!r} contains a non-integer "
                f"bin key; the checkpoint is malformed. Original error: "
                f"{exc}"
            ) from exc
        last_state[tag] = converted


def log_g_at(
    entropy: dict[int, float],
    energy: float,
    energy_spacing: float,
    *,
    bin_left: int | None,
    bin_right: int | None,
) -> float:
    """Density-of-states ``ln g`` at ``energy`` from a cached entropy dict.

    Mirrors :meth:`WangLandauReplica.log_g` without needing a live
    ensemble: the bin index is ``round(energy / energy_spacing)`` (icet's
    convention), the window is the inclusive bin range
    ``[bin_left, bin_right]`` (``None`` meaning unbounded on that side),
    and an unvisited in-window bin has ``ln g = 0``. Out-of-window
    energies return ``-inf``. This lets the parent process evaluate REWL
    exchange acceptance from per-walker entropy snapshots without an IPC
    round trip to the worker.

    Args:
        entropy: ``{bin_index: entropy_value}`` snapshot for the walker.
        energy: energy at which to evaluate ``ln g``.
        energy_spacing: bin width in eV.
        bin_left: inclusive lower window bin index, or ``None`` if the
            window is unbounded below.
        bin_right: inclusive upper window bin index, or ``None`` if the
            window is unbounded above.

    Returns:
        ``ln g(energy)``, or ``-inf`` if ``energy`` is outside the window
        or not a finite number.
    """
    if not np.isfinite(energy):
        return -float(np.inf)
    bin_idx = int(round(energy / energy_spacing))
    if bin_left is not None and bin_idx < bin_left:
        return -float(np.inf)
    if bin_right is not None and bin_idx > bin_right:
        return -float(np.inf)
    return float(entropy.get(bin_idx, 0.0))


@runtime_checkable
class WangLandauSlot(Protocol):
    """Structural interface shared by single-walker and multi-walker WL slots.

    Both ``WangLandauReplica`` (single walker) and
    ``WangLandauWindowGroup`` (multi-walker) satisfy this protocol.
    Used in type annotations by the serial and process pools.
    """

    @property
    def energy_window(self) -> tuple[float | None, float | None]: ...
    @property
    def energy_spacing(self) -> float: ...
    @property
    def ensemble(self) -> Any: ...
    @property
    def cluster_expansion_path(self) -> str | None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def phase(self) -> Phase: ...
    @property
    def schedule(self) -> Schedule: ...
    @property
    def flatness_limit(self) -> float: ...
    @property
    def walker_states(self) -> Sequence[WalkerPostBlockState]: ...
    def apply_plan(self, plan: CoordinatorPlan) -> None: ...
    def halving_criterion_met(self) -> bool: ...
    def advance(self, n_steps: int) -> None: ...
    def log_g(self, energy: float) -> float: ...
    @property
    def n_walkers(self) -> int: ...
    def walker_energy(self, walker: int) -> float: ...
    def walker_occupations(self, walker: int) -> np.ndarray: ...
    def set_walker_occupations(self, walker: int, occupations: np.ndarray) -> None: ...
    def walker_log_g(self, walker: int, energy: float) -> float: ...
    def data_container(self) -> WangLandauDataContainer: ...
    def all_data_containers(self) -> list[WangLandauDataContainer]: ...
    def refresh_last_state(self) -> None: ...
    def window_stats(self) -> dict[str, Any]: ...
    def snapshot_for_checkpoint(self) -> dict[str, Any]: ...
    def finalise_for_reporting(self) -> None: ...
    def attach_mchammer_observer(self, observer: BaseObserver) -> None: ...
    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
    ) -> None: ...


class WangLandauReplica:
    """One Wang-Landau ensemble at one energy window, wrapped for REWL use.

    Invariant: a `WangLandauReplica` always has a configuration whose
    energy lies inside its assigned window. The constructor validates
    this at startup, and both `set_occupations` and `restore_state`
    validate the proposed energy before mutating any state so a
    window-violating call leaves the replica untouched. The REWL
    acceptance formula in `WangLandauParallelTempering._log_prob_ratio`
    relies on this invariant to short-circuit cleanly when only the
    "cross-bin" terms can be -inf.

    Invariant: every bin the walker has been placed at — by
    construction (``bin_init``), ``set_occupations`` (``new_bin``),
    or ``restore_state`` (``new_bin``) — appears as a key in the
    underlying ensemble's ``_histogram`` with count 0. Each of those
    sites uses ``setdefault`` so existing counts from prior visits
    are not overwritten. The Wang-Landau flatness gate iterates over
    the ``_histogram`` values, so a zero-count seeded bin blocks the
    gate until the walker visits it.

    Invariant: ``CoordinatedWangLandauEnsemble._visited_bins`` is the
    set of bins the walker has reached via ``_update_entropy`` since
    window entry. ``_update_entropy`` is the only site that inserts,
    guarded on ``_reached_energy_window``. ``refresh_last_state``
    writes the set to the data container and ``restore_state`` reads
    it back, so membership survives checkpoint round-trips.
    ``window_stats`` reports ``bins_visited`` from this set.

    Args:
        cluster_expansion: icet ClusterExpansion defining the energy.
        atoms: starting structure. Its energy must lie inside the
            window (validated at construction).
        energy_spacing: bin size of the WL energy grid.
        energy_limit_left: lower window edge, or None for unbounded.
        energy_limit_right: upper window edge, or None for unbounded.
        random_seed: seed for this replica's MC random generator.
        ensemble_cls: WL ensemble class. Defaults to
            ``CoordinatedWangLandauEnsemble``, which delegates
            halving to the enclosing ``WangLandauWindowGroup``
            coordinator. Must be a subclass of
            ``CoordinatedWangLandauEnsemble``. To use the 1/t
            schedule, pass ``ensemble_kwargs={'schedule':
            '1_over_t'}``.
        ensemble_kwargs: extra kwargs forwarded to ensemble construction.
            Reserved names (see `_RESERVED_ENSEMBLE_KWARGS`) cannot
            appear here — they are set by the wrapper.
        recency_visits_per_bin: EWMA recency window forwarded to the
            ensemble's recency-flatness diagnostic.
        dos_snapshot_ratio: ratio of the 1/t-regime DOS snapshot ladder
            forwarded to the ensemble; ``None`` disables snapshotting.
        cluster_expansion_path: same semantics as
            `mchammer_pt.replica.Replica`.

    Raises:
        ValueError: if `ensemble_kwargs` contains a reserved name, or
            if the initial configuration's energy is outside the
            window.
    """

    def __init__(
        self,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        energy_spacing: float,
        energy_limit_left: float | None,
        energy_limit_right: float | None,
        random_seed: int,
        *,
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        cluster_expansion_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._energy_spacing = float(energy_spacing)
        self._energy_limit_left = (
            None if energy_limit_left is None else float(energy_limit_left)
        )
        self._energy_limit_right = (
            None if energy_limit_right is None else float(energy_limit_right)
        )
        self._cluster_expansion_path = (
            None
            if cluster_expansion_path is None
            else os.fspath(cluster_expansion_path)
        )
        if not issubclass(ensemble_cls, CoordinatedWangLandauEnsemble):
            raise TypeError(
                f"ensemble_cls must be a subclass of "
                f"CoordinatedWangLandauEnsemble; got "
                f"{ensemble_cls.__name__}. Halving is now coordinated "
                f"by WangLandauWindowGroup, so the plain "
                f"WangLandauEnsemble would autonomously halve and "
                f"conflict with the coordinator."
            )
        extra = dict(ensemble_kwargs) if ensemble_kwargs else {}
        clash = _RESERVED_ENSEMBLE_KWARGS & extra.keys()
        if clash:
            raise ValueError(
                f"ensemble_kwargs must not contain {sorted(clash)}; "
                f"these are set by WangLandauReplica from its own "
                f"arguments (structure/calculator from "
                f"cluster_expansion+atoms; energy_spacing, "
                f"energy_limit_left, energy_limit_right, "
                f"random_seed, recency_visits_per_bin from their "
                f"dedicated parameters; "
                f"dc_filename is always pinned to None to disable "
                f"periodic on-disk writes)."
            )
        atoms_copy: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
        calculator = ClusterExpansionCalculator(atoms_copy, cluster_expansion)
        caller_state = random.getstate()
        try:
            self._ensemble: WangLandauEnsemble = ensemble_cls(
                structure=atoms_copy,
                calculator=calculator,
                energy_spacing=self._energy_spacing,
                energy_limit_left=self._energy_limit_left,
                energy_limit_right=self._energy_limit_right,
                random_seed=int(random_seed),
                dc_filename=None,
                recency_visits_per_bin=recency_visits_per_bin,
                dos_snapshot_ratio=dos_snapshot_ratio,
                **extra,
            )
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)

        # Validate the initial configuration lies inside the window.
        e = self._ensemble
        bin_init = e._get_bin_index(e._potential)
        if bin_init is None or not e._inside_energy_window(bin_init):
            raise ValueError(
                f"initial energy {e._potential} (bin {bin_init}) is "
                f"outside window "
                f"[{self._energy_limit_left}, {self._energy_limit_right}]; "
                f"each WL replica must start with a configuration whose "
                f"energy lies in its window."
            )

        # Maintain the known-bin invariant (see class docstring).
        e._histogram.setdefault(bin_init, 0)

        self.walker_states: tuple[WalkerPostBlockState, ...] = (
            WalkerPostBlockState(
                halving_criterion_met=False,
                fill_factor=1.0,
                entropy={},
                step=0,
                window_entry_step=None,
                histogram={},
                reached_energy_window=False,
                current_energy=0.0,
            ),
        )

    @property
    def energy_window(self) -> tuple[float | None, float | None]:
        return (self._energy_limit_left, self._energy_limit_right)

    @property
    def energy_spacing(self) -> float:
        return self._energy_spacing

    @property
    def ensemble(self) -> WangLandauEnsemble:
        return self._ensemble

    @property
    def phase(self) -> Phase:
        return cast(Phase, self._ensemble._phase)

    @property
    def schedule(self) -> Schedule:
        return cast(Schedule, self._ensemble._schedule)

    @property
    def flatness_limit(self) -> float:
        return float(self._ensemble._flatness_limit)

    def current_energy(self) -> float:
        """Cached running total of the WL ensemble (eV)."""
        return float(self._ensemble._potential)

    def current_occupations(self) -> np.ndarray:
        return self._ensemble.configuration.occupations.copy()

    def log_g(self, energy: float) -> float:
        """Return ln g at the given energy, or -inf if outside the window.

        Returns 0.0 (i.e. g = 1) for unvisited in-window bins, treating
        them as singly-degenerate for REWL exchange acceptance.
        """
        e = self._ensemble
        return log_g_at(
            e._entropy,
            energy,
            self._energy_spacing,
            bin_left=e._bin_left,
            bin_right=e._bin_right,
        )

    @property
    def n_walkers(self) -> int:
        """Number of walkers in this slot (always 1 for a bare replica)."""
        return 1

    def walker_energy(self, walker: int) -> float:
        """Current energy of the single walker (``walker`` is always 0)."""
        return self.current_energy()

    def walker_occupations(self, walker: int) -> np.ndarray:
        """Current occupations of the single walker (``walker`` is always 0)."""
        return self.current_occupations()

    def set_walker_occupations(self, walker: int, occupations: np.ndarray) -> None:
        """Set the single walker's configuration (``walker`` is always 0)."""
        self.set_occupations(occupations)

    def walker_log_g(self, walker: int, energy: float) -> float:
        """``ln g(energy)`` for the single walker (``walker`` is always 0)."""
        return self.log_g(energy)

    def set_occupations(self, occupations: np.ndarray) -> None:
        """Overwrite the replica's configuration and refresh WL-specific caches.

        `WangLandauEnsemble` caches `_potential` as a running total
        mutated in-place by the acceptance loop. `update_occupations`
        alone does NOT refresh it, so this method also recomputes the
        cached potential and the `_reached_energy_window` flag from
        the new configuration. Without this refresh, the next trial
        step would look up the wrong bin in `_entropy`, silently
        corrupting the entropy estimate.

        Validates the proposed configuration's energy before mutating
        any state, so a window violation leaves the replica untouched.
        """
        occ = np.asarray(occupations, dtype=int)
        e = self._ensemble
        proposed_potential = float(e.calculator.calculate_total(occupations=occ))
        new_bin = e._get_bin_index(proposed_potential)
        if new_bin is None or not e._inside_energy_window(new_bin):
            raise ValueError(
                f"set_occupations would leave replica at energy "
                f"{proposed_potential} (bin {new_bin}), outside window "
                f"[{self._energy_limit_left}, {self._energy_limit_right}]."
            )
        # Maintain the known-bin invariant (see class docstring).
        e._histogram.setdefault(new_bin, 0)
        e.update_occupations(sites=list(range(len(occ))), species=list(occ))
        e._potential = proposed_potential
        e._reached_energy_window = True

    def advance(self, n_steps: int) -> None:
        """Run `n_steps` WL trial steps, isolating this replica's RNG stream.

        Mirrors the save/restore discipline used by
        `mchammer_pt.Replica.advance`. icet's `BaseEnsemble.run` may
        short-circuit early once the underlying WL ensemble has
        converged (`_terminate_sampling`); the orchestrator handles
        this case by stopping the global loop when all replicas
        report `converged`. See `converged`.

        After the run, refreshes ``self.walker_states`` from live ensemble
        state so the coordinator can build a SlotView from it.
        """
        previous_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble.run(int(n_steps))
            self._rng_state = random.getstate()
        finally:
            random.setstate(previous_state)
        self.walker_states = (self._snapshot(),)

    def halving_criterion_met(self) -> bool:
        """Return ``True`` if this walker satisfies the halving
        criterion for its current schedule.

        Under ``schedule='halving'`` the criterion is the WL flatness
        test: every bin's count is ``>= flatness_limit * mean(counts)``.
        Under ``schedule='1_over_t'`` the criterion is the BP
        coupon-collector test: every visited bin has been visited at
        least once since the last halve. Walkers that have not yet
        entered the window return ``False``.
        """
        e = self._ensemble
        if not e._reached_energy_window:
            return False
        if not e._histogram:
            return False
        histogram = np.array(list(e._histogram.values()))
        if histogram.size == 0:
            return False
        mean_count = float(np.average(histogram))
        if mean_count <= 0:
            return False
        if e._schedule == "1_over_t":
            # Belardinelli-Pereyra coupon-collector criterion.
            # flatness_limit is not consulted under this schedule.
            return bool(np.all(histogram > 0))
        limit = e._flatness_limit * mean_count
        return bool(np.all(histogram >= limit))

    def _snapshot(self) -> WalkerPostBlockState:
        """Read live ensemble state into a WalkerPostBlockState."""
        e = self._ensemble
        return WalkerPostBlockState(
            halving_criterion_met=self.halving_criterion_met(),
            fill_factor=float(e._fill_factor),
            entropy=dict(e._entropy),
            step=int(e.step),
            window_entry_step=(
                None if e._window_entry_step is None
                else int(e._window_entry_step)
            ),
            histogram=dict(e._histogram),
            reached_energy_window=bool(e._reached_energy_window),
            current_energy=self.current_energy(),
        )

    def apply_plan(self, plan: CoordinatorPlan) -> None:
        """Apply the coordinator's plan to this single walker.

        Order matches ``WangLandauWindowGroup.apply_plan``:
        halve -> write merged entropy -> set phase.
        """
        if plan.halve:
            self.force_halve()
        if plan.merged_entropy is not None:
            self._ensemble._entropy = dict(plan.merged_entropy)
        if plan.switch_to_phase is not None:
            phase = plan.switch_to_phase
            self._ensemble._phase = phase
            if phase == "1_over_t":
                entry = self._ensemble._window_entry_step
                if entry is not None:
                    t = self._ensemble.step - entry + 1
                    self._ensemble._fill_factor = 1.0 / t

    def force_halve(self) -> None:
        """Halve ``_fill_factor`` and record the event in history.

        Halves ``_fill_factor``, records the new value in both
        ``_fill_factor_history`` and ``_entropy_history`` keyed by
        the current MC step (matching upstream mchammer's halving
        convention), and resets the histogram counts to zero while
        preserving keys. Called by ``WangLandauWindowGroup`` when
        the collective flatness gate fires. Sets ``_converged``
        when ``_fill_factor <= _fill_factor_limit``, since
        ``CoordinatedWangLandauEnsemble`` suppresses the
        ``_converged`` write that upstream's ``_update_entropy``
        would have performed.
        """
        from collections import OrderedDict

        e = self._ensemble
        e._fill_factor /= 2.0
        step_key = int(e.step)
        e._fill_factor_history[step_key] = e._fill_factor
        e._entropy_history[step_key] = OrderedDict(
            sorted(e._entropy.items())
        )
        e._histogram = dict.fromkeys(e._histogram, 0)
        if e._fill_factor <= e._fill_factor_limit:
            e._converged = True

    @property
    def converged(self) -> bool:
        """True once the underlying WL ensemble has flagged convergence."""
        return bool(self._ensemble.converged or False)

    def window_stats(self) -> dict[str, Any]:
        """Per-window convergence metrics.

        Returns ``fill_factor``, ``halvings``, ``histogram``,
        ``bins_visited``, ``bins_filled``, ``bins_known``,
        ``recency_flatness``, ``schedule``,
        ``converged``. For a
        single-walker replica ``flatness_mode`` and
        ``per_walker_flat_min`` are omitted (the progress reporter
        falls through to the pooled computation, which is exact
        for n_walkers == 1).

        ``bins_visited`` is ``len(_visited_bins)`` — the count of
        bins the walker has reached via MC since window entry.
        Monotone within a run; survives halvings. ``bins_filled``
        is the count of histogram bins with a positive count; a
        halving zeroes the histogram counts but retains the keys, so
        ``bins_filled`` resets each halving while ``bins_known`` does
        not. ``bins_known`` is ``len(_histogram)`` and includes
        seeded-but-unvisited bins.

        ``phase`` is the current WL phase taken from the underlying
        ensemble (``"halving"`` or ``"1_over_t"``).
        """
        e = self._ensemble
        histogram = dict(e._histogram)
        return {
            "fill_factor": float(e._fill_factor),
            "halvings": max(0, len(e._fill_factor_history) - 1),
            "histogram": histogram,
            "bins_visited": len(e._visited_bins),
            "bins_filled": sum(1 for c in histogram.values() if c > 0),
            "bins_known": len(histogram),
            "recency_flatness": _min_over_mean(e.recency_effective_weights()),
            "schedule": self.schedule,
            "converged": self.converged,
            "phase": self.phase,
        }

    def data_container(self) -> WangLandauDataContainer:
        """The replica's live `WangLandauDataContainer`."""
        return self._ensemble.data_container

    def all_data_containers(self) -> list[WangLandauDataContainer]:
        """Returns a single-element list containing this replica's data container."""
        return [self.data_container()]

    def attach_mchammer_observer(self, observer: BaseObserver) -> None:
        """Attach an mchammer observer; fires inside `advance(...)`."""
        self._ensemble.attach_observer(observer)

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer.

        Constructs ``cls(*args, **kwargs)`` and attaches it.
        """
        self.attach_mchammer_observer(cls(*args, **kwargs))

    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
    ) -> None:
        """Attach an observer constructed via ``factory(self)``.

        ``factory`` must return a ``BaseObserver``.
        """
        observer = factory(self)
        if not isinstance(observer, BaseObserver):
            raise TypeError(
                f"attach_observer_factory: factory returned "
                f"{type(observer).__name__}, not a BaseObserver"
            )
        self.attach_mchammer_observer(observer)

    @property
    def cluster_expansion_path(self) -> str | None:
        return self._cluster_expansion_path

    def refresh_last_state(self) -> None:
        """Populate ``_last_state`` on the live container from ensemble state.

        Writes the fields that ``WindowResult`` reads (entropy,
        histogram, fill_factor, fill_factor_history, entropy_history,
        fill_factor_snapshots, entropy_snapshots), the 1/t-schedule
        fields (schedule, phase, window_entry_step), and visited_bins.
        Idempotent.
        """
        from collections import OrderedDict

        e = self._ensemble
        e._data_container._update_last_state(
            last_step=e.step,
            occupations=e.configuration.occupations.tolist(),
            accepted_trials=e._accepted_trials,
            random_state=self._rng_state,
            fill_factor=e._fill_factor,
            fill_factor_history=e._fill_factor_history,
            entropy_history=e._entropy_history,
            histogram=OrderedDict(sorted(e._histogram.items())),
            entropy=OrderedDict(sorted(e._entropy.items())),
        )
        e._data_container._last_state["schedule"] = e._schedule
        e._data_container._last_state["phase"] = e._phase
        e._data_container._last_state["window_entry_step"] = e._window_entry_step
        e._data_container._last_state["visited_bins"] = sorted(
            e._visited_bins
        )
        e._data_container._last_state["fill_factor_snapshots"] = dict(
            e._fill_factor_snapshots
        )
        e._data_container._last_state["entropy_snapshots"] = {
            step: OrderedDict(sorted(entropy.items()))
            for step, entropy in e._entropy_snapshots.items()
        }

    def finalise_for_reporting(self) -> None:
        """No-op for single-walker slots; the multi-walker counterpart on
        WangLandauWindowGroup merges per-walker entropies."""

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        """Refresh ``_last_state`` and return checkpoint extras.

        Calls ``refresh_last_state`` to populate the container, then
        returns the ``sites_by_species`` extras the checkpoint code
        embeds alongside it.
        """
        self.refresh_last_state()
        e = self._ensemble
        sites_by_species: list[dict[int, list[int]]] = [
            {
                int(species): [int(s) for s in sites]
                for species, sites in sublattice.items()
            }
            for sublattice in e.configuration._sites_by_species
        ]
        return {"sites_by_species": sites_by_species}

    def restore_state(
        self,
        container: BaseDataContainer,
        *,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> None:
        """Mutate this replica to match a saved checkpoint.

        Validates the proposed energy from the container's
        ``_last_state`` before touching any ensemble state, so a
        window violation leaves the replica untouched.

        ``container`` is a `BaseDataContainer` (not a
        `WangLandauDataContainer`) because `read_hdf5` deserialises
        containers generically. Only ``_last_state`` is needed for
        restoration; the existing `WangLandauDataContainer` on the
        ensemble is preserved so WL-specific post-processing methods
        remain available after resume.
        """
        # Deep-copy `_last_state` so that coercion and validation
        # operate on our own dict rather than aliasing the caller's.
        # Without the copy, `_coerce_wl_last_state_keys_to_int`
        # would mutate the caller's container before validation runs,
        # and post-restore mutations would leak back.
        last_state = copy.deepcopy(container._last_state)

        # `BaseDataContainer.read` deserialises `_last_state` via JSON,
        # which coerces integer dict keys to strings.
        # `WangLandauDataContainer.read` overrides this and converts
        # them back. Containers reaching us through
        # `mchammer_pt.history.read_hdf5` are read as plain
        # `BaseDataContainer`s (history.py does not own WL knowledge),
        # so the conversion has to happen here for `_restart_ensemble`
        # to find its integer-keyed bin lookups.
        _coerce_wl_last_state_keys_to_int(last_state)

        # Validate the proposed configuration's energy before mutating
        # any ensemble state.
        proposed_occ = np.asarray(
            last_state["occupations"], dtype=int
        )
        proposed_potential = float(
            self._ensemble.calculator.calculate_total(occupations=proposed_occ)
        )
        new_bin = self._ensemble._get_bin_index(proposed_potential)
        if new_bin is None or not self._ensemble._inside_energy_window(new_bin):
            raise ValueError(
                f"restore_state would leave replica at energy "
                f"{proposed_potential} (bin {new_bin}), outside window "
                f"[{self._energy_limit_left}, {self._energy_limit_right}]."
            )

        # Copy the saved state into the existing WL-typed container
        # rather than replacing it wholesale. `read_hdf5` returns
        # `BaseDataContainer` instances; assigning one to the ensemble
        # would lose the `WangLandauDataContainer` subclass and break
        # WL-specific post-processing.
        self._ensemble._data_container._last_state = last_state
        caller_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._ensemble._restart_ensemble()
            self._rng_state = random.getstate()
        finally:
            random.setstate(caller_state)
        # After `_restart_ensemble`, configuration occupations match
        # the saved state; refresh the WL-cached potential and window
        # flag (reusing the already-computed potential).
        e = self._ensemble
        e._potential = proposed_potential
        e._reached_energy_window = True
        # Older checkpoints may not carry `visited_bins`; treat as empty.
        saved_visited = last_state.get("visited_bins")
        if saved_visited is not None:
            e._visited_bins = {int(b) for b in saved_visited}
        else:
            e._visited_bins = set()
        # Snapshot store: present on checkpoints written after this
        # feature landed; older checkpoints restore to an empty store.
        # `last_state` has already passed through
        # `_coerce_wl_last_state_keys_to_int`, so keys are ints.
        saved_ff_snaps = last_state.get("fill_factor_snapshots")
        saved_entropy_snaps = last_state.get("entropy_snapshots")
        if saved_ff_snaps is not None and saved_entropy_snaps is not None:
            e._fill_factor_snapshots = {
                int(k): float(v) for k, v in saved_ff_snaps.items()
            }
            e._entropy_snapshots = {
                int(step): {int(b): float(val) for b, val in entropy.items()}
                for step, entropy in saved_entropy_snaps.items()
            }
        else:
            e._fill_factor_snapshots = {}
            e._entropy_snapshots = {}
        e._rebuild_max_snapshot_rung()
        # Maintain the known-bin invariant (see class docstring).
        e._histogram.setdefault(new_bin, 0)
        if sites_by_species is not None:
            self._ensemble.configuration._sites_by_species = sites_by_species

    @classmethod
    def restart_from(
        cls,
        container: BaseDataContainer,
        *,
        cluster_expansion: ClusterExpansion,
        atoms: Atoms,
        energy_spacing: float,
        energy_limit_left: float | None,
        energy_limit_right: float | None,
        random_seed: int,
        ensemble_cls: type[CoordinatedWangLandauEnsemble] = (
            CoordinatedWangLandauEnsemble
        ),
        ensemble_kwargs: Mapping[str, Any] | None = None,
        recency_visits_per_bin: int = 1000,
        dos_snapshot_ratio: float | None = 2.0,
        cluster_expansion_path: str | os.PathLike[str] | None = None,
        sites_by_species: list[dict[int, list[int]]] | None = None,
    ) -> WangLandauReplica:
        """Construct a replica whose ensemble has been restored from `container`."""
        replica = cls(
            cluster_expansion=cluster_expansion,
            atoms=atoms,
            energy_spacing=energy_spacing,
            energy_limit_left=energy_limit_left,
            energy_limit_right=energy_limit_right,
            random_seed=random_seed,
            ensemble_cls=ensemble_cls,
            ensemble_kwargs=ensemble_kwargs,
            recency_visits_per_bin=recency_visits_per_bin,
            dos_snapshot_ratio=dos_snapshot_ratio,
            cluster_expansion_path=cluster_expansion_path,
        )
        replica.restore_state(container, sites_by_species=sites_by_species)
        return replica
