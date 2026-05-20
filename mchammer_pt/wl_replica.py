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

from .wl_coordinator import CoordinatorPlan, Phase, Schedule, WalkerPostBlockState
from .wl_ensemble import CoordinatedWangLandauEnsemble

_RESERVED_ENSEMBLE_KWARGS: frozenset[str] = frozenset(
    {
        "structure",
        "calculator",
        "energy_spacing",
        "energy_limit_left",
        "energy_limit_right",
        "random_seed",
        "dc_filename",
    }
)

# `_last_state` fields whose dict keys are integer bin indices.
# JSON round-trips coerce these to strings; the conversion has to be
# reversed before mchammer's `_restart_ensemble` reads them. Matches
# the set `WangLandauDataContainer.read` converts upstream.
_WL_INT_KEY_FIELDS: frozenset[str] = frozenset(
    {"histogram", "entropy", "fill_factor_history", "entropy_history"}
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
    def reroll_exchange_idx(self) -> None: ...
    def is_flat(self) -> bool: ...
    def advance(self, n_steps: int) -> None: ...
    def current_energy(self) -> float: ...
    def current_occupations(self) -> np.ndarray: ...
    def set_occupations(self, occupations: np.ndarray) -> None: ...
    def log_g(self, energy: float) -> float: ...
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
                f"random_seed from their dedicated parameters; "
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

        # Seed the starting bin into the histogram and entropy so the
        # flatness gate is aware of it from construction. Without this,
        # a walker whose first move leaves bin_init and never returns
        # would have bin_init absent from the dict; the flatness check
        # then operates on the saturated subset of bins actually visited
        # and may halve prematurely. See
        # docs/superpowers/specs/2026-05-20-wl-known-bin-seed-design.md.
        e._histogram.setdefault(bin_init, 0)
        e._entropy.setdefault(bin_init, 0.0)

        self.walker_states: tuple[WalkerPostBlockState, ...] = (
            WalkerPostBlockState(
                is_flat=False,
                fill_factor=1.0,
                entropy={},
                step=0,
                window_entry_step=None,
                histogram={},
                reached_energy_window=False,
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
        bin_idx = e._get_bin_index(energy)
        if bin_idx is None or not e._inside_energy_window(bin_idx):
            return -float(np.inf)
        return float(e._entropy.get(bin_idx, 0.0))

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
        # Seed the new bin so the flatness gate sees it. See the
        # docstring of __init__ and
        # docs/superpowers/specs/2026-05-20-wl-known-bin-seed-design.md.
        e._histogram.setdefault(new_bin, 0)
        e._entropy.setdefault(new_bin, 0.0)
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

    def is_flat(self) -> bool:
        """Return ``True`` if this walker's own histogram is flat.

        Uses mchammer's flatness criterion: every bin's count is
        ``>= flatness_limit * mean(counts)``. Walkers that have not
        yet entered the window return ``False``.
        """
        e = self._ensemble
        if not e._reached_energy_window:
            return False
        if not e._histogram:
            return False
        histogram = np.array(list(e._histogram.values()))
        if histogram.size == 0:
            return False
        limit = e._flatness_limit * np.average(histogram)
        return bool(np.all(histogram >= limit))

    def _snapshot(self) -> WalkerPostBlockState:
        """Read live ensemble state into a WalkerPostBlockState."""
        e = self._ensemble
        return WalkerPostBlockState(
            is_flat=self.is_flat(),
            fill_factor=float(e._fill_factor),
            entropy=dict(e._entropy),
            step=int(e.step),
            window_entry_step=(
                None if e._window_entry_step is None
                else int(e._window_entry_step)
            ),
            histogram=dict(e._histogram),
            reached_energy_window=bool(e._reached_energy_window),
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

    def reroll_exchange_idx(self) -> None:
        """No-op: a single-walker slot has no exchange index to re-roll.

        Present to satisfy ``WangLandauSlot``: the pool calls this on
        every slot after applying a coordinator plan.
        """

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

        Returns fill_factor, halvings, histogram, bins_visited,
        bins_known, converged. For a single-walker replica
        ``flatness_mode`` and ``per_walker_flat_min`` are omitted
        (the progress reporter falls through to the pooled
        computation, which is exact for n_walkers == 1).

        ``bins_visited`` is the count of bins whose current histogram
        value is > 0 (the bins active in the current halving phase);
        ``bins_known`` is ``len(_histogram)`` (all bins the flatness
        gate considers, including seeded-but-unvisited entries).
        """
        e = self._ensemble
        histogram = dict(e._histogram)
        return {
            "fill_factor": float(e._fill_factor),
            "halvings": max(0, len(e._fill_factor_history) - 1),
            "histogram": histogram,
            "bins_visited": sum(1 for v in histogram.values() if v > 0),
            "bins_known": len(histogram),
            "converged": self.converged,
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
        histogram, fill_factor, fill_factor_history, entropy_history)
        and the 1/t-schedule fields (schedule, phase,
        window_entry_step). Idempotent.
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
        # Seed the restored bin so the flatness gate sees it from
        # restore-time onward. The saved _last_state may already
        # contain this bin (with a real count, restored via
        # _restart_ensemble); setdefault preserves that. The seed
        # matters when the saved histogram is empty (pre-step
        # checkpoint) or when the restored bin was never visited in
        # the saved run.
        e._histogram.setdefault(new_bin, 0)
        e._entropy.setdefault(new_bin, 0.0)
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
            cluster_expansion_path=cluster_expansion_path,
        )
        replica.restore_state(container, sites_by_species=sites_by_species)
        return replica
