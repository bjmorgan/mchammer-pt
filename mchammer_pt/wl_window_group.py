"""Multi-walker Wang-Landau window group."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from mchammer.observers.base_observer import BaseObserver

from .wl_replica import WangLandauReplica

_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED = (
    "checkpointing is not yet supported for n_walkers_per_window > 1; "
    "pass data_container_file=None and avoid save_checkpoint() / "
    "attach_checkpoint_writer() when using multiple walkers per window."
)


class WalkerPostBlockState(NamedTuple):
    """State a worker reports after each ``ADVANCE``.

    All five fields are captured from a single worker reply at one
    MC step, so the coordinator can use them as a consistent
    snapshot. Read by the coordinator to decide whether to halve,
    merge entropies, or flip the BP phase.
    """

    is_flat: bool
    fill_factor: float
    entropy: dict[int, float]
    step: int
    window_entry_step: int | None

SyncPolicy = Literal["block", "halving"]
"""Entropy-sharing cadence between walkers in a multi-walker window.

- ``"block"``: merge entropies after every block.
- ``"halving"``: merge entropies only at collective halving events
  (Vogel et al. 2013 multi-walker REWL).

The cadence applies only during the halving phase. In the 1/t phase
walkers merge entropies every block regardless of ``sync_policy``,
because no flatness gate exists there and the independence argument
that motivates halving-only cadence does not apply.
"""


_VALID_SYNC_POLICIES: tuple[str, ...] = ("block", "halving")


def _validate_sync_policy(sync_policy: Any) -> None:
    if sync_policy not in _VALID_SYNC_POLICIES:
        raise ValueError(
            f"sync_policy must be one of {_VALID_SYNC_POLICIES}; "
            f"got {sync_policy!r}"
        )


def decide_collective_halve(flags: list[bool]) -> bool:
    """Return ``True`` iff all walkers are flat (collective gate)."""
    if not flags:
        return False
    return all(flags)


def decide_bp_switch(
    phases: list[str], ts: list[int], fs: list[float]
) -> bool:
    """Return ``True`` iff every walker should flip to the 1/t phase.

    The collective Belardinelli-Pereyra switch fires when every walker
    is still in the halving phase and every walker satisfies
    ``1/t > f``.

    Args:
        phases: per-walker ``_phase`` strings.
        ts: per-walker ``step - _window_entry_step + 1``.
        fs: per-walker ``_fill_factor`` after the collective halve.
    """
    if not phases:
        return False
    if any(p != "halving" for p in phases):
        return False
    return all((1.0 / t) > f for t, f in zip(ts, fs, strict=True))


def merge_entropies(
    entropies: list[dict[int, float]],
) -> dict[int, float]:
    """Combine per-walker entropy dicts into a single window estimate.

    Each walker's ``_entropy`` carries a private additive constant (from
    icet's periodic min-shift in ``_update_entropy`` and from
    independent accumulation between merges). Naive averaging would
    combine values whose additive constants differ; this function
    aligns walkers first by subtracting each walker's mean computed
    over the *common* set of bins all walkers visited, then averages
    bin-wise over the walkers that visited each bin. The result is
    finally shifted so ``min(merged) == 0`` (the icet convention,
    matching ``_update_entropy``'s periodic reshift).

    Args:
        entropies: list of ``{bin_index: entropy_value}`` dicts from
            each walker. Empty dicts (walkers that have not entered
            their window) are filtered out before processing.

    Returns:
        Merged entropy dict with ``min(merged) == 0``. Empty if no
        walker has entered the window.

    Raises:
        RuntimeError: if no bin is visited by every (filtered) walker,
            so rebasing across walkers is ill-defined.
    """
    # Filter out walkers with empty entropy dicts.
    visited = [e for e in entropies if e]
    if not visited:
        return {}
    if len(visited) == 1:
        # Single-walker fast path; min-shift to icet convention.
        only = visited[0]
        shift = min(only.values())
        return {b: v - shift for b, v in only.items()}

    # Intersection of bins visited by every walker.
    common = set(visited[0].keys())
    for e in visited[1:]:
        common &= e.keys()
    if not common:
        raise RuntimeError(
            "merge_entropies: walker coverage has no common bin; "
            "cannot rebase across walkers."
        )

    # Per-walker mean over the common bins; subtract from each walker.
    rebased: list[dict[int, float]] = []
    for e in visited:
        offset = sum(e[b] for b in common) / len(common)
        rebased.append({b: v - offset for b, v in e.items()})

    # Bin-wise average over walkers that visited each bin.
    all_bins: set[int] = set()
    for r in rebased:
        all_bins.update(r.keys())
    merged: dict[int, float] = {}
    for b in all_bins:
        contributors = [r[b] for r in rebased if b in r]
        merged[b] = sum(contributors) / len(contributors)

    # Post-shift to icet convention: min(merged) == 0.
    shift = min(merged.values())
    return {b: v - shift for b, v in merged.items()}


if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )


class WangLandauWindowGroup:
    """A group of independent Wang-Landau walkers sharing one energy window.

    Owns the collective halving decision: all walkers must report
    flat against their own histograms before any halve fires. At a
    collective halve, all walkers' fill factors halve in lockstep,
    histograms reset, and entropies are merged bin-wise across the
    group.

    Between halvings, entropy-merge cadence is controlled by
    ``sync_policy``:

    - ``"block"`` (default): merge every block.
    - ``"halving"``: merge only at collective halves (Vogel et al.
      2013).

    Both policies share the same halve gate (all walkers flat).

    Args:
        replicas: pre-constructed WangLandauReplica instances, all
            with the same energy window and energy spacing.
        random_seed: seed for the exchange-walker selection RNG.
        sync_policy: entropy-sharing cadence. See ``SyncPolicy``.
    """

    def __init__(
        self,
        replicas: list[WangLandauReplica],
        *,
        random_seed: int,
        sync_policy: SyncPolicy = "block",
    ) -> None:
        if len(replicas) < 1:
            raise ValueError(
                "WangLandauWindowGroup requires at least one replica"
            )
        if len(replicas) > 1:
            w0 = replicas[0].energy_window
            s0 = replicas[0].energy_spacing
            for r in replicas[1:]:
                if r.energy_window != w0 or r.energy_spacing != s0:
                    raise ValueError(
                        "all replicas in a WangLandauWindowGroup must share "
                        "the same energy window and spacing"
                    )
        _validate_sync_policy(sync_policy)
        self._replicas = list(replicas)
        self._rng = np.random.default_rng(int(random_seed))
        self._exchange_idx: int = 0
        self._sync_policy: SyncPolicy = sync_policy
        # All walkers in a group share _schedule (set via ensemble_kwargs).
        self._schedule: str = self._replicas[0].ensemble._schedule

    def _merge_entropies_into_all(self) -> None:
        """Average ln g bin-wise across all replicas and write back."""
        merged = merge_entropies(
            [dict(r.ensemble._entropy) for r in self._replicas]
        )
        for r in self._replicas:
            r.ensemble._entropy = dict(merged)

    def advance(self, n_steps: int) -> None:
        """Advance all W replicas, then run the coordinator block."""
        for r in self._replicas:
            r.advance(int(n_steps))
        self._run_coordinator_block()
        self._exchange_idx = int(
            self._rng.integers(0, len(self._replicas))
        )

    def _run_coordinator_block(self) -> None:
        """Per-block coordinator routine: flatness check, halve, merge.

        Called after every block of MC steps. Reads each walker's
        current state, applies the sync policy, mutates state in
        place.
        """
        phase = self._replicas[0].ensemble._phase

        if phase == "halving":
            flags = [r.is_flat() for r in self._replicas]
            if decide_collective_halve(flags):
                for r in self._replicas:
                    r.force_halve()
                self._merge_entropies_into_all()
                if self._schedule == "1_over_t":
                    self._maybe_switch_to_one_over_t()
            elif self._sync_policy == "block":
                self._merge_entropies_into_all()
        else:  # 1_over_t phase
            self._merge_entropies_into_all()

    def _maybe_switch_to_one_over_t(self) -> None:
        """Flip every walker to 1/t phase if the collective condition holds.

        Called immediately after a collective halve. If every walker
        satisfies ``1/t > f``, flip the phase and set
        ``_fill_factor = 1/t`` on every walker. Does not write to
        ``_fill_factor_history``: that dict records halve events,
        which share keys with ``_entropy_history``; in the 1/t phase
        ``_fill_factor`` is reconstructed from
        ``step - _window_entry_step + 1``.
        """
        phases = [r.ensemble._phase for r in self._replicas]
        ts: list[int] = []
        fs: list[float] = []
        for r in self._replicas:
            entry = r.ensemble._window_entry_step
            if entry is None:
                return
            ts.append(r.ensemble.step - entry + 1)
            fs.append(float(r.ensemble._fill_factor))
        if decide_bp_switch(phases, ts, fs):
            for r, t in zip(self._replicas, ts, strict=True):
                r.ensemble._phase = "1_over_t"
                r.ensemble._fill_factor = 1.0 / t

    @property
    def ensemble(self) -> Any:
        """The mchammer ensemble for the first walker.

        Used by pool-level metadata queries (ensemble class, observer
        snapshot). All walkers share the same ensemble class and kwargs.
        For read-only metadata queries only; use `attach_mchammer_observer`,
        `attach_observer_class`, or `attach_observer_factory` to add
        observers to all walkers.
        """
        return self._replicas[0].ensemble

    @property
    def energy_window(self) -> tuple[float | None, float | None]:
        return self._replicas[0].energy_window

    @property
    def energy_spacing(self) -> float:
        return self._replicas[0].energy_spacing

    @property
    def cluster_expansion_path(self) -> str | None:
        return self._replicas[0].cluster_expansion_path

    def current_energy(self) -> float:
        return self._replicas[self._exchange_idx].current_energy()

    def current_occupations(self) -> np.ndarray:
        return self._replicas[self._exchange_idx].current_occupations()

    def set_occupations(self, occupations: np.ndarray) -> None:
        self._replicas[self._exchange_idx].set_occupations(occupations)

    def log_g(self, energy: float) -> float:
        return self._replicas[0].log_g(energy)

    @property
    def converged(self) -> bool:
        return all(r.converged for r in self._replicas)

    def is_flat(self) -> bool:
        """Return ``True`` iff every walker in the group is flat."""
        return all(r.is_flat() for r in self._replicas)

    def data_container(self) -> WangLandauDataContainer:
        """Data container for the first walker in the group."""
        return self._replicas[0].data_container()

    def all_data_containers(self) -> list[WangLandauDataContainer]:
        """All per-walker data containers."""
        return [r.data_container() for r in self._replicas]

    def window_stats(self) -> dict[str, Any]:
        """Per-window convergence metrics: fill_factor, halvings, histogram, converged.

        fill_factor and halvings are taken from replica 0 (all in sync after
        advance); histogram is the sum across all walkers.
        """
        e0 = self._replicas[0].ensemble
        combined_hist: dict[int, int] = {}
        for r in self._replicas:
            for k, v in r.ensemble._histogram.items():
                combined_hist[k] = combined_hist.get(k, 0) + v
        return {
            "fill_factor": float(e0._fill_factor),
            "halvings": max(0, len(e0._fill_factor_history) - 1),
            "histogram": combined_hist,
            "converged": self.converged,
        }

    def refresh_last_state(self) -> None:
        """Refresh ``_last_state`` on every walker's container."""
        for r in self._replicas:
            r.refresh_last_state()

    def snapshot_for_checkpoint(self) -> dict[str, Any]:
        if len(self._replicas) > 1:
            raise NotImplementedError(_MULTI_WALKER_CHECKPOINT_NOT_SUPPORTED)
        return self._replicas[0].snapshot_for_checkpoint()

    def attach_mchammer_observer(self, observer: BaseObserver) -> None:
        """Attach observer to all W replicas; each receives its own copy."""
        try:
            blob = pickle.dumps(observer)
        except Exception as exc:
            raise TypeError(
                f"observer of type {type(observer).__name__} is not "
                f"picklable ({exc}); use attach_observer_class instead"
            ) from exc
        for r in self._replicas:
            r.attach_mchammer_observer(pickle.loads(blob))

    def attach_observer_class(
        self,
        cls: type[BaseObserver],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attach a freshly-constructed observer to every walker in the group.

        Each of the W replicas receives its own ``cls(*args, **kwargs)``
        instance, ensuring independent per-walker state.
        """
        for r in self._replicas:
            r.attach_mchammer_observer(cls(*args, **kwargs))

    def attach_observer_factory(
        self,
        factory: Callable[[WangLandauReplica], BaseObserver],
    ) -> None:
        """Attach an observer constructed per walker via ``factory``.

        ``factory(replica)`` is called once per walker and must return a
        fresh ``BaseObserver``.
        """
        for r in self._replicas:
            observer = factory(r)
            if not isinstance(observer, BaseObserver):
                raise TypeError(
                    f"attach_observer_factory: factory returned "
                    f"{type(observer).__name__}, not a BaseObserver"
                )
            r.attach_mchammer_observer(observer)
