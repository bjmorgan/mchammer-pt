"""Per-window analysis output for multi-walker REWL."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .wl_coordinator import merge_entropies

if TYPE_CHECKING:
    from mchammer.data_containers.wang_landau_data_container import (
        WangLandauDataContainer,
    )


@dataclasses.dataclass(frozen=True)
class WindowResult:
    """Per-window analysis output for REWL.

    Wraps per-walker ``WangLandauDataContainer`` instances and provides
    merged entropy and histogram via the same DataFrame interface as
    ``WangLandauDataContainer``.

    Args:
        energy_limit_left: lower bound of the energy window.
        energy_limit_right: upper bound of the energy window.
        energy_spacing: bin width shared across all walkers.
        containers: per-walker data containers for this window.
    """

    energy_limit_left: float
    energy_limit_right: float
    energy_spacing: float
    containers: tuple[WangLandauDataContainer, ...]

    @property
    def n_walkers(self) -> int:
        """Number of walkers in this window."""
        return len(self.containers)

    def get_entropy(
        self, fill_factor_limit: float | None = None,
    ) -> pd.DataFrame | None:
        """Merged entropy across all walkers.

        Returns a DataFrame with ``energy`` and ``entropy`` columns,
        matching ``WangLandauDataContainer.get_entropy()``. Entropy is
        shifted so the minimum is zero.

        Args:
            fill_factor_limit: if given, each walker contributes the
                entropy recorded at the point when its fill factor
                first reached this limit.

        Returns:
            DataFrame, or ``None`` if no walker has entropy data.
        """
        per_walker: list[dict[int, float]] = []
        for c in self.containers:
            entropy = self._extract_entropy(c, fill_factor_limit)
            if entropy is None:
                return None
            per_walker.append(entropy)
        merged = merge_entropies(per_walker)
        if not merged:
            return pd.DataFrame(columns=["energy", "entropy"])
        bins = sorted(merged.keys())
        energies = self.energy_spacing * np.array(bins, dtype=np.float64)
        values = np.array([merged[b] for b in bins], dtype=np.float64)
        values -= values.min()
        return pd.DataFrame(
            data={"energy": energies, "entropy": values},
            index=bins,
        )

    def get_histogram(self) -> pd.DataFrame | None:
        """Summed histogram across all walkers.

        Returns a DataFrame with ``energy`` and ``histogram`` columns,
        matching ``WangLandauDataContainer.get_histogram()``.

        Returns:
            DataFrame, or ``None`` if no walker has histogram data.
        """
        combined: dict[int, int] = {}
        for c in self.containers:
            histogram = self._extract_histogram(c)
            if histogram is None:
                return None
            for k, v in histogram.items():
                combined[k] = combined.get(k, 0) + v
        if not combined:
            return pd.DataFrame(columns=["energy", "histogram"])
        bins = sorted(combined.keys())
        energies = self.energy_spacing * np.array(bins, dtype=np.float64)
        counts = np.array([combined[b] for b in bins], dtype=np.int64)
        return pd.DataFrame(
            data={"energy": energies, "histogram": counts},
            index=bins,
        )

    # _extract_entropy and _extract_histogram access _last_state on
    # WangLandauDataContainer. No public API exposes raw (unshifted)
    # entropy or the bin-index histogram dict needed for cross-walker
    # merging. Coupling is pinned to these two methods.

    @staticmethod
    def _extract_histogram(
        container: WangLandauDataContainer,
    ) -> dict[int, int] | None:
        """Extract the raw histogram dict from a container.

        Args:
            container: per-walker data container.

        Returns:
            Histogram dict (bin index -> count), or ``None`` if absent.
        """
        histogram = container._last_state.get("histogram")
        if histogram is None:
            return None
        return {k: int(v) for k, v in histogram.items()}

    @staticmethod
    def _extract_entropy(
        container: WangLandauDataContainer,
        fill_factor_limit: float | None,
    ) -> dict[int, float] | None:
        """Extract the entropy dict from a container, respecting fill_factor_limit.

        When ``fill_factor_limit`` is ``None``, returns the current
        entropy. When set, first checks that the container's current
        fill factor has reached the limit (returns ``None`` if not),
        then scans the union of ``fill_factor_history`` and
        ``fill_factor_snapshots`` by ascending MC step, returning the
        entropy at the first step whose fill factor is at or below the
        limit. Returns ``None`` if both stores are empty or contain no
        matching step.

        Args:
            container: per-walker data container.
            fill_factor_limit: if given, returns the entropy snapshot
                from the chronologically first step whose fill factor
                is at or below this limit.

        Returns:
            Entropy dict, or ``None`` if data is absent or the limit
            is unmet.
        """
        last_state = container._last_state
        if "entropy" not in last_state:
            return None
        entropy = last_state["entropy"]
        if fill_factor_limit is not None:
            history = last_state.get("entropy_history", {})
            snapshots = last_state.get("entropy_snapshots", {})
            if not history and not snapshots:
                return None
            if last_state.get("fill_factor", 1.0) > fill_factor_limit:
                return None
            ff_history = last_state.get("fill_factor_history", {})
            ff_snapshots = last_state.get("fill_factor_snapshots", {})
            # Coerce step keys to int before merging/sorting. A container
            # read back via ``WangLandauDataContainer.read`` has its
            # halving history int-keyed (icet coerces the fields it knows)
            # but leaves the newer snapshot maps string-keyed, so an
            # un-coerced union would mix int and str keys and
            # ``sorted(...)`` would raise ``TypeError``.
            ff_map = {
                int(s): ff
                for s, ff in {**ff_history, **ff_snapshots}.items()
            }
            entropy_map = {
                int(s): ent
                for s, ent in {**history, **snapshots}.items()
            }
            # `fill_factor_history` carries the initial fill factor at
            # step 0, for which there is no paired `entropy_history`
            # entry (upstream seeds the two unevenly). Skip any step
            # without a paired entropy so the scan never indexes a
            # missing key -- e.g. a fill_factor_limit >= 1.0 would
            # otherwise match the unpaired step 0.
            for step in sorted(ff_map):
                if step in entropy_map and ff_map[step] <= fill_factor_limit:
                    # Bin keys of a snapshot from a raw container read are
                    # also string-keyed; coerce so the merge downstream
                    # sees integer bins.
                    return {
                        int(b): float(v)
                        for b, v in entropy_map[step].items()
                    }
            return None
        return dict(entropy)
