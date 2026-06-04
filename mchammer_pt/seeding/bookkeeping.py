"""Per-window harvest: dedup by occupation vector, cap at the target K."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class WindowHarvest:
    """Accumulate up to ``counts[i]`` distinct configs per window.

    Distinctness is by occupation-vector bytes, scoped per window: the
    same configuration may legitimately seed two different (overlapping)
    windows, but never appears twice in one window.
    """

    def __init__(self, counts: Sequence[int]) -> None:
        self._counts = [int(c) for c in counts]
        self._found: list[list[np.ndarray]] = [[] for _ in self._counts]
        self._seen: list[set[bytes]] = [set() for _ in self._counts]

    def record(self, window_idx: int, occupations: np.ndarray) -> bool:
        """Record ``occupations`` into ``window_idx`` if novel and not full.

        Returns ``True`` if the config was appended, ``False`` if the
        window was already full or the config was a duplicate.
        """
        occ = np.asarray(occupations, dtype=int)
        key = occ.tobytes()
        if len(self._found[window_idx]) >= self._counts[window_idx]:
            return False
        if key in self._seen[window_idx]:
            return False
        self._seen[window_idx].add(key)
        self._found[window_idx].append(occ.copy())
        return True

    def is_full(self, window_idx: int) -> bool:
        return len(self._found[window_idx]) >= self._counts[window_idx]

    def all_full(self) -> bool:
        return all(
            len(self._found[i]) >= self._counts[i]
            for i in range(len(self._counts))
        )

    def configs(self, window_idx: int) -> list[np.ndarray]:
        """The recorded occupation vectors for one window (capped order)."""
        return self._found[window_idx][: self._counts[window_idx]]

    def fill_status(self) -> tuple[int, dict[int, str]]:
        """Return ``(n_filled, {short_window: "found/target"})``."""
        n_filled = sum(
            1
            for i in range(len(self._counts))
            if len(self._found[i]) >= self._counts[i]
        )
        short = {
            i: f"{len(self._found[i])}/{self._counts[i]}"
            for i in range(len(self._counts))
            if len(self._found[i]) < self._counts[i]
        }
        return n_filled, short
