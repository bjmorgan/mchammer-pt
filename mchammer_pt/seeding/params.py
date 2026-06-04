"""Search knobs for the REWL window-seeding search."""

from __future__ import annotations

import numbers
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSearchParams:
    """Tunable knobs for :func:`mchammer_pt.seed_window_configs`.

    Args:
        window_search_penalty: bias coefficient for the into-window
            search; larger drives a configuration into the band more
            aggressively (it is ``1/T`` in spirit). Must be > 0.
        walk_sweeps: per-walk step budget in sweeps; the actual MC step
            count is ``walk_sweeps * n_sites``. Covers the window search
            plus the in-band diffusion that decorrelates the harvested
            seed from its entry point. Must be >= 1.
        max_walks_per_window: number of confined-walk rounds allowed per
            window before the search gives up and raises. Must be >= 1
            (and >= the largest per-window count for that window to be
            fillable).
        n_workers: spawn-pool size. ``None`` uses ``os.cpu_count()``.
            Must be >= 1 when given.

    Raises:
        ValueError: if any knob is out of range.
    """

    window_search_penalty: float = 2.0
    walk_sweeps: int = 50
    max_walks_per_window: int = 20
    n_workers: int | None = None

    def __post_init__(self) -> None:
        if not self.window_search_penalty > 0:
            raise ValueError(
                f"window_search_penalty must be > 0; got "
                f"{self.window_search_penalty}"
            )
        for name in ("walk_sweeps", "max_walks_per_window"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral):
                raise ValueError(
                    f"{name} must be an integer; got {type(value).__name__}."
                )
            if value < 1:
                raise ValueError(f"{name} must be >= 1; got {value}")
        if self.n_workers is not None:
            if not isinstance(self.n_workers, numbers.Integral):
                raise ValueError(
                    f"n_workers must be an integer or None; got "
                    f"{type(self.n_workers).__name__}."
                )
            if self.n_workers < 1:
                raise ValueError(
                    f"n_workers must be >= 1 when given; got {self.n_workers}"
                )
