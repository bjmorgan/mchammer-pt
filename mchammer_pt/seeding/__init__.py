"""Material-agnostic REWL window-seeding search.

Fills each energy window with K distinct in-band starting
configurations for a (multiwalker) replica-exchange Wang-Landau run.
"""

from __future__ import annotations

from .params import SeedSearchParams
from .search import seed_window_configs

__all__ = ["SeedSearchParams", "seed_window_configs"]
