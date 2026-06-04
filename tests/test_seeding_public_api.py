"""The seeding entry points are importable from the package root."""

from __future__ import annotations

import mchammer_pt


def test_public_names_exported():
    assert hasattr(mchammer_pt, "seed_window_configs")
    assert hasattr(mchammer_pt, "SeedSearchParams")
    assert "seed_window_configs" in mchammer_pt.__all__
    assert "SeedSearchParams" in mchammer_pt.__all__


def test_imported_objects_are_the_real_ones():
    from mchammer_pt.seeding import SeedSearchParams as _Params
    from mchammer_pt.seeding import seed_window_configs as _fn

    assert mchammer_pt.seed_window_configs is _fn
    assert mchammer_pt.SeedSearchParams is _Params
