"""The seeding entry points are importable from the package root."""

from __future__ import annotations

import subprocess
import sys

import mchammer_pt


def test_import_succeeds_without_mchammer_moves():
    # `mchammer_pt.seeding` imports mchammer_moves (an optional
    # dependency), so the seeding exports are lazy: `import mchammer_pt`
    # must still succeed when mchammer-moves is absent. Simulate absence
    # in a subprocess by blocking the module.
    code = (
        "import sys; sys.modules['mchammer_moves'] = None; "
        "import mchammer_pt; "
        "print('SeedSearchParams' in mchammer_pt.__all__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


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
