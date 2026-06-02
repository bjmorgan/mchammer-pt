"""Synthetic DOS fixtures used by the coexistence tests.

Each builder returns a ``(energy, entropy)`` DataFrame matching the
contract of ``mchammer_pt.analysis.dos.stitch_entropy``'s output: a
uniform energy grid in eV and ``entropy`` interpreted as ``ln g(E)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lattice_like_dos(
    *,
    a: float,
    beta_c: float,
    c: float,
    E_min: float,
    E_max: float,
    energy_spacing: float,
) -> pd.DataFrame:
    """Build a DOS with monotone ln g whose canonical P(E|T) is bimodal.

    Constructs the DOS by designing the canonical phi function
    ``phi(E) = beta_c * E - ln g(E)`` as a quartic double-well::

        phi(E) = a * (E ** 2 - c ** 2) ** 2

    so that

    .. code-block:: python

        ln g(E) = beta_c * E - a * (E ** 2 - c ** 2) ** 2

    At ``beta = beta_c``, phi is the designed double-well with minima
    at ``E = +/- c`` and a maximum at ``E = 0``. ``d(ln g)/dE`` is
    non-monotonic (local minimum at ``E = -c/sqrt(3)``, local maximum
    at ``E = +c/sqrt(3)``) — exactly the slope-oscillation structure
    that produces bimodal ``P(E|T)`` in a finite beta window around
    ``beta_c``. As ``beta`` moves away from ``beta_c`` the linear
    perturbation tilts phi; one well eventually swallows the other
    and bimodality is lost. The bimodal-beta window has half-width
    ``8 * a * c ** 3 / (3 * sqrt(3))``.

    ``ln g(E)`` is monotonically increasing in ``E`` over the central
    range where the cubic term has not yet overtaken the linear
    ``beta_c * E`` term. The boundary of that range is the real root
    of ``E ** 3 - c ** 2 * E - beta_c / (4 * a) = 0``. For
    ``c = 1, beta_c = 10, a = 1`` this root is at ``E ≈ 1.60``, so a
    safe energy range is ``[E_min, E_max] = [-1.5, 1.5]``.

    Returned DataFrame: ``energy`` (eV) and ``entropy`` (``ln g``)
    columns on a uniform grid spanning ``[E_min, E_max]`` with
    spacing ``energy_spacing``. ``entropy`` is rebased so its
    minimum is zero (matching ``stitch_entropy``'s output
    convention).
    """
    n_bins = int(round((E_max - E_min) / energy_spacing)) + 1
    energies = E_min + np.arange(n_bins) * energy_spacing
    ln_g = beta_c * energies - a * (energies ** 2 - c ** 2) ** 2
    ln_g -= ln_g.min()
    return pd.DataFrame({"energy": energies, "entropy": ln_g})
