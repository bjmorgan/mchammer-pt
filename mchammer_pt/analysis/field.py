"""Reconstruct a canonical field map from reweighted per-pixel means.

A fixed-size ``np.ndarray`` observer (``S = math.prod(shape)`` pixels) rides
the scalar observable pipeline as ``S`` pixels: the recorder flattens the
array C-order via :meth:`numpy.ndarray.ravel` and labels the pixels
``{tag}_0 .. {tag}_{S-1}``. After
:func:`mchammer_pt.analysis.observables.reweight_observables`, each pixel
appears as a ``{tag}_{i}_mean`` column. :func:`field_map` reads those columns
back in the same C-order and folds them into an ``(n_T, *shape)`` array -- the
canonical field map.

The helper owns the ravel-order convention, so a downstream caller
reconstructs the field without re-deriving mchammer-pt's internal flattening
order.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def field_map(
    canonical: pd.DataFrame,
    tag: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Fold reweighted per-pixel means back into a canonical field map.

    Intended for multi-pixel fields (``S = math.prod(shape) >= 2``). A
    one-element observer is recorded under the scalar name ``{tag}`` rather
    than ``{tag}_0``, so this helper does not apply to it; read its
    ``{tag}_mean`` column directly.

    Args:
        canonical: Output of
            :func:`mchammer_pt.analysis.observables.reweight_observables`,
            carrying a ``T_K`` column and one ``{tag}_{i}_mean`` column per
            pixel ``i = 0 .. S-1``. Sibling ``{tag}_{i}_sq_mean`` and
            ``{tag}_{i}_binder`` columns, if present, are ignored.
        tag: The field observer's tag; the recorder labels its pixels
            ``{tag}_{i}``.
        shape: The field's shape, e.g. ``(N, N)``. ``math.prod(shape)`` must
            equal the number of recorded pixels ``S``.

    Returns:
        ``(temperatures, values)``. ``temperatures`` is the ``(n_T,)`` array
        from the ``T_K`` column; ``values`` is an ``(n_T, *shape)`` array of
        the canonical field mean -- the ``{tag}_{i}_mean`` columns read in C
        (ravel) order and reshaped to ``shape``.

    Raises:
        ValueError: if ``canonical`` lacks a ``T_K`` column; if ``shape`` has
            no pixels; or if the pixel-mean columns are not exactly
            ``{tag}_0_mean .. {tag}_{S-1}_mean`` for ``S = math.prod(shape)``
            (a wrong ``tag``, a ``shape`` that mis-sizes the field, or a
            partial column set all trip this).
    """
    if "T_K" not in canonical.columns:
        raise ValueError("canonical frame has no 'T_K' column")

    size = math.prod(shape)
    if size <= 0:
        raise ValueError(f"shape {shape!r} encloses no pixels (prod == {size})")

    pixel = re.compile(rf"^{re.escape(tag)}_(\d+)_mean$")
    indices = sorted(
        int(m.group(1)) for c in canonical.columns if (m := pixel.match(c))
    )
    if indices != list(range(size)):
        raise ValueError(
            f"field {tag!r} with shape {shape} expects {size} contiguous "
            f"pixel-mean columns {tag}_0_mean .. {tag}_{size - 1}_mean; found "
            f"index set {indices!r}. Check that math.prod(shape) matches the "
            f"recorded field size and that this is a reweight_observables "
            f"output for tag {tag!r}."
        )

    temperatures = canonical["T_K"].to_numpy(dtype=float)
    n_T = len(temperatures)
    flat = np.column_stack(
        [canonical[f"{tag}_{i}_mean"].to_numpy(dtype=float) for i in range(size)]
    )  # (n_T, S)
    values = flat.reshape((n_T, *shape))
    return temperatures, values
