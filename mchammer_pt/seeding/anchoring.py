"""Decide which energy end each window is anchored to.

Lower-half windows are seeded from the ground-state anchor and the WL
window search climbs into the band; upper-half windows are seeded from a
fresh random fill and the search settles down into the band. The split
is material-agnostic: it depends only on energies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

Anchor = Literal["bottom", "top"]


def assign_anchors(
    windows: Sequence[tuple[float | None, float | None]],
    e_gs: float,
    e_top: float,
) -> list[Anchor]:
    """Assign each window a ``"bottom"`` or ``"top"`` anchor by energy.

    An unbounded left edge forces ``"bottom"`` (the window reaches down
    to the ground state); an unbounded right edge forces ``"top"``.
    Otherwise the window centre ``(lo + hi) / 2`` is compared against the
    midpoint ``(e_gs + e_top) / 2``: strictly below -> ``"bottom"``,
    else ``"top"``.

    Args:
        windows: per-window ``(lo, hi)`` edges; ``None`` means unbounded.
        e_gs: energy of the bottom anchor (ground state).
        e_top: representative high (disordered-limit) energy.

    Returns:
        One anchor label per window, in window order.
    """
    midpoint = 0.5 * (e_gs + e_top)
    anchors: list[Anchor] = []
    for lo, hi in windows:
        if lo is None:
            anchors.append("bottom")
        elif hi is None:
            anchors.append("top")
        else:
            centre = 0.5 * (lo + hi)
            anchors.append("bottom" if centre < midpoint else "top")
    return anchors


def validate_anchor_override(
    anchors: Sequence[str],
    n_windows: int,
) -> list[Anchor]:
    """Validate a caller-supplied explicit anchor override.

    Args:
        anchors: one ``"bottom"``/``"top"`` per window.
        n_windows: expected number of windows.

    Returns:
        The anchors as a list.

    Raises:
        ValueError: on length mismatch or an invalid label.
    """
    anchors = list(anchors)
    if len(anchors) != n_windows:
        raise ValueError(
            f"anchors has length {len(anchors)} but there are "
            f"{n_windows} windows; supply one anchor per window."
        )
    for i, a in enumerate(anchors):
        if a not in ("bottom", "top"):
            raise ValueError(
                f"anchors[{i}] = {a!r} is invalid; each anchor must be "
                f"'bottom' or 'top'."
            )
    return anchors  # type: ignore[return-value]
