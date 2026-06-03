"""Expand per-window WL initial structures into per-walker structures."""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms


def expand_initial_structures(
    atoms: Sequence[Atoms | Sequence[Atoms]],
    walkers_per_window: Sequence[int],
) -> list[list[Atoms]]:
    """Expand per-window initial structures into per-walker structures.

    Each element of ``atoms`` is either a single :class:`ase.Atoms`
    (broadcast: every walker in that window starts from it) or a
    sequence of ``Atoms`` of length ``walkers_per_window[w]`` (one per
    walker, in walker order). Windows may mix the two forms.

    No copying happens here. Isolation between walkers is enforced at
    per-walker construction (``WangLandauReplica`` copies its ``atoms``;
    ``AtomsSpec.from_atoms`` copies every array), so a broadcast element
    is returned as repeated references and still yields independent
    walkers downstream.

    Args:
        atoms: per-window structures. ``len(atoms)`` must equal
            ``len(walkers_per_window)``; the caller (the orchestrator)
            validates the window count before calling.
        walkers_per_window: walker count for each window.

    Returns:
        ``out[w][j]`` -- the starting structure for walker ``j`` of
        window ``w``. Each inner list has length
        ``walkers_per_window[w]``.

    Raises:
        ValueError: a per-window sequence is empty, contains a
            non-``Atoms`` element, or has a length other than that
            window's walker count.
    """
    out: list[list[Atoms]] = []
    for w, (element, n_walkers) in enumerate(
        zip(atoms, walkers_per_window, strict=True)
    ):
        if isinstance(element, Atoms):
            out.append([element] * n_walkers)
            continue
        walkers = list(element)
        if not walkers:
            raise ValueError(
                f"atoms[{w}] is an empty sequence; supply one Atoms per "
                f"walker (window {w} has {n_walkers}) or a single Atoms "
                f"to broadcast."
            )
        for k, structure in enumerate(walkers):
            if not isinstance(structure, Atoms):
                raise ValueError(
                    f"atoms[{w}][{k}] is {type(structure).__name__}, not an "
                    f"Atoms; each per-walker entry must be an ase.Atoms."
                )
        if len(walkers) != n_walkers:
            raise ValueError(
                f"atoms[{w}] has {len(walkers)} structures but window {w} "
                f"has {n_walkers} walkers; supply exactly one Atoms per "
                f"walker or a single Atoms to broadcast."
            )
        out.append(walkers)
    return out
