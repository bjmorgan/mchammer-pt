"""Energy-binned microcanonical moment recorder for frozen-g WL runs.

Owns one mchammer observer and accumulates count/sum/sum2/sum4 of its
scalar outputs per energy bin. Passive: never mutates the ensemble.

Signature discovery
-------------------
``BaseObserver`` does not declare S (the number of scalars returned)
in its interface -- S is only known after ``get_observable`` is first
called. The recorder sets ``_names`` on the first ``record`` call. On
restore via ``from_state``, ``_names`` and ``_size`` are set from the
stored state; the attached observer's actual signature is validated on
the first post-restore ``record`` call, which raises ``ValueError`` on
size or name disagreement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from mchammer.observers.base_observer import BaseObserver


def _coerce_scalars(value: Any) -> tuple[np.ndarray, list[str] | None]:
    """Coerce an observer return value to a 1-D float array and optional names.

    Args:
        value: The raw return value from ``observer.get_observable``.

    Returns:
        A tuple of ``(array, names)`` where ``array`` is a 1-D float
        ndarray and ``names`` is either a list of string keys (for Mapping
        inputs) or ``None`` (for scalar/sequence inputs).

    Raises:
        TypeError: If ``value`` is a string or bytes, which cannot
            represent a physical observable.
    """
    if isinstance(value, Mapping):
        names = sorted(value)
        return np.asarray([value[k] for k in names], float), [str(n) for n in names]
    if isinstance(value, (str, bytes)):
        raise TypeError("observer returned a string; expected scalar/sequence/mapping")
    if isinstance(value, Sequence):
        return np.asarray(value, float), None
    return np.asarray([value], float), None


class EnergyBinnedObservableRecorder:
    """Accumulates microcanonical moments of a single observer's output per energy bin.

    For each energy bin ``b``, tracks count, sum, sum-of-squares, and
    sum-of-fourth-powers of the observer's S-vector output, enabling
    computation of microcanonical means and variances during post-processing.

    Signature (S and observable names) is inferred from the first
    ``record`` call or from a serialised state on restore.

    Args:
        observer: Any ``mchammer.BaseObserver`` subclass whose
            ``get_observable`` returns a scalar, sequence, or Mapping.
    """

    def __init__(self, observer: BaseObserver) -> None:
        self._observer = observer
        # Set on first record or from_state; None means not yet determined.
        self._names: tuple[str, ...] | None = None
        self._size: int | None = None
        # Per-bin accumulators: bin_index -> ndarray of length S
        self._count: dict[int, int] = {}
        self._sum: dict[int, np.ndarray] = {}
        self._sum2: dict[int, np.ndarray] = {}
        self._sum4: dict[int, np.ndarray] = {}
        self._skipped: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tag(self) -> str:
        """Human-readable tag from the underlying observer."""
        return self._observer.tag

    @property
    def interval(self) -> int:
        """Observation interval from the underlying observer."""
        return self._observer.interval

    @property
    def signature(self) -> tuple[int, tuple[str, ...]]:
        """Current ``(S, names)`` signature.

        Returns:
            A tuple of ``(S, names)`` where S is the number of scalars
            and ``names`` is a tuple of string labels. Raises
            ``RuntimeError`` if no observation has been recorded yet.

        Raises:
            RuntimeError: If the signature has not been determined yet
                (no observations recorded and no state restored).
        """
        if self._size is None or self._names is None:
            raise RuntimeError(
                "signature not yet determined; call record() first "
                "or restore from state"
            )
        return self._size, self._names

    # ------------------------------------------------------------------
    # Core accumulation
    # ------------------------------------------------------------------

    def record(self, structure: Any, bin_index: int) -> None:
        """Observe ``structure`` and add moments into ``bin_index``.

        Evaluates the observer, coerces the result to a float array, drops
        non-finite observations (incrementing the skipped tally instead),
        and accumulates count/sum/sum2/sum4 into ``bin_index``.

        Args:
            structure: The current atomic structure, forwarded to the
                observer's ``get_observable`` method.
            bin_index: Integer index of the energy bin that the structure
                currently occupies.

        Raises:
            ValueError: If the observation's size or names disagree with
                the previously established signature.
        """
        raw = self._observer.get_observable(structure)
        vals, names_from_obs = _coerce_scalars(raw)

        # Determine names: Mapping provides them; otherwise use tag-based defaults.
        if names_from_obs is not None:
            resolved_names = tuple(names_from_obs)
        elif len(vals) == 1:
            resolved_names = (self.tag,)
        else:
            resolved_names = tuple(f"{self.tag}_{i}" for i in range(len(vals)))

        # Set or validate signature.
        if self._names is None:
            self._names = resolved_names
            self._size = len(vals)
        else:
            if len(vals) != self._size or resolved_names != self._names:
                raise ValueError(
                    f"observer signature mismatch: expected {self._size} scalars "
                    f"with names {self._names!r}, got {len(vals)} scalars "
                    f"with names {resolved_names!r}"
                )

        # Drop non-finite observations.
        if not np.all(np.isfinite(vals)):
            self._skipped[bin_index] = self._skipped.get(bin_index, 0) + 1
            return

        # Create zero accumulators for new bins.
        if bin_index not in self._count:
            s = self._size  # guaranteed non-None after signature check above
            assert s is not None  # for type narrowing
            self._count[bin_index] = 0
            self._sum[bin_index] = np.zeros(s)
            self._sum2[bin_index] = np.zeros(s)
            self._sum4[bin_index] = np.zeros(s)

        self._count[bin_index] += 1
        self._sum[bin_index] += vals
        self._sum2[bin_index] += vals**2
        self._sum4[bin_index] += vals**4

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Serialise the recorder's accumulated state to a columnar dict.

        Bins are sorted ascending. All per-bin arrays are stored as nested
        lists for JSON/HDF5 compatibility (no int-key coercion for bin
        indices).

        Returns:
            A dict with keys:
            ``tag``, ``interval``, ``names``, ``bins``, ``count``,
            ``sum``, ``sum2``, ``sum4``, ``skipped``.
        """
        bins_sorted = sorted(self._count)
        return {
            "tag": self.tag,
            "interval": self.interval,
            "names": list(self._names) if self._names is not None else [],
            "bins": bins_sorted,
            "count": [self._count[b] for b in bins_sorted],
            "sum": [self._sum[b].tolist() for b in bins_sorted],
            "sum2": [self._sum2[b].tolist() for b in bins_sorted],
            "sum4": [self._sum4[b].tolist() for b in bins_sorted],
            "skipped": dict(self._skipped),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        observer: BaseObserver,
    ) -> EnergyBinnedObservableRecorder:
        """Restore a recorder from a serialised state dict.

        After restoration, subsequent ``record`` calls ADD to the restored
        accumulators rather than resetting them. The observer's signature
        (size and names) is set from the stored state and validated against
        the attached observer on the first post-restore ``record`` call,
        which raises ``ValueError`` on size or name disagreement.

        Args:
            state: A dict as produced by ``to_state``.
            observer: The observer to attach.

        Returns:
            A new ``EnergyBinnedObservableRecorder`` seeded with the
            restored data.

        Raises:
            ValueError: If the stored state is internally inconsistent
                (bins and count lengths disagree).
        """
        stored_names = tuple(state["names"])
        stored_size = len(stored_names)

        if stored_size == 0:
            # No observations were ever recorded; nothing to seed.
            rec = cls(observer)
            rec._skipped = dict(state.get("skipped", {}))
            return rec

        if len(state["bins"]) != len(state["count"]):
            raise ValueError("corrupt state: bins and count lengths disagree")

        rec = cls(observer)
        rec._names = stored_names
        rec._size = stored_size

        bins: list[int] = list(state["bins"])
        counts: list[int] = list(state["count"])
        sums: list[list[float]] = list(state["sum"])
        sum2s: list[list[float]] = list(state["sum2"])
        sum4s: list[list[float]] = list(state["sum4"])

        for b, cnt, s, s2, s4 in zip(bins, counts, sums, sum2s, sum4s, strict=True):
            rec._count[b] = cnt
            rec._sum[b] = np.array(s, float)
            rec._sum2[b] = np.array(s2, float)
            rec._sum4[b] = np.array(s4, float)

        rec._skipped = {int(k): int(v) for k, v in state.get("skipped", {}).items()}

        return rec
