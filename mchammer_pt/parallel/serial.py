"""In-process replica pool: advances replicas sequentially in the caller."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from mchammer.data_containers.base_data_container import (  # type: ignore[import-untyped]
    BaseDataContainer,
)
from mchammer.observers.base_observer import (  # type: ignore[import-untyped]
    BaseObserver,
)

from ..replica import Replica


class SerialPool:
    """Advances replicas sequentially in the calling process.

    The pool owns a list of `Replica` instances and exposes the full
    `ObservablePool` surface. Use for debugging, for small runs, or
    when the per-cycle walltime is dominated by something other than
    MC time.
    """

    def __init__(self, replicas: Sequence[Replica]) -> None:
        self._replicas: list[Replica] = list(replicas)

    def __len__(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> list[Replica]:
        """The pool's `Replica` instances. Returns a copy."""
        return list(self._replicas)

    @property
    def temperatures(self) -> list[float]:
        return [r.temperature for r in self._replicas]

    def advance_all(self, n_steps: int) -> None:
        for replica in self._replicas:
            replica.advance(n_steps)

    def current_energies(self) -> np.ndarray:
        return np.array([r.current_energy() for r in self._replicas], dtype=np.float64)

    def current_energy(self, i: int) -> float:
        return self._replicas[i].current_energy()

    def current_occupations(self, i: int) -> np.ndarray:
        return self._replicas[i].current_occupations()

    def swap_configurations(self, i: int, j: int) -> None:
        occ_i = self._replicas[i].current_occupations()
        occ_j = self._replicas[j].current_occupations()
        self._replicas[i].set_occupations(occ_j)
        self._replicas[j].set_occupations(occ_i)

    def attach_observer(
        self,
        observer: BaseObserver,
        replicas: Sequence[int] | Literal["all"] = "all",
    ) -> None:
        target_indices = (
            range(len(self._replicas))
            if replicas == "all"
            else [int(i) for i in replicas]
        )
        for i in target_indices:
            self._replicas[i].attach_mchammer_observer(observer)

    def data_containers(self) -> list[BaseDataContainer]:
        return [r.data_container() for r in self._replicas]

    def shutdown(self) -> None:
        # Nothing to release: the serial pool holds no external resources.
        return None
