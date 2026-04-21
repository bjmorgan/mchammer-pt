"""Backend protocol for advancing replicas concurrently.

The orchestrator delegates the "advance each replica by N MC steps"
phase of each cycle to a Backend. The serial and multiprocessing
backends implement this protocol; future backends (threads, MPI) can
be added without changes to the orchestrator.
"""

from __future__ import annotations

from typing import Protocol

from ..replica import Replica


class Backend(Protocol):
    """Executes replica-advance phases of a PT run."""

    def advance_all(self, replicas: list[Replica], n_steps: int) -> None:
        """Advance every replica by ``n_steps`` trial steps."""
        ...

    def shutdown(self) -> None:
        """Release any resources held by the backend (workers, handles)."""
        ...
