"""In-process serial replica advance."""

from __future__ import annotations

from ..replica import Replica


class SerialBackend:
    """Advances replicas sequentially in the calling process.

    Use for debugging, for small runs, or when the per-cycle walltime
    is dominated by something other than MC time.
    """

    def advance_all(self, replicas: list[Replica], n_steps: int) -> None:
        for replica in replicas:
            replica.advance(n_steps)

    def shutdown(self) -> None:
        """Nothing to do: the serial backend holds no resources."""
        return None
