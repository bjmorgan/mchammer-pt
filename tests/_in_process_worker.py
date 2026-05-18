"""In-process ``Connection``-alike for testing pool workers without
spawning a subprocess. Drop-in for ``mp.Pipe`` in tests that do not
depend on the pickle boundary or real-process isolation.

The helper builds a real worker (``WangLandauWorker`` for WL,
``CanonicalWorker`` for canonical) and dispatches each ``send()``
inline through ``worker._handle``, capturing replies into an internal
queue. The worker's full handler table runs against a real replica
instance -- no hand-written fake. Pickle is bypassed; payloads pass
by reference.

Tests that must exercise the real ``mp.Pipe`` pickle boundary or
real-process isolation cannot use this helper -- spawn a subprocess
directly (see ``tests/test_wl_pool.py::_spawn_wl_worker``).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from mchammer_pt.parallel._comms import Reply
from mchammer_pt.parallel._worker import (
    BaseWorker,
    CanonicalWorker,
    WangLandauWorker,
    _Shutdown,
)
from mchammer_pt.replica import Replica
from mchammer_pt.wl_replica import WangLandauReplica


class InProcessWorkerConn:
    """Duplex ``Connection``-alike that dispatches sends inline to a real worker.

    Construct with a real ``WangLandauReplica`` (default) for REWL
    tests, or via :meth:`for_canonical` with a ``Replica`` for
    canonical-ensemble tests. The worker's full ``_handle`` dispatch
    runs against it in-process. Replies are queued and returned by
    ``recv()``.

    Pickle is bypassed -- objects pass by reference.
    """

    def __init__(self, replica: WangLandauReplica) -> None:
        self._replies: deque[Reply] = deque()
        self._worker: BaseWorker = WangLandauWorker.for_replica(
            replica, reply_sink=self._replies.append
        )

    @classmethod
    def for_canonical(cls, replica: Replica) -> InProcessWorkerConn:
        """Build a conn driving a ``CanonicalWorker`` against ``replica``."""
        conn = cls.__new__(cls)
        conn._replies = deque()
        conn._worker = CanonicalWorker.for_replica(
            replica, reply_sink=conn._replies.append
        )
        return conn

    def send(self, msg: tuple[Any, ...]) -> None:
        """Dispatch the command inline through the worker handler.

        ``_Shutdown`` propagates out of the production read loop to
        break it; here it is swallowed so the in-process conn looks
        like a normal pipe to callers that drive ``SHUTDOWN`` through
        the pool's shutdown path.
        """
        try:
            self._worker._handle(msg)
        except _Shutdown:
            pass

    def recv(self) -> Reply:
        """Pop the next queued reply."""
        return self._replies.popleft()

    def poll(self, timeout: float = 0.0) -> bool:
        """Whether a reply is queued; ``timeout`` is ignored."""
        return bool(self._replies)

    def close(self) -> None:
        """No-op; nothing to release."""
