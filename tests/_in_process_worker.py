"""In-process ``Connection``-alike for testing WL workers without
spawning a subprocess. Drop-in for ``mp.Pipe`` in tests that do not
depend on the pickle boundary or real-process isolation.

The helper builds a real ``WangLandauWorker`` and dispatches each
``send()`` inline through ``worker._handle``, capturing replies into
an internal queue. The worker's full handler table runs against a
real ``WangLandauReplica`` -- no hand-written fake.

Tier-2 (real-subprocess) opcode tests remain in place to pin the IPC
protocol so the in-process tier cannot silently drift.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from mchammer_pt.parallel._comms import Reply
from mchammer_pt.parallel._worker import WangLandauWorker
from mchammer_pt.wl_replica import WangLandauReplica


class InProcessWorkerConn:
    """Duplex ``Connection``-alike that dispatches sends inline to a real worker.

    Construct with a real ``WangLandauReplica``; the worker's full
    ``_handle`` dispatch runs against it in-process. Replies are
    queued and returned by ``recv()``.

    Pickle is bypassed -- objects pass by reference.
    """

    def __init__(self, replica: WangLandauReplica) -> None:
        self._replies: deque[Reply] = deque()
        self._worker = WangLandauWorker.for_replica(
            replica, reply_sink=self._replies.append
        )

    def send(self, msg: tuple[Any, ...]) -> None:
        """Dispatch the command inline through the worker handler."""
        self._worker._handle(msg)

    def recv(self) -> Reply:
        """Pop the next queued reply."""
        return self._replies.popleft()

    def poll(self, timeout: float = 0.0) -> bool:
        """Whether a reply is queued; ``timeout`` is ignored."""
        return bool(self._replies)

    def close(self) -> None:
        """No-op; nothing to release."""
