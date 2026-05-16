"""Tests for the BaseWorker command loop and dispatch."""

from __future__ import annotations

import multiprocessing as mp
import threading

import numpy as np
import pytest

from mchammer_pt.parallel._comms import Reply
from mchammer_pt.parallel._worker import BaseWorker


class StubReplica:
    """Minimal replica stand-in for testing BaseWorker dispatch."""

    def __init__(self):
        self.advanced_steps = 0
        self._energy = 1.5
        self._occupations = np.array([29, 79, 29, 79], dtype=np.int64)

    def advance(self, n_steps: int) -> None:
        self.advanced_steps += n_steps

    def current_energy(self) -> float:
        return self._energy

    def current_occupations(self) -> np.ndarray:
        return self._occupations.copy()

    def set_occupations(self, occ: np.ndarray) -> None:
        self._occupations = occ


class StubWorker(BaseWorker):
    """Concrete BaseWorker subclass for testing."""

    def __init__(self, conn, *, fail_build: bool = False):
        super().__init__(conn)
        self._fail_build = fail_build

    def _build_replica(self):
        if self._fail_build:
            raise ValueError("construction failed")
        return StubReplica()


def _run_worker(worker_conn, **kwargs) -> threading.Thread:
    """Start a StubWorker in a background thread and return the thread."""
    worker = StubWorker(worker_conn, **kwargs)
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    return t


class TestBaseWorkerStartup:

    def test_startup_handshake_ok(self):
        parent, child = mp.Pipe(duplex=True)
        t = _run_worker(child)
        reply = Reply(*parent.recv())
        assert reply == Reply("OK", "STARTUP", None)
        parent.send(("SHUTDOWN",))
        parent.recv()
        t.join(timeout=5)

    def test_startup_handshake_err_on_build_failure(self):
        parent, child = mp.Pipe(duplex=True)
        t = _run_worker(child, fail_build=True)
        reply = Reply(*parent.recv())
        assert reply.status == "ERR"
        assert reply.op == "STARTUP"
        assert "construction failed" in reply.payload
        t.join(timeout=5)


class TestBaseWorkerDispatch:

    @pytest.fixture(autouse=True)
    def _start_worker(self):
        self.parent, child = mp.Pipe(duplex=True)
        self.thread = _run_worker(child)
        reply = Reply(*self.parent.recv())
        assert reply == Reply("OK", "STARTUP", None)
        yield
        try:
            self.parent.send(("SHUTDOWN",))
            self.parent.recv()
        except (EOFError, BrokenPipeError):
            pass
        self.thread.join(timeout=5)

    def test_energy_returns_value(self):
        self.parent.send(("ENERGY",))
        reply = Reply(*self.parent.recv())
        assert reply == Reply("OK", "ENERGY", 1.5)

    def test_advance_increments_steps(self):
        self.parent.send(("ADVANCE", 10))
        reply = Reply(*self.parent.recv())
        assert reply == Reply("OK", "ADVANCE", None)

    def test_get_occ_returns_array(self):
        self.parent.send(("GET_OCC",))
        reply = Reply(*self.parent.recv())
        assert reply.status == "OK"
        assert reply.op == "GET_OCC"
        assert isinstance(reply.payload, np.ndarray)

    def test_set_occ_updates_state(self):
        new_occ = np.array([79, 29, 79, 29], dtype=np.int64)
        self.parent.send(("SET_OCC", new_occ))
        reply = Reply(*self.parent.recv())
        assert reply == Reply("OK", "SET_OCC", None)
        self.parent.send(("GET_OCC",))
        reply = Reply(*self.parent.recv())
        np.testing.assert_array_equal(reply.payload, new_occ)

    def test_unknown_opcode_returns_err(self):
        self.parent.send(("NONSENSE",))
        reply = Reply(*self.parent.recv())
        assert reply.status == "ERR"
        assert reply.op == "NONSENSE"
        assert "unknown command" in reply.payload

    def test_shutdown_sends_reply_and_exits(self):
        self.parent.send(("SHUTDOWN",))
        reply = Reply(*self.parent.recv())
        assert reply == Reply("OK", "SHUTDOWN", None)
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
