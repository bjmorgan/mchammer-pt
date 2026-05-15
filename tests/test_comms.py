"""Tests for the _comms pipe communication primitives."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from mchammer_pt.parallel._comms import Reply, recv_reply, request


class TestRecvReply:
    """Tests for recv_reply: receive and validate a single Reply."""

    def test_happy_path_returns_payload(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("OK", "ENERGY", 42.0))
        result = recv_reply(parent, "ENERGY", label=0)
        assert result == 42.0

    def test_opcode_mismatch_raises(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("OK", "ADVANCE", None))
        with pytest.raises(RuntimeError, match="sent 'ENERGY'.*reply to 'ADVANCE'"):
            recv_reply(parent, "ENERGY", label=0)

    def test_err_status_raises_runtime_error(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("ERR", "ENERGY", "traceback text"))
        with pytest.raises(RuntimeError, match="traceback text"):
            recv_reply(parent, "ENERGY", label=0)

    def test_err_pickle_status_raises_type_error(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("ERR_PICKLE", "GET_OBSERVERS", "pickle error"))
        with pytest.raises(TypeError, match="pickle"):
            recv_reply(parent, "GET_OBSERVERS", label=0)

    def test_pipe_death_raises_runtime_error(self):
        parent, child = mp.Pipe(duplex=True)
        child.close()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            recv_reply(parent, "ENERGY", label=0)

    def test_label_appears_in_error_message(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("ERR", "ENERGY", "boom"))
        with pytest.raises(RuntimeError, match="window 2 walker 1"):
            recv_reply(parent, "ENERGY", label="window 2 walker 1")


class TestRequest:
    """Tests for request: atomic send + recv_reply."""

    def test_sends_message_and_returns_payload(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("OK", "ENERGY", 42.0))
        result = request(parent, ("ENERGY",), label=0)
        assert result == 42.0
        assert child.recv() == ("ENERGY",)

    def test_sends_message_with_args(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("OK", "ADVANCE", None))
        request(parent, ("ADVANCE", 100), label=0)
        cmd = child.recv()
        assert cmd == ("ADVANCE", 100)

    def test_error_propagates_from_recv_reply(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("ERR", "ENERGY", "boom"))
        with pytest.raises(RuntimeError):
            request(parent, ("ENERGY",), label=0)
