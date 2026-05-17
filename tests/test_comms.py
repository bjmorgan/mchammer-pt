"""Tests for the _comms pipe communication primitives."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from mchammer_pt.parallel._comms import (
    Reply,
    broadcast_gather,
    fanout_gather,
    recv_reply,
    request,
)


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


class TestBroadcastGather:
    """Tests for broadcast_gather: send same message to all, gather replies."""

    def test_happy_path_returns_payloads_in_order(self):
        targets = []
        children = []
        for i in range(3):
            parent, child = mp.Pipe(duplex=True)
            child.send(Reply("OK", "ENERGY", float(i)))
            targets.append((parent, i))
            children.append(child)
        payloads = broadcast_gather(targets, ("ENERGY",))
        assert payloads == [0.0, 1.0, 2.0]
        for child in children:
            assert child.recv() == ("ENERGY",)

    def test_single_target(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("OK", "ADVANCE", None))
        payloads = broadcast_gather([(parent, 0)], ("ADVANCE", 50))
        assert payloads == [None]

    def test_error_on_second_connection_raises(self):
        p0, c0 = mp.Pipe(duplex=True)
        p1, c1 = mp.Pipe(duplex=True)
        c0.send(Reply("OK", "ENERGY", 1.0))
        c1.send(Reply("ERR", "ENERGY", "boom"))
        with pytest.raises(RuntimeError, match="boom"):
            broadcast_gather([(p0, 0), (p1, 1)], ("ENERGY",))

    def test_pipe_death_mid_gather_raises(self):
        p0, c0 = mp.Pipe(duplex=True)
        p1, c1 = mp.Pipe(duplex=True)
        c0.send(Reply("OK", "ENERGY", 1.0))
        c1.close()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            broadcast_gather([(p0, 0), (p1, 1)], ("ENERGY",))

    def test_send_phase_broken_pipe_raises(self):
        p0, c0 = mp.Pipe(duplex=True)
        c0.close()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            broadcast_gather([(p0, 0)], ("ENERGY",))

    def test_empty_targets_returns_empty_list(self):
        assert broadcast_gather([], ("ENERGY",)) == []


class TestFanoutGather:
    """Tests for fanout_gather: per-target message, gather replies in order."""

    def test_per_target_distinct_payloads(self):
        """Each target receives its own message; replies returned in order."""
        targets = []
        children = []
        msgs = [("SET_ENTROPY", {0: 1.0}), ("SET_ENTROPY", {0: 2.0}),
                ("SET_ENTROPY", {0: 3.0})]
        for i, msg in enumerate(msgs):
            parent, child = mp.Pipe(duplex=True)
            child.send(Reply("OK", "SET_ENTROPY", None))
            targets.append((parent, f"t{i}", msg))
            children.append((child, msg))
        payloads = fanout_gather(targets)
        assert payloads == [None, None, None]
        # Each child received the message paired with its target.
        for child, expected_msg in children:
            assert child.recv() == expected_msg

    def test_single_target(self):
        parent, child = mp.Pipe(duplex=True)
        child.send(Reply("OK", "ADVANCE", 7.0))
        payloads = fanout_gather([(parent, "x", ("ADVANCE", 50))])
        assert payloads == [7.0]

    def test_error_on_second_connection_raises(self):
        p0, c0 = mp.Pipe(duplex=True)
        p1, c1 = mp.Pipe(duplex=True)
        c0.send(Reply("OK", "SET_ENTROPY", None))
        c1.send(Reply("ERR", "SET_ENTROPY", "boom"))
        with pytest.raises(RuntimeError, match="boom"):
            fanout_gather([
                (p0, 0, ("SET_ENTROPY", {0: 1.0})),
                (p1, 1, ("SET_ENTROPY", {0: 2.0})),
            ])

    def test_pipe_death_mid_gather_raises(self):
        p0, c0 = mp.Pipe(duplex=True)
        p1, c1 = mp.Pipe(duplex=True)
        c0.send(Reply("OK", "SET_ENTROPY", None))
        c1.close()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            fanout_gather([
                (p0, 0, ("SET_ENTROPY", {0: 1.0})),
                (p1, 1, ("SET_ENTROPY", {0: 2.0})),
            ])

    def test_send_phase_broken_pipe_raises(self):
        p0, c0 = mp.Pipe(duplex=True)
        c0.close()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            fanout_gather([(p0, 0, ("SET_ENTROPY", {0: 1.0}))])

    def test_opcode_validated_per_target(self):
        """Reply validation uses each target's own msg[0] tag."""
        p0, c0 = mp.Pipe(duplex=True)
        # Target sent SET_ENTROPY but worker mis-tags reply as ADVANCE.
        c0.send(Reply("OK", "ADVANCE", None))
        with pytest.raises(
            RuntimeError, match="sent 'SET_ENTROPY'.*reply to 'ADVANCE'"
        ):
            fanout_gather([(p0, 0, ("SET_ENTROPY", {0: 1.0}))])

    def test_empty_targets_returns_empty_list(self):
        assert fanout_gather([]) == []
