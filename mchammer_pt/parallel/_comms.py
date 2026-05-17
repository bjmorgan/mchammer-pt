"""Pipe communication primitives for worker pools.

Provides structurally paired send/receive functions that prevent
the class of bug where a send is not matched by a receive (or
vice versa), and opcode round-trip validation that catches
worker-side dispatch errors.

The wire protocol: workers reply with ``Reply(status, op, payload)``
named tuples. ``op`` echoes the opcode from the command that
triggered the reply. ``recv_reply`` validates this echo.
"""

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any, Literal, NamedTuple


class Reply(NamedTuple):
    """Worker-to-parent reply on a pipe connection."""

    status: Literal["OK", "ERR", "ERR_PICKLE"]
    op: str
    payload: Any


def recv_reply(conn: Connection, expected_op: str, label: Any) -> Any:
    """Receive a Reply and validate its opcode and status.

    Args:
        conn: parent-side pipe connection.
        expected_op: the opcode that was sent; the reply's ``op``
            field must match.
        label: identifier for error messages (e.g. replica index
            or ``"window 2 walker 1"``).

    Returns:
        The reply payload on success.

    Raises:
        RuntimeError: opcode mismatch, worker ERR, or pipe death.
        TypeError: worker ERR_PICKLE.
    """
    try:
        reply = Reply(*conn.recv())
    except EOFError as exc:
        raise RuntimeError(
            f"worker {expected_op} ({label}) exited unexpectedly"
        ) from exc
    if reply.op != expected_op:
        raise RuntimeError(
            f"protocol desync for {label}: sent {expected_op!r} but "
            f"received reply to {reply.op!r}"
        )
    if reply.status == "ERR_PICKLE":
        raise TypeError(
            f"reply from worker {expected_op} ({label}) could not be "
            f"round-tripped through pickle: {reply.payload}"
        )
    if reply.status != "OK":
        raise RuntimeError(
            f"worker {expected_op} ({label}) failed: {reply.payload}"
        )
    return reply.payload


def request(conn: Connection, msg: tuple[Any, ...], label: Any) -> Any:
    """Send a command and receive the validated reply.

    Atomic send/recv pair: structurally impossible to send without
    consuming the reply.

    Args:
        conn: parent-side pipe connection.
        msg: command tuple, e.g. ``("ENERGY",)`` or
            ``("ADVANCE", 100)``. ``msg[0]`` is the opcode.
        label: identifier for error messages.

    Returns:
        The reply payload on success.
    """
    try:
        conn.send(msg)
    except BrokenPipeError:
        raise RuntimeError(f"worker {label!r} pipe broke during send") from None
    return recv_reply(conn, msg[0], label)


def broadcast_gather(
    targets: list[tuple[Connection, Any]],
    msg: tuple[Any, ...],
) -> list[Any]:
    """Send the same command to all targets, then gather validated replies.

    Phase 1: sends ``msg`` to every connection.
    Phase 2: receives from every connection in order, validating
    each reply via ``recv_reply``.

    Args:
        targets: list of ``(connection, label)`` pairs.
        msg: command tuple sent to all targets.

    Returns:
        List of payloads in the same order as ``targets``.
    """
    for conn, label in targets:
        try:
            conn.send(msg)
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"worker {msg[0]} ({label}) exited unexpectedly"
            ) from exc
    return [recv_reply(conn, msg[0], label) for conn, label in targets]


def fanout_gather(
    targets: list[tuple[Connection, str, tuple[Any, ...]]],
) -> list[Any]:
    """Send a per-target command, then gather validated replies.

    Differs from ``broadcast_gather`` only in that each target gets
    its own message — useful when callers want all workers to act in
    parallel but with different payloads (e.g. per-window
    ``SET_ENTROPY`` with different merged entropies). The send phase
    is fully issued before any recv, so all workers run concurrently.

    Args:
        targets: list of ``(connection, label, msg)`` triples. Each
            message is sent to its paired connection; the op tag
            ``msg[0]`` drives reply validation.

    Returns:
        List of payloads in the same order as ``targets``.
    """
    for conn, label, msg in targets:
        try:
            conn.send(msg)
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"worker {msg[0]} ({label}) exited unexpectedly"
            ) from exc
    return [
        recv_reply(conn, msg[0], label) for conn, label, msg in targets
    ]
