"""Smoke tests for InProcessWorkerConn."""

from __future__ import annotations

from mchammer.calculators import ClusterExpansionCalculator

from mchammer_pt.wl_replica import WangLandauReplica
from tests._in_process_worker import InProcessWorkerConn
from tests._wl_fixtures import make_wl_atoms, make_wl_ce


def _make_wl_replica_with_entropy(entropy: dict[int, float]) -> WangLandauReplica:
    """Build a WL replica and overwrite its entropy dict for testing."""
    ce, atoms = make_wl_ce(), make_wl_atoms()
    e0 = float(
        ClusterExpansionCalculator(atoms, ce).calculate_total(
            occupations=atoms.numbers
        )
    )
    replica = WangLandauReplica(
        cluster_expansion=ce,
        atoms=atoms,
        energy_spacing=0.1,
        energy_limit_left=e0 - 100.0,
        energy_limit_right=e0 + 100.0,
        random_seed=0,
    )
    replica.ensemble._entropy = dict(entropy)
    return replica


def test_in_process_conn_dispatches_get_entropy():
    """In-process conn returns a real worker's GET_ENTROPY reply."""
    replica = _make_wl_replica_with_entropy({0: 1.0, 1: 2.0})
    conn = InProcessWorkerConn(replica)

    conn.send(("GET_ENTROPY",))
    reply = conn.recv()

    assert reply.status == "OK"
    assert reply.op == "GET_ENTROPY"
    assert reply.payload == {0: 1.0, 1: 2.0}


def test_in_process_conn_set_then_get_entropy():
    """SET_ENTROPY -> GET_ENTROPY round-trip via the in-process conn."""
    replica = _make_wl_replica_with_entropy({0: 0.0})
    conn = InProcessWorkerConn(replica)

    new_entropy = {0: 5.0, 1: 7.0}
    conn.send(("SET_ENTROPY", new_entropy))
    set_reply = conn.recv()
    assert set_reply.status == "OK"

    conn.send(("GET_ENTROPY",))
    get_reply = conn.recv()
    assert get_reply.payload == {0: 5.0, 1: 7.0}


def test_in_process_conn_poll_reflects_queue_state():
    """poll() is False before send, True after, False after recv."""
    replica = _make_wl_replica_with_entropy({})
    conn = InProcessWorkerConn(replica)

    assert conn.poll() is False
    conn.send(("GET_ENTROPY",))
    assert conn.poll() is True
    conn.recv()
    assert conn.poll() is False
