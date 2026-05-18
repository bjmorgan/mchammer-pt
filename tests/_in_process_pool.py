"""In-process drop-in for :class:`ProcessPool` used by single-round-trip
tests.

Constructs the same parent-side ``ProcessPool`` object but replaces the
spawned worker processes with :class:`InProcessWorkerConn` instances
backed by real :class:`Replica` instances. The pool's public API
(``attach_observer``, ``advance_all``, ``data_containers``,
``get_observers``, ...) is exercised unchanged; only the IPC layer is
inlined.

Use for tests that exercise a single configuration round-trip through
the pool's plumbing and do not depend on real process isolation or the
pickle boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from mchammer.ensembles import CanonicalEnsemble

from mchammer_pt.parallel.processes import ProcessPool
from mchammer_pt.replica import Replica
from tests._in_process_worker import InProcessWorkerConn


class _DummyProcess:
    """Stand-in for ``mp.Process`` in :class:`ProcessPool._workers` tuples.

    The pool only touches the process object during ``shutdown`` (for
    ``join`` and ``terminate``). Both are no-ops here -- in-process
    workers have no lifecycle.
    """

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        pass


def make_in_process_pool(
    toy_ce: Any,
    toy_atoms: Atoms,
    tmp_path: Path,
    *,
    temperatures: list[float] | None = None,
    seeds: list[int] | None = None,
) -> ProcessPool:
    """Build a :class:`ProcessPool` whose workers are in-process conns.

    Mirrors the test-suite ``_make_process`` helper: three replicas at
    300/400/500 K with seeds 0/1/2 by default. The cluster-expansion
    file is written to ``tmp_path`` so worker-side factories can reload
    it via ``replica.cluster_expansion_path``.
    """
    if temperatures is None:
        temperatures = [300.0, 400.0, 500.0]
    if seeds is None:
        seeds = [0, 1, 2]
    ce_path = tmp_path / "toy.ce"
    toy_ce.write(str(ce_path))

    pool = ProcessPool.__new__(ProcessPool)
    pool._temperatures = [float(T) for T in temperatures]
    pool._workers = []
    ensemble_cls: type[CanonicalEnsemble] = CanonicalEnsemble
    for T, s in zip(temperatures, seeds, strict=True):
        replica = Replica(
            cluster_expansion=toy_ce,
            atoms=toy_atoms,
            temperature=T,
            random_seed=int(s),
            cluster_expansion_path=str(ce_path),
        )
        conn = InProcessWorkerConn.for_canonical(replica)
        # _workers expects (Process, Connection); the in-process
        # substitutes are duck-typed for the surface the pool uses
        # (send/recv/poll/close on conn; join/is_alive/terminate on
        # the process). Mypy can't see this without unsafe casts.
        pool._workers.append((_DummyProcess(), conn))  # type: ignore[arg-type]
    pool._ensemble_cls_fqn = (
        f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
    )
    pool._ensemble_kwargs_hash = ""
    return pool
