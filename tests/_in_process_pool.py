"""In-process drop-ins for :class:`ProcessPool` and
:class:`ProcessWangLandauPool` used by single-round-trip tests.

Each helper constructs the same parent-side pool object but replaces
the spawned worker processes with :class:`InProcessWorkerConn`
instances backed by real :class:`Replica` /
:class:`WangLandauReplica` instances. The pool API surface is
exercised unchanged; only the IPC layer is inlined.

Use for tests that exercise a single round-trip through the pool's
plumbing and do not depend on real process isolation or the pickle
boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from mchammer.ensembles import CanonicalEnsemble

from mchammer_pt.parallel.processes import (
    ProcessPool,
    ProcessWangLandauPool,
    ProcessWangLandauWindow,
)
from mchammer_pt.replica import Replica
from mchammer_pt.wl_ensemble import CoordinatedWangLandauEnsemble
from mchammer_pt.wl_replica import WangLandauReplica
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
        pool._workers.append((_DummyProcess(), conn))  # type: ignore[arg-type]
    pool._ensemble_cls_fqn = (
        f"{ensemble_cls.__module__}.{ensemble_cls.__qualname__}"
    )
    pool._ensemble_kwargs_hash = ""
    return pool


def make_in_process_wl_pool(
    tmp_path: Path,
    *,
    windows: list[tuple[float | None, float | None]],
    seeds: list[int] | None = None,
    n_walkers_per_window: int | list[int] = 1,
    ensemble_kwargs: dict[str, Any] | None = None,
    flatness_mode: str = "pooled",
    merge_cadence: str = "at_halve",
) -> ProcessWangLandauPool:
    """Build a :class:`ProcessWangLandauPool` whose workers are in-process conns.

    Mirrors the production constructor's argument layout closely
    enough that tests can swap one call for the other. The cluster
    expansion and atoms come from the WL test fixtures. The CE file
    is written into ``tmp_path`` so worker-side factories can reload
    it via ``replica.cluster_expansion_path``.
    """
    from tests._wl_fixtures import make_wl_atoms, make_wl_ce

    ce, atoms = make_wl_ce(), make_wl_atoms()
    ce_path = tmp_path / "ce.ce"
    ce.write(str(ce_path))

    n_windows = len(windows)
    if seeds is None:
        seeds = list(range(n_windows))
    if isinstance(n_walkers_per_window, int):
        walkers_per_window = [n_walkers_per_window] * n_windows
    else:
        walkers_per_window = list(n_walkers_per_window)
    extra_kwargs = dict(ensemble_kwargs) if ensemble_kwargs else {}

    pool = ProcessWangLandauPool.__new__(ProcessWangLandauPool)
    pool._flatness_mode = flatness_mode  # type: ignore[assignment]
    pool._merge_cadence = merge_cadence  # type: ignore[assignment]
    pool._windows = list(windows)
    pool._energy_spacing = 0.1
    pool._slots = []

    for (lo, hi), window_seed, W_w in zip(
        windows, seeds, walkers_per_window, strict=True
    ):
        if W_w == 1:
            walker_seeds = [int(window_seed)]
            rng_seed = int(window_seed)
        else:
            sub = np.random.SeedSequence(int(window_seed))
            children = sub.spawn(W_w + 1)
            walker_seeds = [
                int(c.generate_state(1)[0]) for c in children[:W_w]
            ]
            rng_seed = int(children[W_w].generate_state(1)[0])
        workers: list[Any] = []
        for w_seed in walker_seeds:
            replica = WangLandauReplica(
                cluster_expansion=ce,
                atoms=atoms,
                energy_spacing=0.1,
                energy_limit_left=lo,
                energy_limit_right=hi,
                random_seed=int(w_seed),
                ensemble_cls=CoordinatedWangLandauEnsemble,
                ensemble_kwargs=extra_kwargs,
                cluster_expansion_path=str(ce_path),
            )
            conn = InProcessWorkerConn(replica)
            workers.append((_DummyProcess(), conn))
        pool._slots.append(ProcessWangLandauWindow(
            workers=workers,
            rng=np.random.default_rng(rng_seed),
            flatness_mode=flatness_mode,  # type: ignore[arg-type]
            merge_cadence=merge_cadence,  # type: ignore[arg-type]
            schedule=str(extra_kwargs.get("schedule", "halving")),
        ))
    pool._ensemble_cls_fqn = (
        f"{CoordinatedWangLandauEnsemble.__module__}."
        f"{CoordinatedWangLandauEnsemble.__qualname__}"
    )
    pool._ensemble_kwargs_hash = ""
    return pool
