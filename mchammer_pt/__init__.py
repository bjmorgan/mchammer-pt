"""mchammer-pt: parallel tempering for mchammer canonical Monte Carlo."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .analysis.dos import reweight_canonical_from_dos, stitch_entropy
from .base import BaseParallelTempering
from .callbacks import (
    CycleCallback,
    ExchangeCallback,
    ExchangePrinter,
    ProgressPrinter,
    SwapRateTracker,
    WangLandauProgressPrinter,
)
from .canonical import CanonicalParallelTempering
from .checkpoint import CheckpointWriter, completed_cycles
from .diagnostics import (
    energy_autocorrelation_time,
    round_trip_counts,
    swap_acceptance_rates,
)
from .history import ExchangeHistory, read_hdf5, write_hdf5
from .parallel.backend import (
    CanonicalPool,
    ObservablePool,
    ReplicaPool,
    WangLandauObservablePool,
    WangLandauPool,
)
from .parallel.processes import ProcessPool, ProcessWangLandauPool
from .parallel.serial import SerialPool, SerialWangLandauPool
from .replica import Replica
from .wl import WangLandauParallelTempering
from .wl_merge_diagnostics import MergeEvent
from .wl_replica import WangLandauReplica

try:
    __version__ = version("mchammer-pt")
except PackageNotFoundError:
    # Not installed (e.g. imported from a raw source checkout).
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseParallelTempering",
    "CanonicalParallelTempering",
    "CanonicalPool",
    "CheckpointWriter",
    "completed_cycles",
    "CycleCallback",
    "ExchangeCallback",
    "ExchangeHistory",
    "ExchangePrinter",
    "MergeEvent",
    "ObservablePool",
    "ProcessPool",
    "ProcessWangLandauPool",
    "ProgressPrinter",
    "Replica",
    "ReplicaPool",
    "SeedSearchParams",
    "SerialPool",
    "SerialWangLandauPool",
    "SwapRateTracker",
    "WangLandauObservablePool",
    "WangLandauParallelTempering",
    "WangLandauPool",
    "WangLandauProgressPrinter",
    "WangLandauReplica",
    "__version__",
    "energy_autocorrelation_time",
    "read_hdf5",
    "reweight_canonical_from_dos",
    "round_trip_counts",
    "seed_window_configs",
    "stitch_entropy",
    "swap_acceptance_rates",
    "write_hdf5",
]

if TYPE_CHECKING:
    from .seeding import SeedSearchParams, seed_window_configs

# ``seed_window_configs`` / ``SeedSearchParams`` live in
# ``mchammer_pt.seeding``, which imports ``mchammer_moves`` -- an optional
# dependency (see ``mchammer_pt.contrib``). Import them lazily so that
# ``import mchammer_pt`` still succeeds when mchammer-moves is not
# installed; the import error surfaces only when these names are used.
_SEEDING_EXPORTS = frozenset({"SeedSearchParams", "seed_window_configs"})


def __getattr__(name: str) -> object:
    if name in _SEEDING_EXPORTS:
        import importlib

        return getattr(importlib.import_module("mchammer_pt.seeding"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
