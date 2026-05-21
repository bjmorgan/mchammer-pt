"""mchammer-pt: parallel tempering for mchammer canonical Monte Carlo."""

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
from .checkpoint import CheckpointWriter
from .diagnostics import (
    energy_autocorrelation_time,
    round_trip_counts,
    swap_acceptance_rates,
)
from .analysis.dos import reweight_canonical_from_dos, stitch_entropy
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

__version__ = "0.11.0"

__all__ = [
    "BaseParallelTempering",
    "CanonicalParallelTempering",
    "CanonicalPool",
    "CheckpointWriter",
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
    "stitch_entropy",
    "swap_acceptance_rates",
    "write_hdf5",
]
