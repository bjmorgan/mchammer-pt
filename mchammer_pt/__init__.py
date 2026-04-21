"""mchammer-pt: parallel tempering for mchammer canonical Monte Carlo."""

from .base import BaseParallelTempering
from .callbacks import ExchangeCallback, ExchangePrinter, SwapRateTracker
from .canonical import CanonicalParallelTempering
from .diagnostics import (
    energy_autocorrelation_time,
    round_trip_counts,
    swap_acceptance_rates,
)
from .history import ExchangeHistory, read_hdf5, write_hdf5
from .parallel.backend import ObservablePool, ReplicaPool
from .parallel.processes import ProcessPool
from .parallel.serial import SerialPool
from .replica import Replica

# Temporary aliases (removed in commit 4 of this refactor):
Backend = ReplicaPool
SerialBackend = SerialPool
ProcessBackend = ProcessPool

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BaseParallelTempering",
    "CanonicalParallelTempering",
    "ExchangeCallback",
    "ExchangeHistory",
    "ExchangePrinter",
    "ObservablePool",
    "ProcessBackend",
    "ProcessPool",
    "Replica",
    "ReplicaPool",
    "SerialBackend",
    "SerialPool",
    "SwapRateTracker",
    "__version__",
    "energy_autocorrelation_time",
    "read_hdf5",
    "round_trip_counts",
    "swap_acceptance_rates",
    "write_hdf5",
]
