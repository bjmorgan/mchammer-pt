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
from .parallel.backend import Backend
from .parallel.processes import ProcessBackend
from .parallel.serial import SerialBackend
from .replica import Replica

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BaseParallelTempering",
    "CanonicalParallelTempering",
    "ExchangeCallback",
    "ExchangeHistory",
    "ExchangePrinter",
    "ProcessBackend",
    "Replica",
    "SerialBackend",
    "SwapRateTracker",
    "__version__",
    "energy_autocorrelation_time",
    "read_hdf5",
    "round_trip_counts",
    "swap_acceptance_rates",
    "write_hdf5",
]
