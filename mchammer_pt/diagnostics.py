"""PT diagnostics: round-trip counting, autocorrelation, swap rates.

Pure functions operating on `ExchangeHistory` (or its constituent
arrays). They do not touch the orchestrator, are trivially unit-
testable, and form the basis for any user-built quality checks.
"""

from __future__ import annotations

import warnings

import numpy as np

from .history import ExchangeHistory


def round_trip_counts(
    replica_labels_per_cycle: np.ndarray,
    window_of_position: np.ndarray | None = None,
    n_windows: int | None = None,
) -> np.ndarray:
    """Per-carrier count of completed ladder round trips.

    A round trip is one complete bottom -> top -> bottom traversal (or
    top -> bottom -> top): two endpoint visits on opposite rungs with
    no intervening same-rung visit. Partial traversals do not count.

    The label array is position-indexed: ``labels[cycle][position]`` is
    the carrier id occupying that ``(window, slot)`` position. Each
    position maps to a window rung via ``window_of_position``; a carrier
    completes a round trip when its window reaches the bottom (``0``)
    and the top (``n_windows - 1``) alternately.

    Args:
        replica_labels_per_cycle: shape ``(n_cycles+1, N_w)``, the
            carrier id at each position per cycle.
        window_of_position: shape ``(N_w,)``, the window index for each
            position. Defaults to ``arange(N_w)`` (one position per
            window: the single-walker case).
        n_windows: number of window rungs. Defaults to the number of
            positions (single-walker case).

    Returns:
        1-D array of shape ``(N_w,)`` giving the round-trip count for
        each carrier id.
    """
    labels = np.asarray(replica_labels_per_cycle)
    n_cycles_plus_one, n_positions = labels.shape
    if window_of_position is None:
        window_of_position = np.arange(n_positions, dtype=np.int64)
    else:
        window_of_position = np.asarray(window_of_position)
    if n_windows is None:
        n_windows = n_positions
    top = n_windows - 1
    counts = np.zeros(n_positions, dtype=np.int64)
    # Per carrier: 0 = no endpoint yet, 1 = last endpoint was bottom,
    # 2 = last endpoint was top. A round trip is each 1->2 or 2->1.
    state = np.zeros(n_positions, dtype=np.int8)
    for cycle in range(n_cycles_plus_one):
        row = labels[cycle]
        for position in range(n_positions):
            window = int(window_of_position[position])
            if window == 0:
                new_state = 1
            elif window == top:
                new_state = 2
            else:
                continue
            carrier = int(row[position])
            if state[carrier] != 0 and state[carrier] != new_state:
                counts[carrier] += 1
            state[carrier] = new_state
    # Two endpoint transitions per full round trip; halve.
    return counts // 2


def swap_acceptance_rates(history: ExchangeHistory) -> np.ndarray:
    """Per-pair acceptance fractions, NaN where no attempts were made."""
    attempts = history.swap_attempted.astype(np.float64)
    accepts = history.swap_accepted.astype(np.float64)
    return np.where(attempts > 0, accepts / np.maximum(attempts, 1), np.nan)


def energy_autocorrelation_time(energies: np.ndarray) -> float:
    """Integrated autocorrelation time of a 1-D energy trace.

    Uses the standard windowed estimator with Sokal's automatic window
    selection (``M = min_M { M >= c * tau(M) }``, c = 5). Returns a
    float; for IID data the estimate is ~1.

    If the Sokal window never closes within the trace length (trace is
    too short relative to the autocorrelation time), returns ``nan``
    and emits a ``UserWarning`` with the trace length. A hard failure
    for nonsense inputs (length < 4, zero-variance trace) also returns
    ``nan`` — without a warning, since those cases are obvious from
    the input.
    """
    x = np.asarray(energies, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return float("nan")
    x = x - x.mean()
    # Autocovariance via FFT for speed.
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    if acf[0] == 0.0:
        return float("nan")
    acf = acf / acf[0]

    # Prefix sum of acf[1:] so tau(m) = 1 + 2 * cumsum[m] is O(1) per
    # m and the worst-case loop is O(n) rather than O(n^2).
    cumulative = np.cumsum(acf[1:])
    c = 5.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * float(cumulative[m - 1])
        if m >= c * tau:
            return tau
    warnings.warn(
        f"Sokal window did not close in {n} samples; autocorrelation "
        "time is longer than the trace. Returning nan.",
        UserWarning,
        stacklevel=2,
    )
    return float("nan")
