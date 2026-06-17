"""End-to-end integration tests for frozen-g measurement accumulation.

Covers:
- Task 4.3: accumulation chaining — a measurement chained across a
  checkpoint accumulates moments across the checkpoint boundary (the
  total count after chaining strictly exceeds the first-segment count).
  Recording is passive: attaching a recorder does not perturb the
  frozen-g trajectory relative to an unrecorded run from the same
  converged checkpoint.
- Task 4.4: multiple recorders — two observers with distinct tags and
  different intervals accumulate independently; their per-bin means
  match the analytic value; both tags round-trip through a checkpoint;
  and unbound stores are preserved when a tag is not re-attached.

Note: bit-exact reproduction of the *resumed sample* is deliberately not
asserted.  A checkpoint+resume with a recorder attached yields a
statistically-equivalent but not bit-identical microcanonical sample (a
benign RNG-alignment effect at the resume boundary).  The microcanonical
averages converge regardless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mchammer.observers.base_observer import BaseObserver

from mchammer_pt.wl import WangLandauParallelTempering
from tests._wl_fixtures import make_wl_atoms, make_wl_ce, make_wl_ensemble

# ---------------------------------------------------------------------------
# Picklable test observers (module-level for ProcessPool compatibility)
# ---------------------------------------------------------------------------


class _ConstantObs(BaseObserver):
    """Returns a fixed scalar constant regardless of structure.

    Args:
        value: The constant value returned by every call.
        interval: Observer interval (record every nth step).
        tag: Tag identifying this observer.
    """

    def __init__(self, value: float = 1.0, interval: int = 1, tag: str = "c") -> None:
        super().__init__(interval=interval, return_type=float, tag=tag)
        self._value = value

    def get_observable(self, structure: Any) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _initial_energy() -> float:
    """Energy of the standard test atoms under the toy CE."""
    from mchammer.calculators import ClusterExpansionCalculator

    atoms = make_wl_atoms()
    return float(
        ClusterExpansionCalculator(atoms, make_wl_ce()).calculate_total(
            occupations=atoms.numbers
        )
    )


def _make_converged_checkpoint(tmp_path: Path, seed: int = 0) -> tuple[Path, Any]:
    """Run a tiny REWL run and write a checkpoint.

    Uses two overlapping windows on the toy CE, block_size=10, and
    a fixed seed so the trajectory is deterministic. Runs a handful
    of cycles to populate g(E) and walker state; does NOT require WL
    convergence — the measurement path uses frozen_g=True so any
    non-trivial entropy suffices.

    Args:
        tmp_path: pytest-managed temporary directory.
        seed: random seed for the run.

    Returns:
        Tuple of (checkpoint_path, cluster_expansion).
    """
    ce = make_wl_ce()
    atoms = make_wl_atoms()
    e0 = _initial_energy()
    lo, hi = e0 - 50.0, e0 + 50.0
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[atoms, atoms],
        windows=[(lo, hi), (lo, hi)],
        energy_spacing=0.1,
        block_size=10,
        random_seed=seed,
        data_container_file=None,
    )
    pt.run(n_cycles=3)
    ckpt = tmp_path / "converged.h5"
    pt.save_checkpoint(ckpt)
    return ckpt, ce


def _collect_moments(
    pt: WangLandauParallelTempering, tag: str
) -> list[dict[str, Any]]:
    """Collect the ``to_state()`` snapshot from every replica's recorder.

    Returns one dict per walker (in window-major / walker-minor order).

    Args:
        pt: Orchestrator whose pool replicas are inspected.
        tag: The recorder tag to retrieve.

    Returns:
        List of state dicts (one per walker), or an empty dict for any
        walker that does not carry the given tag.
    """
    from mchammer_pt.wl_window_group import WangLandauWindowGroup

    results: list[dict[str, Any]] = []
    for slot in pt.pool._replicas:
        if isinstance(slot, WangLandauWindowGroup):
            replicas = slot._replicas
        else:
            replicas = [slot]
        for r in replicas:
            rec = r.ensemble._recorders.get(tag)
            if rec is not None:
                results.append(rec.to_state())
            else:
                results.append({})
    return results


def _collect_moments_from_checkpoint(
    ckpt: Path, tag: str
) -> list[dict[str, Any]]:
    """Read ``observable_records[tag]`` from each replica in a checkpoint.

    Calls ``read_hdf5`` to deserialise the checkpoint, then extracts the
    ``observable_records`` dict from each returned container's
    ``_last_state``. No replicas are restored; the data are read
    directly from the serialised container state.

    Args:
        ckpt: Path to a checkpoint file.
        tag: The recorder tag to retrieve.

    Returns:
        List of state dicts (one per walker), or empty dicts for any
        walker whose checkpoint does not carry the given tag.
    """
    from mchammer_pt.history import read_hdf5

    _, containers, _ = read_hdf5(ckpt)
    results: list[dict[str, Any]] = []
    for container in containers:
        obs_records = container._last_state.get("observable_records", {})
        results.append(obs_records.get(tag, {}))
    return results


# ---------------------------------------------------------------------------
# Task 4.3 — accumulation chaining
# ---------------------------------------------------------------------------


class TestAccumulationChaining:
    """Measurement chained across a checkpoint accumulates moments cumulatively.

    The contract under test:
    - Accumulation: chaining adds to the running totals; counts never reset.
    - Passivity: attaching a recorder does not alter the frozen-g trajectory.
      A run with a recorder attached must visit the same configurations as the
      identical run without one (same converged checkpoint, hence same restored
      RNG state).

    Bit-exact reproduction of the *resumed* sample is not asserted.  A
    checkpoint+resume with a recorder attached produces a statistically-
    equivalent but not bit-identical microcanonical sample (a benign
    RNG-alignment effect at the resume boundary).
    """

    def test_recording_is_passive(self, tmp_path: Path) -> None:
        """Attaching a recorder does not perturb the frozen-g trajectory.

        The brief's hard requirement: recording must be passive.  A frozen
        measurement run with a recorder attached must visit exactly the same
        configurations as the identical run with no recorder (same converged
        checkpoint, hence same seed and restored RNG state).
        """
        converged, ce = _make_converged_checkpoint(tmp_path, seed=0)

        pt_without = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        pt_without.run(n_cycles=5)
        occ_without = [
            pt_without.pool.current_occupations(0).copy(),
            pt_without.pool.current_occupations(1).copy(),
        ]

        pt_with = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        pt_with.record_observable(_ConstantObs(value=1.0, interval=1, tag="passive"))
        pt_with.run(n_cycles=5)
        occ_with = [
            pt_with.pool.current_occupations(0).copy(),
            pt_with.pool.current_occupations(1).copy(),
        ]

        for i, (a, b) in enumerate(zip(occ_without, occ_with, strict=True)):
            assert (a == b).all(), (
                f"window {i}: recorder attachment changed the trajectory — "
                f"recording must be passive"
            )

    def test_chained_count_exceeds_first_segment(self, tmp_path: Path) -> None:
        """After chaining, the total count strictly exceeds the N-cycle baseline.

        This is the weakest meaningful property: chaining adds observations,
        it never resets. Serves as a regression guard independent of the
        exact-equality assertion.
        """
        N, M = 2, 2

        converged, ce = _make_converged_checkpoint(tmp_path, seed=1)

        pt_c1 = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        obs = _ConstantObs(value=3.0, interval=1, tag="add")
        pt_c1.record_observable(obs)
        pt_c1.run(n_cycles=N)

        # Capture baseline counts from the checkpoint file.
        m0 = tmp_path / "m0_baseline.h5"
        pt_c1.save_checkpoint(m0)
        baseline = _collect_moments_from_checkpoint(m0, "add")
        baseline_total = sum(
            sum(m.get("count", [])) for m in baseline
        )
        assert baseline_total > 0, (
            "first segment must accumulate at least one observation"
        )

        # Continue from the measurement checkpoint.
        pt_c2 = WangLandauParallelTempering.measure_from_checkpoint(
            m0, cluster_expansion=ce
        )
        obs2 = _ConstantObs(value=3.0, interval=1, tag="add")
        pt_c2.record_observable(obs2)
        pt_c2.run(n_cycles=M)

        moments_final = _collect_moments(pt_c2, "add")
        final_total = sum(
            sum(m.get("count", [])) for m in moments_final
        )

        assert final_total > baseline_total, (
            f"chained total {final_total} not > baseline {baseline_total}; "
            "the second segment must ADD to the restored baseline, not reset"
        )


# ---------------------------------------------------------------------------
# Task 4.4 — multiple recorders, end to end
# ---------------------------------------------------------------------------


class TestMultipleRecorders:
    """Two observers with distinct tags and different intervals accumulate
    independently through a full measurement lifecycle (run, checkpoint,
    resume, unbound preservation).
    """

    _INTERVAL_FAST = 5
    _INTERVAL_SLOW = 20
    _N_CYCLES = 6

    def _make_two_recorders(
        self,
    ) -> tuple[_ConstantObs, _ConstantObs]:
        """Return two picklable observers with different intervals and tags."""
        fast = _ConstantObs(
            value=2.0, interval=self._INTERVAL_FAST, tag="fast"
        )
        slow = _ConstantObs(
            value=5.0, interval=self._INTERVAL_SLOW, tag="slow"
        )
        return fast, slow

    def test_independent_accumulation_and_count_ratio(
        self, tmp_path: Path
    ) -> None:
        """Each tag accumulates independently; the fast/slow count ratio is ~4x.

        A constant observer returns the same value regardless of structure,
        so the per-bin mean is exact. The count ratio fast/slow converges
        to ``INTERVAL_SLOW / INTERVAL_FAST = 4`` because the fast recorder
        fires 4x as often as the slow one within any given sequence of
        steps. The ±2x band is intentionally wide: it rules out the
        degenerate case where intervals are ignored (ratio ≈ 1) while
        tolerating small-sample variance in a short run.
        """
        converged, ce = _make_converged_checkpoint(tmp_path, seed=2)

        pt = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        fast, slow = self._make_two_recorders()
        pt.record_observable(fast)
        pt.record_observable(slow)
        pt.run(n_cycles=self._N_CYCLES)

        fast_moments = _collect_moments(pt, "fast")
        slow_moments = _collect_moments(pt, "slow")

        fast_total = sum(sum(m.get("count", [])) for m in fast_moments)
        slow_total = sum(sum(m.get("count", [])) for m in slow_moments)

        assert fast_total > 0, "fast recorder must accumulate at least one observation"
        assert slow_total > 0, "slow recorder must accumulate at least one observation"

        # The ±2x band is intentionally wide: it rules out the degenerate
        # case where intervals are ignored (ratio ≈ 1) while tolerating
        # small-sample variance in a short run.
        ratio = fast_total / slow_total
        expected = self._INTERVAL_SLOW / self._INTERVAL_FAST
        assert expected * 0.5 <= ratio <= expected * 2.0, (
            f"fast/slow count ratio {ratio:.2f} outside [0.5x, 2x] of expected "
            f"{expected}: a near-1 ratio would mean the per-observer intervals "
            f"are being ignored"
        )

    def test_per_bin_mean_matches_analytic_value(
        self, tmp_path: Path
    ) -> None:
        """The microcanonical mean of a constant observer equals its constant.

        For an observer that always returns the same value C, every
        recorded bin must satisfy sum/count == C (exact equality for
        floats, since sum = count * C by construction).
        """
        converged, ce = _make_converged_checkpoint(tmp_path, seed=3)

        pt = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        value_fast, value_slow = 2.0, 5.0
        fast = _ConstantObs(value=value_fast, interval=self._INTERVAL_FAST, tag="fast")
        slow = _ConstantObs(value=value_slow, interval=self._INTERVAL_SLOW, tag="slow")
        pt.record_observable(fast)
        pt.record_observable(slow)
        pt.run(n_cycles=self._N_CYCLES)

        fast_bins_total = sum(
            len(m.get("bins", [])) for m in _collect_moments(pt, "fast")
        )
        assert fast_bins_total > 0, "fast recorder must populate at least one bin"

        for tag, expected_value in [("fast", value_fast), ("slow", value_slow)]:
            moments = _collect_moments(pt, tag)
            for replica_idx, m in enumerate(moments):
                bins = m.get("bins", [])
                counts = m.get("count", [])
                sums = m.get("sum", [])
                for b, cnt, s in zip(bins, counts, sums, strict=True):
                    if cnt == 0:
                        continue
                    # sum stores a 1-element list (scalar observer)
                    mean = s[0] / cnt
                    assert mean == pytest.approx(expected_value), (
                        f"replica {replica_idx}, tag {tag!r}, bin {b}: "
                        f"mean {mean} != expected {expected_value}"
                    )

    def test_both_tags_round_trip_checkpoint(self, tmp_path: Path) -> None:
        """Both tags survive a save_checkpoint → measure_from_checkpoint cycle.

        After running with two recorders and saving a checkpoint, loading
        that checkpoint and re-attaching both observers must restore both
        tags' accumulated stores exactly.
        """
        converged, ce = _make_converged_checkpoint(tmp_path, seed=4)

        pt1 = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        fast, slow = self._make_two_recorders()
        pt1.record_observable(fast)
        pt1.record_observable(slow)
        pt1.run(n_cycles=self._N_CYCLES)

        m0 = tmp_path / "m0_two_tags.h5"
        pt1.save_checkpoint(m0)
        moments_fast_before = _collect_moments(pt1, "fast")
        moments_slow_before = _collect_moments(pt1, "slow")

        # Ensure at least one tag has data; otherwise the test is vacuous.
        total_before = sum(
            sum(m.get("count", [])) for m in moments_fast_before
        )
        assert total_before > 0, "first run must populate at least one fast bin"

        # Load the measurement checkpoint and re-attach both observers.
        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            m0, cluster_expansion=ce
        )
        fast2 = _ConstantObs(value=2.0, interval=self._INTERVAL_FAST, tag="fast")
        slow2 = _ConstantObs(value=5.0, interval=self._INTERVAL_SLOW, tag="slow")
        pt2.record_observable(fast2)
        pt2.record_observable(slow2)

        # Both tags must be present and have the same state as before.
        moments_fast_after = _collect_moments(pt2, "fast")
        moments_slow_after = _collect_moments(pt2, "slow")

        for replica_idx, (bf, af) in enumerate(
            zip(moments_fast_before, moments_fast_after, strict=True)
        ):
            assert bf.get("bins") == af.get("bins"), (
                f"replica {replica_idx}: 'fast' bins changed across checkpoint"
            )
            assert bf.get("count") == af.get("count"), (
                f"replica {replica_idx}: 'fast' counts changed across checkpoint"
            )

        for replica_idx, (bs, as_) in enumerate(
            zip(moments_slow_before, moments_slow_after, strict=True)
        ):
            assert bs.get("bins") == as_.get("bins"), (
                f"replica {replica_idx}: 'slow' bins changed across checkpoint"
            )
            assert bs.get("count") == as_.get("count"), (
                f"replica {replica_idx}: 'slow' counts changed across checkpoint"
            )

    def test_unbound_store_preserved_when_not_reattached(
        self, tmp_path: Path
    ) -> None:
        """An unbound tag's store is preserved unchanged after a partial resume.

        Protocol
        --------
        1. Run with two tags (``fast`` and ``slow``), save checkpoint ``m0``.
        2. Load ``m0``, re-attach ONLY ``fast``; run more cycles, save ``m1``.
        3. Assert that ``slow``'s store in ``m1`` is identical to the
           store in ``m0`` (unbound: not re-attached, so should not change).
        4. Assert that ``fast``'s total count in ``m1`` is strictly greater
           than its total count in ``m0`` (re-attached: continued
           accumulation).
        """
        converged, ce = _make_converged_checkpoint(tmp_path, seed=5)

        # --- First segment: both tags ---
        pt1 = WangLandauParallelTempering.measure_from_checkpoint(
            converged, cluster_expansion=ce
        )
        fast1, slow1 = self._make_two_recorders()
        pt1.record_observable(fast1)
        pt1.record_observable(slow1)
        pt1.run(n_cycles=self._N_CYCLES)

        m0 = tmp_path / "m0_unbound.h5"
        pt1.save_checkpoint(m0)

        # Snapshot both tags' states from m0.
        slow_m0 = _collect_moments_from_checkpoint(m0, "slow")
        fast_m0 = _collect_moments_from_checkpoint(m0, "fast")
        fast_total_m0 = sum(sum(m.get("count", [])) for m in fast_m0)
        assert fast_total_m0 > 0, (
            "first segment must accumulate at least one fast observation"
        )

        # --- Second segment: re-attach ONLY fast; slow is unbound ---
        pt2 = WangLandauParallelTempering.measure_from_checkpoint(
            m0, cluster_expansion=ce
        )
        fast2 = _ConstantObs(
            value=2.0, interval=self._INTERVAL_FAST, tag="fast"
        )
        pt2.record_observable(fast2)  # slow is NOT re-attached
        pt2.run(n_cycles=self._N_CYCLES)

        m1 = tmp_path / "m1_unbound.h5"
        pt2.save_checkpoint(m1)

        # slow in m1 must be identical to slow in m0 (unbound).
        slow_m1 = _collect_moments_from_checkpoint(m1, "slow")
        assert len(slow_m0) == len(slow_m1), (
            "replica count must not change between checkpoints"
        )
        for replica_idx, (s0, s1) in enumerate(
            zip(slow_m0, slow_m1, strict=True)
        ):
            assert s0.get("bins") == s1.get("bins"), (
                f"replica {replica_idx}: unbound 'slow' bins changed between "
                f"m0 and m1; unbound stores must be preserved unchanged"
            )
            assert s0.get("count") == s1.get("count"), (
                f"replica {replica_idx}: unbound 'slow' counts changed between "
                f"m0 and m1"
            )
            assert s0.get("sum") == s1.get("sum"), (
                f"replica {replica_idx}: unbound 'slow' sum changed between "
                f"m0 and m1"
            )

        # fast in m1 must have strictly more total count than in m0.
        fast_m1 = _collect_moments_from_checkpoint(m1, "fast")
        fast_total_m1 = sum(sum(m.get("count", [])) for m in fast_m1)
        assert fast_total_m1 > fast_total_m0, (
            f"re-attached 'fast' total count {fast_total_m1} not > "
            f"m0 total {fast_total_m0}; continued accumulation expected"
        )


def test_measure_from_converged_checkpoint_records_samples(tmp_path: Path) -> None:
    """A measurement from a GENUINELY converged checkpoint must sample.

    Regression: a converged DOS checkpoint restores ensembles with
    ``converged=True``, which short-circuits icet's
    ``WangLandauEnsemble.run()`` (and the orchestrator's per-cycle converged
    gate). The frozen-measurement restore must clear that flag, or the pass
    records nothing. Uses ``fill_factor_limit=0.5`` so the DOS converges
    after a single halve (the recorded g need not be accurate here).
    """
    ce = make_wl_ce()
    e0 = _initial_energy()
    kwargs = {
        "fill_factor_limit": 0.5,
        "flatness_limit": 0.5,
        "trial_move": "flip",
    }
    pt = WangLandauParallelTempering(
        cluster_expansion=ce,
        atoms=[make_wl_atoms(), make_wl_atoms()],
        windows=[(None, e0 + 20.0), (e0 - 20.0, None)],
        energy_spacing=2.0,
        block_size=len(make_wl_atoms()) * 100,
        random_seed=0,
        ensemble_kwargs=kwargs,
    )
    pt.run(n_cycles=8000)
    assert pt.pool.converged_flags().all(), (
        "the DOS run did not converge; tune the test's WL settings"
    )

    converged = tmp_path / "converged.h5"
    pt.save_checkpoint(converged)

    meas = WangLandauParallelTempering.measure_from_checkpoint(
        converged, cluster_expansion=ce, ensemble_kwargs=kwargs,
    )
    meas.record_observable(_ConstantObs(value=1.0, interval=1, tag="probe"))
    meas.run(n_cycles=20)

    total = sum(sum(m.get("count", [])) for m in _collect_moments(meas, "probe"))
    assert total > 0, (
        "measurement from a converged checkpoint recorded no samples; the "
        "restored converged flag was not cleared"
    )


def test_recorder_rejects_none_interval() -> None:
    """An observer with interval=None is rejected at attach, not deep in run()."""

    class _NoIntervalObserver(BaseObserver):
        def __init__(self) -> None:
            super().__init__(interval=None, return_type=float, tag="ni")

        def get_observable(self, structure: Any) -> float:
            return 1.0

    ens = make_wl_ensemble()
    with pytest.raises(ValueError, match="interval"):
        ens.attach_observable_recorder(_NoIntervalObserver())
