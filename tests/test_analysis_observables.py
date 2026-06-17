"""Tests for mchammer_pt.analysis.observables."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from mchammer.data_containers.wang_landau_data_container import (
    WangLandauDataContainer,
)

from mchammer_pt.analysis.observables import (
    stitch_observable_moments,
)


def _make_record(
    tag: str,
    names: list[str],
    bins: list[int],
    counts: list[int],
    sums: list[list[float]],
    sum2s: list[list[float]],
    sum4s: list[list[float]],
    interval: int = 1,
) -> dict:
    """Build a to_state() dict directly."""
    return {
        "tag": tag,
        "names": names,
        "interval": interval,
        "bins": bins,
        "count": counts,
        "sum": sums,
        "sum2": sum2s,
        "sum4": sum4s,
        "skipped": {},
    }


def _mock_wl_dc(
    observable_records: dict,
    energy_spacing: float,
    energy_limit_left: float | None,
    energy_limit_right: float | None,
) -> object:
    """Build a mock WangLandauDataContainer with observable_records in _last_state."""
    dc = MagicMock(spec=WangLandauDataContainer)
    dc._last_state = {
        "entropy": {},
        "histogram": {},
        "fill_factor": 0.5,
        "fill_factor_history": {},
        "entropy_history": {},
        "observable_records": observable_records,
    }
    dc.ensemble_parameters = {
        "energy_spacing": energy_spacing,
        "energy_limit_left": energy_limit_left,
        "energy_limit_right": energy_limit_right,
    }
    return dc


# ---------------------------------------------------------------------------
# Fixtures: two overlapping windows with two walkers in window 0 and one
# walker in window 1.
#
#   energy_spacing = 1.0
#   Window 0: energy_limit_left=0.0, energy_limit_right=4.0
#             bins 0..4 are in-window; bins < 0 or > 4 are out-of-window
#   Window 1: energy_limit_left=2.0, energy_limit_right=6.0
#             bins 2..6 are in-window
#   Overlap region: bins 2..4 (energies 2.0, 3.0, 4.0)
# ---------------------------------------------------------------------------

ENERGY_SPACING = 1.0


def _make_walker0a() -> dict:
    """Walker 0a: window 0, tag 'energy', S=1, bins 0-4 plus out-of-window bin 5."""
    return _make_record(
        tag="energy",
        names=["energy"],
        bins=[0, 1, 2, 3, 4, 5],   # bin 5 (energy=5.0) is outside window 0 (right=4.0)
        counts=[10, 10, 10, 10, 10, 5],
        sums=[[1.0], [2.0], [3.0], [4.0], [5.0], [99.0]],
        sum2s=[[1.0], [4.0], [9.0], [16.0], [25.0], [99.0**2]],
        sum4s=[[1.0], [16.0], [81.0], [256.0], [625.0], [99.0**4]],
    )


def _make_walker0b() -> dict:
    """Walker 0b: also in window 0, same tag, same bins 0..4."""
    return _make_record(
        tag="energy",
        names=["energy"],
        bins=[0, 1, 2, 3, 4],
        counts=[5, 5, 5, 5, 5],
        sums=[[0.5], [1.0], [1.5], [2.0], [2.5]],
        sum2s=[[0.25], [1.0], [2.25], [4.0], [6.25]],
        sum4s=[[0.0625], [1.0], [5.0625], [16.0], [39.0625]],
    )


def _make_walker1() -> dict:
    """Walker 1: in window 1, tag 'energy', bins 2..6. Bins 0,1 out of window."""
    return _make_record(
        tag="energy",
        names=["energy"],
        bins=[1, 2, 3, 4, 5, 6],  # bin 1 (energy=1.0) is outside window 1 (left=2.0)
        counts=[99, 20, 20, 20, 20, 20],
        sums=[[999.0], [10.0], [11.0], [12.0], [13.0], [14.0]],
        sum2s=[[999.0**2], [100.0], [121.0], [144.0], [169.0], [196.0]],
        sum4s=[[999.0**4], [10000.0], [14641.0], [20736.0], [28561.0], [38416.0]],
    )


def _make_containers_single_tag() -> list:
    """Two walkers in window 0, one walker in window 1."""
    dc0a = _mock_wl_dc(
        observable_records={"energy": _make_walker0a()},
        energy_spacing=ENERGY_SPACING,
        energy_limit_left=0.0,
        energy_limit_right=4.0,
    )
    dc0b = _mock_wl_dc(
        observable_records={"energy": _make_walker0b()},
        energy_spacing=ENERGY_SPACING,
        energy_limit_left=0.0,
        energy_limit_right=4.0,
    )
    dc1 = _mock_wl_dc(
        observable_records={"energy": _make_walker1()},
        energy_spacing=ENERGY_SPACING,
        energy_limit_left=2.0,
        energy_limit_right=6.0,
    )
    return [dc0a, dc0b, dc1]


# ---------------------------------------------------------------------------
# stitch_observable_moments: basic behaviour
# ---------------------------------------------------------------------------

class TestStitchObservableMoments:
    """Core behaviour of stitch_observable_moments."""

    def test_output_is_dict_keyed_by_tag(self):
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        assert isinstance(result, dict)
        assert "energy" in result

    def test_output_dataframe_has_required_columns(self):
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"]
        assert list(df.columns) == [
            "energy", "count", "energy_sum", "energy_sum2", "energy_sum4",
        ]

    def test_energy_column_is_bin_times_spacing(self):
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"]
        # All energies must be integer multiples of energy_spacing
        energies = df["energy"].to_numpy()
        residuals = energies % ENERGY_SPACING
        assert np.allclose(residuals, 0.0, atol=1e-9)

    def test_sorted_by_energy_ascending(self):
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"]
        assert list(df["energy"]) == sorted(df["energy"].tolist())

    def test_only_bins_with_positive_count_emitted(self):
        # A container with no records contributes nothing.
        dc_empty = _mock_wl_dc(
            observable_records={},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=0.0,
            energy_limit_right=4.0,
        )
        containers = _make_containers_single_tag() + [dc_empty]
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"]
        assert (df["count"] > 0).all()


class TestInWindowFiltering:
    """Out-of-window bins are excluded; overlap bins accumulate from both windows."""

    def test_out_of_window_bins_excluded(self):
        """Walker 0a has bin 5 (energy=5.0) outside window 0 (right=4.0);
        walker 1 has bin 1 (energy=1.0) outside window 1 (left=2.0).
        Neither should contribute to the merged result for those energies
        solely from an out-of-window walker."""
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"].set_index("energy")

        # Bin 1 (energy=1.0): only walkers 0a and 0b contribute.
        # Walker 1's bin 1 is outside its window (left=2.0), so excluded.
        e1_count = df.loc[1.0, "count"]
        # Walker 0a: 10 counts at bin 1; walker 0b: 5 counts at bin 1 => 15 total
        assert e1_count == 15, f"expected 15 at energy=1.0, got {e1_count}"

        # Bin 5 (energy=5.0): only walker 1 contributes.
        # Walker 0a's bin 5 is outside its window (right=4.0), so excluded.
        e5_count = df.loc[5.0, "count"]
        # Walker 1: 20 counts at bin 5
        assert e5_count == 20, f"expected 20 at energy=5.0, got {e5_count}"

    def test_overlap_bins_accumulate_from_both_windows(self):
        """Bins 2,3,4 are in both window 0 and window 1; both walkers contribute."""
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"].set_index("energy")

        # Bin 2 (energy=2.0):
        #   Walker 0a: count=10, sum=[3.0]
        #   Walker 0b: count=5,  sum=[1.5]
        #   Walker 1:  count=20, sum=[10.0]  (in-window: left=2.0 <= 2.0*1.0 <= 6.0)
        # Total count = 35, total sum = 14.5
        assert df.loc[2.0, "count"] == 35, f"energy=2.0 count: {df.loc[2.0, 'count']}"
        assert df.loc[2.0, "energy_sum"] == pytest.approx(14.5), (
            f"energy=2.0 sum: {df.loc[2.0, 'energy_sum']}"
        )

    def test_bins_outside_all_windows_are_absent(self):
        """No bin exists that no walker had in-window."""
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"]
        # The full range of energies present should be 0..6
        # (0,1 from walkers 0a/0b; 2,3,4 from all; 5,6 from walker 1)
        expected_energies = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
        actual_energies = set(df["energy"].tolist())
        assert actual_energies == expected_energies

    def test_none_bounds_are_treated_as_unbounded(self):
        """None on either window bound means unbounded in that direction."""
        dc = _mock_wl_dc(
            observable_records={"e": _make_record(
                tag="e",
                names=["e"],
                bins=[-100, -1, 0, 1, 100],
                counts=[10, 10, 10, 10, 10],
                sums=[[1.0]] * 5,
                sum2s=[[1.0]] * 5,
                sum4s=[[1.0]] * 5,
            )},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        result = stitch_observable_moments([dc], ENERGY_SPACING)
        df = result["e"]
        assert len(df) == 5  # all bins included


class TestBinwiseSumAddition:
    """Moments are added bin-wise (extensive: plain addition, no rebasing)."""

    def test_count_sums_across_walkers(self):
        """Bin 0 (energy=0.0): walkers 0a and 0b only (in-window for window 0).
        Walker 0a count=10, walker 0b count=5 => total 15."""
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"].set_index("energy")
        assert df.loc[0.0, "count"] == 15

    def test_sum_adds_correctly(self):
        """sum column adds bin-wise across contributing walkers."""
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"].set_index("energy")
        # Bin 0: walker 0a sum=[1.0], walker 0b sum=[0.5] => 1.5
        assert df.loc[0.0, "energy_sum"] == pytest.approx(1.5)

    def test_sum2_adds_correctly(self):
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"].set_index("energy")
        # Bin 0: walker 0a sum2=[1.0], walker 0b sum2=[0.25] => 1.25
        assert df.loc[0.0, "energy_sum2"] == pytest.approx(1.25)

    def test_sum4_adds_correctly(self):
        containers = _make_containers_single_tag()
        result = stitch_observable_moments(containers, ENERGY_SPACING)
        df = result["energy"].set_index("energy")
        # Bin 0: walker 0a sum4=[1.0], walker 0b sum4=[0.0625] => 1.0625
        assert df.loc[0.0, "energy_sum4"] == pytest.approx(1.0625)


class TestMultiScalarObserver:
    """S=2 observer produces correctly named per-scalar columns."""

    def _make_s2_containers(self) -> list:
        """Two walkers with a 2-scalar observer ('a', 'b') in a single window."""
        r0 = _make_record(
            tag="vec",
            names=["a", "b"],
            bins=[0, 1, 2],
            counts=[10, 10, 10],
            sums=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            sum2s=[[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]],
            sum4s=[[1.0, 16.0], [81.0, 256.0], [625.0, 1296.0]],
        )
        r1 = _make_record(
            tag="vec",
            names=["a", "b"],
            bins=[1, 2, 3],
            counts=[5, 5, 5],
            sums=[[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
            sum2s=[[0.25, 1.0], [2.25, 4.0], [6.25, 9.0]],
            sum4s=[[0.0625, 1.0], [5.0625, 16.0], [39.0625, 81.0]],
        )
        dc0 = _mock_wl_dc(
            observable_records={"vec": r0},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        dc1 = _mock_wl_dc(
            observable_records={"vec": r1},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        return [dc0, dc1]

    def test_s2_columns_present(self):
        result = stitch_observable_moments(self._make_s2_containers(), ENERGY_SPACING)
        df = result["vec"]
        expected_cols = [
            "energy", "count",
            "a_sum", "a_sum2", "a_sum4", "b_sum", "b_sum2", "b_sum4",
        ]
        assert list(df.columns) == expected_cols

    def test_s2_sum_values_correct(self):
        result = stitch_observable_moments(self._make_s2_containers(), ENERGY_SPACING)
        df = result["vec"].set_index("energy")
        # Bin 1: r0 count=10 sum=[3,4]; r1 count=5 sum=[0.5,1] -> a_sum=3.5, b_sum=5.0
        assert df.loc[1.0, "a_sum"] == pytest.approx(3.5)
        assert df.loc[1.0, "b_sum"] == pytest.approx(5.0)

    def test_s2_count_correct(self):
        result = stitch_observable_moments(self._make_s2_containers(), ENERGY_SPACING)
        df = result["vec"].set_index("energy")
        # Bin 1: r0 count=10, r1 count=5 => 15
        assert df.loc[1.0, "count"] == 15

    def test_s2_bin0_only_from_r0(self):
        """Bin 0 only exists in r0."""
        result = stitch_observable_moments(self._make_s2_containers(), ENERGY_SPACING)
        df = result["vec"].set_index("energy")
        assert df.loc[0.0, "count"] == 10
        assert df.loc[0.0, "a_sum"] == pytest.approx(1.0)
        assert df.loc[0.0, "b_sum"] == pytest.approx(2.0)


class TestMultipleTags:
    """Multiple tags produce separate DataFrames, keyed correctly."""

    def test_two_tags_returned(self):
        r_energy = _make_record(
            tag="energy", names=["energy"],
            bins=[0, 1], counts=[10, 10],
            sums=[[1.0], [2.0]], sum2s=[[1.0], [4.0]], sum4s=[[1.0], [16.0]],
        )
        r_mag = _make_record(
            tag="mag", names=["mag"],
            bins=[0, 1], counts=[8, 8],
            sums=[[0.1], [0.2]], sum2s=[[0.01], [0.04]], sum4s=[[0.0001], [0.0016]],
        )
        dc = _mock_wl_dc(
            observable_records={"energy": r_energy, "mag": r_mag},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        result = stitch_observable_moments([dc], ENERGY_SPACING)
        assert set(result.keys()) == {"energy", "mag"}

    def test_tags_are_independent(self):
        r_energy = _make_record(
            tag="energy", names=["energy"],
            bins=[0], counts=[10],
            sums=[[1.0]], sum2s=[[1.0]], sum4s=[[1.0]],
        )
        r_mag = _make_record(
            tag="mag", names=["mag"],
            bins=[0, 1], counts=[5, 5],
            sums=[[0.5], [0.5]], sum2s=[[0.25], [0.25]], sum4s=[[0.0625], [0.0625]],
        )
        dc = _mock_wl_dc(
            observable_records={"energy": r_energy, "mag": r_mag},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        result = stitch_observable_moments([dc], ENERGY_SPACING)
        assert len(result["energy"]) == 1  # only bin 0
        assert len(result["mag"]) == 2      # bins 0 and 1


class TestContainersWithNoRecords:
    """Containers without observable_records contribute nothing."""

    def test_container_with_no_observable_records_key(self):
        """Container whose _last_state lacks 'observable_records' is skipped."""
        dc_no_records = MagicMock(spec=WangLandauDataContainer)
        dc_no_records._last_state = {"entropy": {}}
        dc_no_records.ensemble_parameters = {
            "energy_spacing": ENERGY_SPACING,
            "energy_limit_left": None,
            "energy_limit_right": None,
        }
        r = _make_record(
            tag="energy", names=["energy"],
            bins=[0], counts=[5],
            sums=[[1.0]], sum2s=[[1.0]], sum4s=[[1.0]],
        )
        dc_with = _mock_wl_dc(
            observable_records={"energy": r},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        result = stitch_observable_moments([dc_no_records, dc_with], ENERGY_SPACING)
        assert "energy" in result
        assert result["energy"].loc[0, "count"] == 5

    def test_all_containers_empty_returns_empty_dict(self):
        dc = _mock_wl_dc(
            observable_records={},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        result = stitch_observable_moments([dc], ENERGY_SPACING)
        assert result == {}


class TestSignatureMismatch:
    """Walkers with inconsistent names for the same tag raise ValueError."""

    def test_mismatched_names_raises(self):
        r0 = _make_record(
            tag="obs", names=["a"],
            bins=[0], counts=[5],
            sums=[[1.0]], sum2s=[[1.0]], sum4s=[[1.0]],
        )
        r1 = _make_record(
            tag="obs", names=["b"],  # different name
            bins=[0], counts=[5],
            sums=[[1.0]], sum2s=[[1.0]], sum4s=[[1.0]],
        )
        dc0 = _mock_wl_dc(
            observable_records={"obs": r0},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        dc1 = _mock_wl_dc(
            observable_records={"obs": r1},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        with pytest.raises(ValueError, match="inconsistent"):
            stitch_observable_moments([dc0, dc1], ENERGY_SPACING)

    def test_mismatched_size_raises(self):
        r0 = _make_record(
            tag="obs", names=["a", "b"],
            bins=[0], counts=[5],
            sums=[[1.0, 2.0]], sum2s=[[1.0, 4.0]], sum4s=[[1.0, 16.0]],
        )
        r1 = _make_record(
            tag="obs", names=["a"],  # S=1 vs S=2
            bins=[0], counts=[5],
            sums=[[1.0]], sum2s=[[1.0]], sum4s=[[1.0]],
        )
        dc0 = _mock_wl_dc(
            observable_records={"obs": r0},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        dc1 = _mock_wl_dc(
            observable_records={"obs": r1},
            energy_spacing=ENERGY_SPACING,
            energy_limit_left=None,
            energy_limit_right=None,
        )
        with pytest.raises(ValueError, match="inconsistent"):
            stitch_observable_moments([dc0, dc1], ENERGY_SPACING)


def test_skipped_total_surfaced_in_dataframe_attrs():
    """Dropped non-finite observations are summed per tag into df.attrs['skipped']."""
    record = _make_record(
        tag="obs", names=["obs"],
        bins=[1, 2], counts=[10, 10],
        sums=[[1.0], [2.0]], sum2s=[[1.0], [4.0]], sum4s=[[1.0], [16.0]],
    )
    record["skipped"] = {0: 3, 1: 2}  # five dropped observations in total
    dc = _mock_wl_dc(
        observable_records={"obs": record},
        energy_spacing=1.0,
        energy_limit_left=None,
        energy_limit_right=None,
    )
    frames = stitch_observable_moments([dc], 1.0)
    assert frames["obs"].attrs["skipped"] == 5


def test_in_window_filter_is_round_based_at_fractional_edge():
    """A fractional window edge uses round(hi/spacing), matching the recorder.

    With hi=2.6 and spacing=1.0, round(2.6)=3, so bin 3 (energy 3.0) is
    in-window even though 3.0 > 2.6; bin 4 is excluded. A raw energy<=hi
    filter would wrongly drop bin 3 -- a bin the recorder counted.
    """
    record = _make_record(
        tag="obs", names=["obs"],
        bins=[0, 1, 2, 3, 4], counts=[10, 10, 10, 10, 10],
        sums=[[0.0], [1.0], [2.0], [3.0], [4.0]],
        sum2s=[[0.0], [1.0], [4.0], [9.0], [16.0]],
        sum4s=[[0.0], [1.0], [16.0], [81.0], [256.0]],
    )
    dc = _mock_wl_dc(
        observable_records={"obs": record},
        energy_spacing=1.0,
        energy_limit_left=None,
        energy_limit_right=2.6,
    )
    df = stitch_observable_moments([dc], 1.0)["obs"]
    energies = df["energy"].tolist()
    assert 3.0 in energies  # round(2.6)=3 -> bin 3 kept (round-based)
    assert 4.0 not in energies  # bin 4 > round(2.6)=3 -> excluded
