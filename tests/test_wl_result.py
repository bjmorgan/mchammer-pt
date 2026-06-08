"""Unit tests for WindowResult."""

from __future__ import annotations

import pandas as pd
import pytest

from mchammer_pt.wl_result import WindowResult


def _make_mock_container(
    entropy: dict[int, float],
    histogram: dict[int, int],
    energy_spacing: float,
    fill_factor: float = 0.5,
    fill_factor_history: dict | None = None,
    entropy_history: dict | None = None,
    fill_factor_snapshots: dict | None = None,
    entropy_snapshots: dict | None = None,
) -> object:
    """Build a minimal mock that quacks like WangLandauDataContainer."""
    from unittest.mock import MagicMock

    container = MagicMock()
    container._last_state = {
        "entropy": dict(entropy),
        "histogram": dict(histogram),
        "fill_factor": fill_factor,
        "fill_factor_history": fill_factor_history or {},
        "entropy_history": entropy_history or {},
        "fill_factor_snapshots": fill_factor_snapshots or {},
        "entropy_snapshots": entropy_snapshots or {},
    }
    container.ensemble_parameters = {"energy_spacing": energy_spacing}
    container.fill_factor = fill_factor
    return container


def test_get_entropy_merges_two_walkers():
    """Entropy is averaged bin-wise across walkers."""
    c0 = _make_mock_container(
        entropy={0: 2.0, 1: 4.0},
        histogram={0: 10, 1: 5},
        energy_spacing=0.5,
    )
    c1 = _make_mock_container(
        entropy={0: 6.0, 1: 8.0},
        histogram={0: 8, 1: 3},
        energy_spacing=0.5,
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=1.0,
        energy_spacing=0.5,
        containers=(c0, c1),
    )
    df = wr.get_entropy()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"energy", "entropy"}
    # bin 0: (2+6)/2 = 4.0, bin 1: (4+8)/2 = 6.0
    # After min-shift: 4-4=0, 6-4=2
    row0 = df.loc[0]
    row1 = df.loc[1]
    assert row0["energy"] == pytest.approx(0.0)  # 0 * 0.5
    assert row1["energy"] == pytest.approx(0.5)  # 1 * 0.5
    assert row0["entropy"] == pytest.approx(0.0)  # min-shifted
    assert row1["entropy"] == pytest.approx(2.0)


def test_get_histogram_sums_two_walkers():
    """Histogram is summed bin-wise across walkers."""
    c0 = _make_mock_container(
        entropy={0: 1.0},
        histogram={0: 10, 1: 5},
        energy_spacing=0.5,
    )
    c1 = _make_mock_container(
        entropy={0: 1.0},
        histogram={0: 8, 2: 3},
        energy_spacing=0.5,
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=1.0,
        energy_spacing=0.5,
        containers=(c0, c1),
    )
    df = wr.get_histogram()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"energy", "histogram"}
    # bin 0: 10+8=18, bin 1: 5+0=5, bin 2: 0+3=3
    assert df.loc[0, "histogram"] == 18
    assert df.loc[1, "histogram"] == 5
    assert df.loc[2, "histogram"] == 3


def test_n_walkers():
    """n_walkers returns the number of containers."""
    c = _make_mock_container(entropy={}, histogram={}, energy_spacing=0.1)
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c, c, c),
    )
    assert wr.n_walkers == 3


def test_get_entropy_single_walker_matches_container():
    """W=1: get_entropy produces same values as the container would."""
    entropy = {0: 3.0, 1: 5.0, 2: 4.0}
    c = _make_mock_container(
        entropy=entropy,
        histogram={0: 10, 1: 8, 2: 6},
        energy_spacing=0.25,
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=1.0,
        energy_spacing=0.25,
        containers=(c,),
    )
    df = wr.get_entropy()
    # Entropy: {0: 3.0, 1: 5.0, 2: 4.0}
    # min=3.0, shifted: {0: 0, 1: 2, 2: 1}
    assert df.loc[0, "entropy"] == pytest.approx(0.0)
    assert df.loc[1, "entropy"] == pytest.approx(2.0)
    assert df.loc[2, "entropy"] == pytest.approx(1.0)
    assert df.loc[0, "energy"] == pytest.approx(0.0)
    assert df.loc[1, "energy"] == pytest.approx(0.25)
    assert df.loc[2, "energy"] == pytest.approx(0.50)


def test_get_entropy_empty_returns_empty_dataframe():
    """Empty entropy dicts produce an empty DataFrame."""
    c = _make_mock_container(entropy={}, histogram={}, energy_spacing=0.1)
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c,),
    )
    df = wr.get_entropy()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_get_entropy_no_entropy_key_returns_none():
    """Container with no entropy data returns None."""
    from unittest.mock import MagicMock

    c = MagicMock()
    c._last_state = {"histogram": {0: 5}}
    c.ensemble_parameters = {"energy_spacing": 0.1}
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c,),
    )
    assert wr.get_entropy() is None


def test_get_histogram_no_histogram_key_returns_none():
    """Container with no histogram data returns None."""
    from unittest.mock import MagicMock

    c = MagicMock()
    c._last_state = {"entropy": {0: 1.0}}
    c.ensemble_parameters = {"energy_spacing": 0.1}
    wr = WindowResult(
        energy_limit_left=0.0,
        energy_limit_right=1.0,
        energy_spacing=0.1,
        containers=(c,),
    )
    assert wr.get_histogram() is None


def test_get_entropy_respects_fill_factor_limit():
    """fill_factor_limit selects historical entropy per walker before merge."""
    c0 = _make_mock_container(
        entropy={0: 10.0, 1: 20.0},
        histogram={0: 5, 1: 3},
        energy_spacing=1.0,
        fill_factor=0.125,
        fill_factor_history={0: 1.0, 100: 0.5, 200: 0.25, 300: 0.125},
        entropy_history={
            100: {0: 2.0, 1: 4.0},
            200: {0: 3.0, 1: 5.0},
            300: {0: 10.0, 1: 20.0},
        },
    )
    c1 = _make_mock_container(
        entropy={0: 12.0, 1: 22.0},
        histogram={0: 7, 1: 2},
        energy_spacing=1.0,
        fill_factor=0.125,
        fill_factor_history={0: 1.0, 100: 0.5, 200: 0.25, 300: 0.125},
        entropy_history={
            100: {0: 6.0, 1: 8.0},
            200: {0: 7.0, 1: 9.0},
            300: {0: 12.0, 1: 22.0},
        },
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c0, c1),
    )
    # fill_factor_limit=0.5: first ff_history entry <= 0.5 is step 100
    # c0 entropy at step 100: {0: 2.0, 1: 4.0}
    # c1 entropy at step 100: {0: 6.0, 1: 8.0}
    # merged: {0: 4.0, 1: 6.0}, shifted: {0: 0.0, 1: 2.0}
    df = wr.get_entropy(fill_factor_limit=0.5)
    assert df.loc[0, "entropy"] == pytest.approx(0.0)
    assert df.loc[1, "entropy"] == pytest.approx(2.0)


def test_get_entropy_fill_factor_limit_not_yet_reached():
    """fill_factor_limit returns None when the walker hasn't converged far enough."""
    c = _make_mock_container(
        entropy={0: 1.0, 1: 2.0},
        histogram={0: 5, 1: 3},
        energy_spacing=1.0,
        fill_factor=0.5,
        fill_factor_history={0: 1.0, 100: 0.5},
        entropy_history={100: {0: 1.0, 1: 2.0}},
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c,),
    )
    assert wr.get_entropy(fill_factor_limit=0.25) is None


def test_get_entropy_fill_factor_limit_no_matching_history_step():
    """Returns None when fill_factor passes the gate but no history step matches."""
    c = _make_mock_container(
        entropy={0: 1.0, 1: 2.0},
        histogram={0: 5, 1: 3},
        energy_spacing=1.0,
        fill_factor=0.25,
        fill_factor_history={0: 1.0, 100: 0.5},
        entropy_history={100: {0: 1.0, 1: 2.0}},
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c,),
    )
    assert wr.get_entropy(fill_factor_limit=0.25) is None


def test_get_entropy_reads_one_over_t_snapshot_below_last_halving():
    """A limit between the last halving and the final f resolves to a 1/t
    snapshot -- the case that returns None without the snapshot store."""
    c = _make_mock_container(
        entropy={0: 10.0, 1: 20.0},
        histogram={0: 5, 1: 3},
        energy_spacing=1.0,
        fill_factor=1.0 / 128,
        fill_factor_history={0: 1.0, 100: 0.25, 300: 1.0 / 32},
        entropy_history={
            100: {0: 1.0, 1: 2.0},
            300: {0: 5.0, 1: 9.0},
        },
        fill_factor_snapshots={400: 1.0 / 64, 500: 1.0 / 128},
        entropy_snapshots={
            400: {0: 8.0, 1: 16.0},
            500: {0: 10.0, 1: 20.0},
        },
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c,),
    )
    # Limit 1/64 is below the last halving (1/32): first step with
    # ff <= 1/64 is step 400 -> entropy {0: 8, 1: 16}, shifted {0: 0, 1: 8}.
    df = wr.get_entropy(fill_factor_limit=1.0 / 64)
    assert df is not None
    assert df.loc[0, "entropy"] == pytest.approx(0.0)
    assert df.loc[1, "entropy"] == pytest.approx(8.0)


def test_get_entropy_union_scan_picks_chronologically_first_crossing():
    """The union scan returns the earliest step whose f <= limit, whether
    that step sits in the halving history or the 1/t snapshot store."""
    c = _make_mock_container(
        entropy={0: 0.0, 1: 0.0},
        histogram={0: 1, 1: 1},
        energy_spacing=1.0,
        fill_factor=1.0 / 128,
        fill_factor_history={0: 1.0, 100: 0.25, 300: 1.0 / 32},
        entropy_history={
            100: {0: 1.0, 1: 1.0},
            300: {0: 2.0, 1: 2.0},
        },
        fill_factor_snapshots={400: 1.0 / 64, 500: 1.0 / 128},
        entropy_snapshots={
            400: {0: 4.0, 1: 4.0},
            500: {0: 8.0, 1: 8.0},
        },
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c,),
    )
    # Limit 0.25: first step with ff <= 0.25 is step 100 (halving history).
    df_coarse = wr.get_entropy(fill_factor_limit=0.25)
    assert df_coarse is not None
    # Limit 1/100 (between 1/64 and 1/128): first step with ff <= 1/100 is
    # step 500 (the finest snapshot). entropy {0: 8, 1: 8} -> shifted 0.
    df_fine = wr.get_entropy(fill_factor_limit=1.0 / 100)
    assert df_fine is not None
    assert df_fine.loc[0, "entropy"] == pytest.approx(0.0)


def test_get_entropy_one_over_t_snapshot_merges_two_walkers():
    """Per-walker 1/t snapshots merge across walkers at a requested rung."""
    c0 = _make_mock_container(
        entropy={0: 0.0, 1: 0.0},
        histogram={0: 1, 1: 1},
        energy_spacing=1.0,
        fill_factor=1.0 / 64,
        fill_factor_history={0: 1.0, 100: 1.0 / 32},
        entropy_history={100: {0: 1.0, 1: 1.0}},
        fill_factor_snapshots={400: 1.0 / 64},
        entropy_snapshots={400: {0: 2.0, 1: 4.0}},
    )
    c1 = _make_mock_container(
        entropy={0: 0.0, 1: 0.0},
        histogram={0: 1, 1: 1},
        energy_spacing=1.0,
        fill_factor=1.0 / 64,
        fill_factor_history={0: 1.0, 100: 1.0 / 32},
        entropy_history={100: {0: 1.0, 1: 1.0}},
        fill_factor_snapshots={420: 1.0 / 64},
        entropy_snapshots={420: {0: 6.0, 1: 8.0}},
    )
    wr = WindowResult(
        energy_limit_left=-1.0,
        energy_limit_right=2.0,
        energy_spacing=1.0,
        containers=(c0, c1),
    )
    # Each walker's snapshot at f = 1/64: c0 {0:2,1:4}, c1 {0:6,1:8}.
    # merge_entropies rebases per walker then averages, then min-shifts.
    # Both walkers have slope +2 from bin 0 to bin 1, so merged is {0:0,1:2}.
    df = wr.get_entropy(fill_factor_limit=1.0 / 64)
    assert df is not None
    assert df.loc[0, "entropy"] == pytest.approx(0.0)
    assert df.loc[1, "entropy"] == pytest.approx(2.0)


def test_reconstruct_dos_ladder_and_stitch_each_rung():
    """Reconstruct per-window DOS at a ladder of fill factors below the last
    halving, stitch each rung, and confirm the whole-range DOS is recoverable
    at every rung and refines between rungs."""
    from mchammer_pt.analysis.dos import stitch_entropy

    energy_spacing = 1.0
    # Two overlapping windows on a common grid: window A covers bins 0..3,
    # window B covers bins 2..5 (overlap at bins 2, 3). Each window's walker
    # carries a 1/t snapshot store at three rungs f = 1/64, 1/128, 1/256,
    # with the entropy curve sharpening (steeper slope) as f decreases.
    def window_container(bins, slopes_by_ff):
        ff_snaps = {}
        ent_snaps = {}
        step = 400
        for ff, slope in slopes_by_ff.items():
            ff_snaps[step] = ff
            ent_snaps[step] = {b: slope * b for b in bins}
            step += 100
        finest_ff = min(slopes_by_ff)
        return _make_mock_container(
            entropy=ent_snaps[max(ent_snaps)],
            histogram={b: 5 for b in bins},
            energy_spacing=energy_spacing,
            fill_factor=finest_ff,
            fill_factor_history={0: 1.0, 100: 1.0 / 32},
            entropy_history={100: {b: 0.0 for b in bins}},
            fill_factor_snapshots=ff_snaps,
            entropy_snapshots=ent_snaps,
        )

    ladder = [1.0 / 64, 1.0 / 128, 1.0 / 256]
    cA = window_container(
        [0, 1, 2, 3], {1.0 / 64: 1.0, 1.0 / 128: 1.5, 1.0 / 256: 2.0}
    )
    cB = window_container(
        [2, 3, 4, 5], {1.0 / 64: 1.0, 1.0 / 128: 1.5, 1.0 / 256: 2.0}
    )
    wrA = WindowResult(
        energy_limit_left=0.0, energy_limit_right=3.0,
        energy_spacing=energy_spacing, containers=(cA,),
    )
    wrB = WindowResult(
        energy_limit_left=2.0, energy_limit_right=5.0,
        energy_spacing=energy_spacing, containers=(cB,),
    )

    stitched_by_rung = {}
    for limit in ladder:
        dfA = wrA.get_entropy(fill_factor_limit=limit)
        dfB = wrB.get_entropy(fill_factor_limit=limit)
        # Every rung lies below the last halving (1/32): without the
        # snapshot store these would be None.
        assert dfA is not None and dfB is not None
        stitched, _errors = stitch_entropy([dfA, dfB], energy_spacing)
        stitched_by_rung[limit] = stitched

    # A whole-range DOS is recoverable at every rung...
    assert len(stitched_by_rung) == len(ladder)
    for stitched in stitched_by_rung.values():
        assert not stitched.empty
        assert set(stitched.columns) >= {"energy", "entropy"}
    # ...and the reconstruction refines between rungs (the finest rung's
    # top-bin entropy exceeds the coarsest rung's, reflecting the sharper
    # slope). Compare the highest populated bin.
    coarse = stitched_by_rung[1.0 / 64]
    fine = stitched_by_rung[1.0 / 256]
    assert fine["entropy"].max() > coarse["entropy"].max()
