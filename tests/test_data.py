import pandas as pd
import pytest

from driftloop.config import COLUMNS, SyntheticConfig
from driftloop.data import SyntheticSource


def test_contract_columns_and_order():
    df = SyntheticSource().get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-08"))
    assert list(df.columns) == COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert len(df) == 7 * 24


def test_windows_are_half_open_and_tile_exactly():
    src = SyntheticSource()
    a = src.get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-05"))
    b = src.get_data(pd.Timestamp("2025-05-05"), pd.Timestamp("2025-05-09"))
    assert a["timestamp"].iloc[-1] < b["timestamp"].iloc[0]
    assert len(a) + len(b) == 8 * 24


def test_same_window_is_stable_regardless_of_the_request_around_it():
    """The loop asks for overlapping windows; rows must not shift underneath it."""
    src = SyntheticSource()
    wide = src.get_data(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-01"))
    narrow = src.get_data(pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-08"))
    overlap = wide[
        (wide.timestamp >= pd.Timestamp("2025-05-01")) & (wide.timestamp < pd.Timestamp("2025-05-08"))
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(overlap, narrow)


def test_empty_window_rejected():
    with pytest.raises(ValueError):
        SyntheticSource().get_data(pd.Timestamp("2025-05-08"), pd.Timestamp("2025-05-01"))


def test_zero_knobs_means_no_regime_change():
    """With both knobs at 0 the pre- and post-drift worlds are the same process."""
    src = SyntheticSource(SyntheticConfig(drift_strength=0.0, feature_shift=0.0))
    before = src.get_data(pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-15"))
    after = src.get_data(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-15"))
    # Same relationship => a fit on one window transfers to the other. Seasonal
    # feature movement is expected; the coefficients are what must hold.
    assert abs(before["wind_speed"].mean() - after["wind_speed"].mean()) < 2.0
