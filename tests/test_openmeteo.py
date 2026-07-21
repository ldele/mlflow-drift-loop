"""Open-Meteo source tests. No network: the two endpoint responses are mocked,
so we exercise the join, NaN handling, caching, and the column contract."""

import pandas as pd
import pytest

from driftloop.config import COLUMNS, OpenMeteoConfig
from driftloop.data import openmeteo
from driftloop.data.openmeteo import OpenMeteoSource


def _fake_responses():
    """Six hourly rows. The air-quality feed is missing hour 2 (NaN pm2_5) and
    lacks hour 5 entirely, so the join + dropna should yield four clean rows."""
    times = [f"2025-06-01T0{h}:00" for h in range(6)]
    weather = {
        "hourly": {
            "time": times,
            "temperature_2m": [15.0, 15.5, 16.0, 16.5, 17.0, 17.5],
            "wind_speed_10m": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "relative_humidity_2m": [60, 61, 62, 63, 64, 65],
        }
    }
    air = {
        "hourly": {
            "time": times[:5],  # no hour 5 row at all -> inner join drops it
            "pm2_5": [20.0, 21.0, None, 23.0, 24.0],  # hour 2 is NaN -> dropped
        }
    }
    return weather, air


@pytest.fixture()
def mocked_source(tmp_path, monkeypatch):
    weather, air = _fake_responses()

    def fake_get_json(url, params):
        return weather if "archive" in url else air

    monkeypatch.setattr(openmeteo, "_get_json", fake_get_json)
    return OpenMeteoSource(OpenMeteoConfig(), cache_dir=tmp_path)


def test_join_drops_missing_target_rows_and_keeps_contract(mocked_source):
    df = mocked_source.timeline()
    assert list(df.columns) == COLUMNS
    # 6 weather rows, air missing hour 5 (join) and hour 2 (NaN) -> 4 clean rows.
    assert len(df) == 4
    assert df["timestamp"].is_monotonic_increasing
    assert not df[COLUMNS].isna().any().any()
    # spot-check the mapping landed on the right columns
    assert df["temperature"].iloc[0] == 15.0
    assert df["pm25"].iloc[0] == 20.0


def test_timeline_is_cached_to_disk_and_not_refetched(tmp_path, monkeypatch):
    weather, air = _fake_responses()
    calls = {"n": 0}

    def counting_get_json(url, params):
        calls["n"] += 1
        return weather if "archive" in url else air

    monkeypatch.setattr(openmeteo, "_get_json", counting_get_json)
    source = OpenMeteoSource(OpenMeteoConfig(), cache_dir=tmp_path)

    source.timeline()  # two endpoint calls
    assert calls["n"] == 2
    assert source._cache_path().exists()

    # A fresh instance reads the parquet cache instead of hitting the API.
    fresh = OpenMeteoSource(OpenMeteoConfig(), cache_dir=tmp_path)
    fresh.timeline()
    assert calls["n"] == 2  # unchanged


def test_get_data_slices_the_window(mocked_source):
    window = mocked_source.get_data(
        pd.Timestamp("2025-06-01T01:00"), pd.Timestamp("2025-06-01T04:00")
    )
    # half-open [01:00, 04:00): hours 1 and 3 survive (hour 2 was NaN-dropped).
    assert list(window["timestamp"].dt.hour) == [1, 3]


def test_empty_window_rejected(mocked_source):
    with pytest.raises(ValueError):
        mocked_source.get_data(pd.Timestamp("2025-06-01T05:00"), pd.Timestamp("2025-06-01T05:00"))
