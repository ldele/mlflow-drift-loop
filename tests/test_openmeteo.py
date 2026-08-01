"""Open-Meteo source tests. No network: the two endpoint responses are mocked,
so we exercise the join, NaN handling, caching, and the column contract.

The source has two modes. At ``forecast_lead_days=0`` the features come from the
ERA5 analysis for the target hour. Above 0 they come from the archived forecast
run issued that many days earlier, which arrives under ``_previous_dayN`` names
and has to be mapped back onto the contract. Both are covered, because the
renaming is exactly the kind of thing that breaks silently and leaves the model
training on the wrong columns."""

import pandas as pd
import pytest

from driftloop.config import COLUMNS, OpenMeteoConfig
from driftloop.data import openmeteo
from driftloop.data.openmeteo import OpenMeteoSource


def _fake_responses(lead_days: int):
    """Six hourly rows. The air-quality feed is missing hour 2 (NaN pm2_5) and
    lacks hour 5 entirely, so the join + dropna should yield four clean rows.

    Built from WEATHER_VARS rather than a hand-written list, so adding a feature
    to the contract does not fail these tests for the wrong reason."""
    times = [f"2025-06-01T0{h}:00" for h in range(6)]
    suffix = f"_previous_day{lead_days}" if lead_days > 0 else ""
    weather = {
        "hourly": {
            "time": times,
            **{
                f"{api}{suffix}": [10.0 + i + 0.5 * h for h in range(6)]
                for i, api in enumerate(openmeteo.WEATHER_VARS)
            },
        }
    }
    air = {
        "hourly": {
            "time": times[:5],  # no hour 5 row at all -> inner join drops it
            "pm2_5": [20.0, 21.0, None, 23.0, 24.0],  # hour 2 is NaN -> dropped
        }
    }
    return weather, air


def _install(monkeypatch, lead_days: int) -> list[tuple[str, dict]]:
    """Patch the HTTP layer and hand back the calls it received."""
    weather, air = _fake_responses(lead_days)
    calls: list[tuple[str, dict]] = []

    def fake_get_json(url, params):
        calls.append((url, params))
        return air if "air-quality" in url else weather

    monkeypatch.setattr(openmeteo, "_get_json", fake_get_json)
    return calls


@pytest.fixture()
def mocked_source(tmp_path, monkeypatch):
    """The shipped configuration: a seven-day-ahead forecast."""
    _install(monkeypatch, OpenMeteoConfig().forecast_lead_days)
    return OpenMeteoSource(OpenMeteoConfig(), cache_dir=tmp_path)


def test_join_drops_missing_target_rows_and_keeps_contract(mocked_source):
    df = mocked_source.timeline()
    assert list(df.columns) == COLUMNS
    # 6 weather rows, air missing hour 5 (join) and hour 2 (NaN) -> 4 clean rows.
    assert len(df) == 4
    assert df["timestamp"].is_monotonic_increasing
    assert not df[COLUMNS].isna().any().any()
    # Every API name must land on its own contract column. Checking all of them
    # rather than spot-checking one: a rename map that crossed or dropped a pair
    # would still pass a single assertion, and the model would train on the
    # wrong column with no error anywhere.
    for i, column in enumerate(openmeteo.WEATHER_VARS.values()):
        assert df[column].iloc[0] == 10.0 + i, f"{column} got the wrong series"
    assert df["pm25"].iloc[0] == 20.0


def test_forecast_mode_requests_the_previous_run_and_renames_it(tmp_path, monkeypatch):
    """At lead N the features must come from the run issued N days earlier."""
    calls = _install(monkeypatch, 7)
    cfg = OpenMeteoConfig(forecast_lead_days=7)
    df = OpenMeteoSource(cfg, cache_dir=tmp_path).timeline()

    weather_call = next(c for c in calls if "air-quality" not in c[0])
    url, params = weather_call
    assert "historical-forecast-api" in url
    requested = params["hourly"].split(",")
    # Every fetched feature must carry the lead suffix; a bare name here would
    # mean that feature was silently taken from the analysis instead.
    assert requested == [f"{api}_previous_day7" for api in openmeteo.WEATHER_VARS]
    # The suffixed names must not survive into the frame the model trains on.
    assert list(df.columns) == COLUMNS


def test_nowcast_mode_reads_the_analysis_instead(tmp_path, monkeypatch):
    calls = _install(monkeypatch, 0)
    cfg = OpenMeteoConfig(forecast_lead_days=0)
    df = OpenMeteoSource(cfg, cache_dir=tmp_path).timeline()

    url, params = next(c for c in calls if "air-quality" not in c[0])
    assert "archive-api" in url
    assert "previous_day" not in params["hourly"]
    assert list(df.columns) == COLUMNS


def test_lead_is_part_of_the_cache_identity(tmp_path):
    """Same place, same span, different lead -> different data, different file."""
    a = OpenMeteoSource(OpenMeteoConfig(forecast_lead_days=0), cache_dir=tmp_path)
    b = OpenMeteoSource(OpenMeteoConfig(forecast_lead_days=7), cache_dir=tmp_path)
    assert a._cache_path() != b._cache_path()


def test_lead_beyond_the_archive_is_rejected(tmp_path, monkeypatch):
    _install(monkeypatch, openmeteo.MAX_LEAD_DAYS + 1)
    cfg = OpenMeteoConfig(forecast_lead_days=openmeteo.MAX_LEAD_DAYS + 1)
    with pytest.raises(ValueError, match="previous-run archive"):
        OpenMeteoSource(cfg, cache_dir=tmp_path).timeline()


def test_timeline_is_cached_to_disk_and_not_refetched(tmp_path, monkeypatch):
    calls = _install(monkeypatch, OpenMeteoConfig().forecast_lead_days)
    source = OpenMeteoSource(OpenMeteoConfig(), cache_dir=tmp_path)

    source.timeline()  # two endpoint calls
    assert len(calls) == 2
    assert source._cache_path().exists()

    # A fresh instance reads the parquet cache instead of hitting the API.
    fresh = OpenMeteoSource(OpenMeteoConfig(), cache_dir=tmp_path)
    fresh.timeline()
    assert len(calls) == 2  # unchanged


def test_get_data_slices_the_window(mocked_source):
    window = mocked_source.get_data(
        pd.Timestamp("2025-06-01T01:00"), pd.Timestamp("2025-06-01T04:00")
    )
    # half-open [01:00, 04:00): hours 1 and 3 survive (hour 2 was NaN-dropped).
    assert list(window["timestamp"].dt.hour) == [1, 3]


def test_empty_window_rejected(mocked_source):
    with pytest.raises(ValueError):
        mocked_source.get_data(pd.Timestamp("2025-06-01T05:00"), pd.Timestamp("2025-06-01T05:00"))
