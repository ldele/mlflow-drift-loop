"""The gate of the forecast-miss split.

Fixed by docs/registrations/2026-11-23-why-a-forecast-was-wrong.md, so a change
here is a change to a registered design and should fail loudly. No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]

import split_error as se  # noqa: E402

from driftloop.config import OpenMeteoConfig  # noqa: E402
from driftloop.data import OpenMeteoSource  # noqa: E402


def test_the_registered_thresholds() -> None:
    assert se.MIN_COVERAGE == 0.95
    assert se.MIN_CITIES == 12


def test_coverage_is_the_share_of_week_ahead_hours_with_a_same_day_row() -> None:
    hours = pd.date_range("2026-01-01", periods=20, freq="h")
    ahead = pd.DataFrame({"timestamp": hours})
    same_day = pd.DataFrame({"timestamp": hours[:19].append(pd.DatetimeIndex(["2027-01-01"]))})
    # 19 of the 20 week-ahead hours; the extra same-day hour counts for nothing.
    assert se.coverage(ahead, same_day) == 19 / 20
    assert se.coverage(ahead.iloc[:0], same_day) == 0.0


def test_the_same_day_source_keeps_place_and_span_and_reads_the_forecast_archive(tmp_path) -> None:
    week_ahead = OpenMeteoSource(OpenMeteoConfig(forecast_lead_days=7), cache_dir=tmp_path)
    same_day = se.day0_source(week_ahead)
    cfg = same_day.config
    assert cfg.forecast_lead_days == 0 and cfg.day0_from_forecast_archive
    assert (cfg.latitude, cfg.longitude, cfg.origin, cfg.horizon) == (
        week_ahead.config.latitude, week_ahead.config.longitude,
        week_ahead.config.origin, week_ahead.config.horizon)
    assert same_day.cache_dir == tmp_path
    assert same_day._cache_path().name.endswith("_lead0d_forecast.parquet")
