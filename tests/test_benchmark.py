"""Baseline fairness at a forecast horizon.

The autoregressive baselines are the only predictors allowed to see past PM2.5,
which makes them the place a leak would hide. A forecaster issuing at T for
T+lead may use observations up to T and no later, so persistence has to repeat
the reading from issue time rather than the one just before the target. Getting
this wrong would hand the baseline a week of hindsight and make the model look
worse than it is -- silently, and in the direction that flatters nobody."""

import numpy as np
import pandas as pd

from driftloop.benchmark import autoregressive_lags, detail_map, predictor_columns
from driftloop.config import FEATURES, TARGET


def _timeline(hours: int = 500) -> pd.DataFrame:
    stamps = pd.date_range("2025-06-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            **{f: rng.normal(size=hours) for f in FEATURES},
            # A ramp, so a lagged column is trivially distinguishable from the
            # actual: value == row index.
            TARGET: np.arange(hours, dtype=float),
        }
    )


def test_nowcast_lags_are_one_hour_and_one_day():
    assert autoregressive_lags(0) == (1, 24)


def test_forecast_lags_step_back_to_issue_time():
    persistence, seasonal = autoregressive_lags(7)
    assert persistence == 7 * 24
    # Seasonal naive must not collapse onto persistence at a whole-day lead,
    # or the table shows the same column twice under two names.
    assert seasonal != persistence
    assert seasonal == 7 * 24 + 24


def test_persistence_never_sees_past_issue_time():
    """The value used for a target row must be the one from `lead` hours before."""
    lead_days = 7
    timeline = _timeline()
    columns = predictor_columns(timeline, timeline.iloc[:200], lead_days=lead_days)

    lead_hours = lead_days * 24
    actual = columns["actual"].to_numpy(dtype=float)
    persistence = columns["persistence"].to_numpy(dtype=float)

    observed = ~np.isnan(persistence)
    # TARGET is the row index, so the gap is exactly the lag in rows.
    assert np.all(actual[observed] - persistence[observed] == lead_hours)
    # And the first `lead_hours` rows have no issue-time observation at all.
    assert np.all(np.isnan(persistence[:lead_hours]))


def test_details_describe_the_horizon_they_were_scored_at():
    assert "issued" in detail_map(7)["persistence"]
    assert "issued" not in detail_map(0)["persistence"]
