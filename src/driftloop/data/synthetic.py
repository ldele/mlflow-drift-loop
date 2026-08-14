"""Synthetic data source with a controllable drift knob.

The whole timeline is generated once from a fixed seed and then sliced, so
``get_data(a, b)`` returns the same rows for a given window every time. The loop
requests overlapping windows on every run, and the champion's training data must
not change underneath it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from driftloop.config import COLUMNS, SyntheticConfig
from driftloop.data.base import add_cyclical_features, validate_frame

# y = f(x) + noise. Before the drift date, f uses PRE_COEFS.
# Read it as: warm, still, damp air -> more PM2.5, wind is the big cleaner,
# rain washes it out, and strong sun mixes the boundary layer and dilutes it.
PRE_COEFS = {
    "intercept": 12.0,
    "temperature": 0.45,
    "wind_speed": -2.2,
    "humidity": 0.08,
    "precipitation": -3.0,
    "shortwave_radiation": -0.010,
}

# After the drift date f becomes PRE + drift_strength * DELTA_COEFS.
# The autumn-inversion regime: stagnation dominates, the temperature term flips,
# and weak winter sun stops doing the mixing it used to.
DELTA_COEFS = {
    "intercept": 6.0,
    "temperature": -0.75,
    "wind_speed": -2.6,
    "humidity": 0.16,
    "precipitation": -1.0,
    "shortwave_radiation": 0.008,
}

# surface_pressure is generated but carries no coefficient: a feature the world
# supplies and the target does not depend on. Kept so that drift with no
# consequence for the model is representable.


@lru_cache(maxsize=8)
def _full_timeline(cfg: SyntheticConfig) -> pd.DataFrame:
    idx = pd.date_range(cfg.origin, cfg.horizon, freq="h")
    n = len(idx)
    rng = np.random.default_rng(cfg.seed)

    doy = idx.dayofyear.to_numpy(dtype=float)
    hour = idx.hour.to_numpy(dtype=float)
    season = np.sin(2 * np.pi * (doy - 100) / 365.25)  # peaks in mid-summer
    diurnal = np.sin(2 * np.pi * (hour - 15) / 24)  # peaks mid-afternoon

    temperature = 12.0 + 10.0 * season + 4.0 * diurnal + rng.normal(0, 1.5, n)
    wind_speed = 3.0 + 1.8 * np.sin(2 * np.pi * (doy - 30) / 365.25) + rng.gamma(2.0, 0.9, n) - 1.8
    humidity = 62.0 - 18.0 * season - 0.9 * diurnal + rng.normal(0, 6.0, n)

    # Ramp from 0 -> 1 over `transition_days` starting at `drift_date`.
    days_since = (idx - cfg.drift_date) / pd.Timedelta(1, unit="D")
    ramp = np.clip(days_since.to_numpy(dtype=float) / max(cfg.transition_days, 1e-9), 0.0, 1.0)

    # Covariate drift: the world itself changes (stagnant, cooler, damper).
    temperature = temperature - 2.0 * cfg.feature_shift * ramp
    wind_speed = wind_speed - 1.6 * cfg.feature_shift * ramp
    humidity = humidity + 8.0 * cfg.feature_shift * ramp

    wind_speed = np.clip(wind_speed, 0.2, None)
    humidity = np.clip(humidity, 10.0, 100.0)

    # Sun follows the day and the season, and is zero at night. This is the
    # stand-in for boundary-layer mixing.
    daylight = np.clip(np.sin(2 * np.pi * (hour - 6) / 24), 0.0, None)
    shortwave_radiation = 780.0 * daylight * (0.45 + 0.55 * (season + 1) / 2)
    shortwave_radiation = np.clip(shortwave_radiation + rng.normal(0, 40.0, n), 0.0, None)

    # Rain is mostly zero with occasional wet hours.
    precipitation = rng.gamma(0.6, 0.9, n) * (rng.random(n) < 0.12)

    # Synoptic pressure wandering around a standard sea-level value.
    surface_pressure = 1013.0 + 9.0 * np.sin(2 * np.pi * (doy - 15) / 90.0) + rng.normal(0, 4.0, n)

    # Covariate drift also reaches the two features that carry a coefficient,
    # so feature_shift and drift_strength stay separable rather than one
    # sneaking in through the other.
    shortwave_radiation = np.clip(shortwave_radiation - 90.0 * cfg.feature_shift * ramp, 0.0, None)

    # Concept drift: the relationship changes.
    k = cfg.drift_strength * ramp

    def coef(name: str) -> np.ndarray:
        return PRE_COEFS[name] + k * DELTA_COEFS[name]

    pm25 = (
        coef("intercept")
        + coef("temperature") * temperature
        + coef("wind_speed") * wind_speed
        + coef("humidity") * humidity
        + coef("precipitation") * precipitation
        + coef("shortwave_radiation") * shortwave_radiation
    )
    pm25 = np.clip(pm25 + rng.normal(0, cfg.noise_sigma, n), 0.5, None)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "humidity": humidity,
            "precipitation": precipitation,
            "surface_pressure": surface_pressure,
            "shortwave_radiation": shortwave_radiation,
            "pm25": pm25,
        }
    )
    return add_cyclical_features(df)[COLUMNS]


class SyntheticSource:
    """Data source implementing the ``get_data`` contract."""

    # The synthetic world has no forecast in it: features and target are drawn
    # for the same hour, which is a lead of zero.
    forecast_lead_days = 0

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()

    def get_data(self, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        if window_end <= window_start:
            raise ValueError(f"empty window: [{window_start}, {window_end})")
        full = _full_timeline(self.config)
        mask = (full["timestamp"] >= window_start) & (full["timestamp"] < window_end)
        return validate_frame(full.loc[mask].copy())
