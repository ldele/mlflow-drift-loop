"""Synthetic data source with a controllable drift knob.

The whole timeline is generated once from a fixed seed and then sliced, so
``get_data(a, b)`` returns the same rows no matter what window you ask for.
That determinism matters: the loop asks for overlapping windows on every run
and the champion's training data must not silently change underneath it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from driftloop.config import COLUMNS, SyntheticConfig
from driftloop.data.base import validate_frame

# y = f(x) + noise. Before the drift date, f uses PRE_COEFS.
# Read it as: warm, still, damp air -> more PM2.5, wind is the big cleaner.
PRE_COEFS = {"intercept": 12.0, "temperature": 0.45, "wind_speed": -2.2, "humidity": 0.08}

# After the drift date f becomes PRE + drift_strength * DELTA_COEFS.
# The autumn-inversion regime: stagnation dominates, the temperature term flips.
DELTA_COEFS = {"intercept": 6.0, "temperature": -0.75, "wind_speed": -2.6, "humidity": 0.16}


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

    # Concept drift: the relationship changes.
    k = cfg.drift_strength * ramp
    pm25 = (
        (PRE_COEFS["intercept"] + k * DELTA_COEFS["intercept"])
        + (PRE_COEFS["temperature"] + k * DELTA_COEFS["temperature"]) * temperature
        + (PRE_COEFS["wind_speed"] + k * DELTA_COEFS["wind_speed"]) * wind_speed
        + (PRE_COEFS["humidity"] + k * DELTA_COEFS["humidity"]) * humidity
    )
    pm25 = np.clip(pm25 + rng.normal(0, cfg.noise_sigma, n), 0.5, None)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "humidity": humidity,
            "pm25": pm25,
        }
    )
    return df[COLUMNS]


class SyntheticSource:
    """Data source implementing the ``get_data`` contract."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()

    def get_data(self, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        if window_end <= window_start:
            raise ValueError(f"empty window: [{window_start}, {window_end})")
        full = _full_timeline(self.config)
        mask = (full["timestamp"] >= window_start) & (full["timestamp"] < window_end)
        return validate_frame(full.loc[mask].copy())
