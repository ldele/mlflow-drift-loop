"""The swappable data-layer interface.

One contract, two implementations (synthetic now, Open-Meteo in Phase 2)::

    get_data(window_start, window_end) -> DataFrame[timestamp, features..., target]

Windows are half-open: ``[start, end)``.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

import numpy as np

from driftloop.config import COLUMNS, TIMESTAMP


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour-of-day as a point on a circle.

    Raw hour 0-23 is useless to a linear model: it would make 23:00 and 00:00
    the furthest apart values when they are adjacent. Sine and cosine together
    place each hour on a circle, so midnight is next to 23:00 and the model can
    express a smooth daily cycle with two coefficients.

    Shared by every source, so the encoding cannot drift apart between them.
    """
    hours = pd.to_datetime(df[TIMESTAMP]).dt.hour.to_numpy(dtype=float)
    radians = 2 * np.pi * hours / 24.0
    df["hour_sin"] = np.sin(radians)
    df["hour_cos"] = np.cos(radians)
    return df


class DataSource(Protocol):
    """Anything that can serve a time window of feature/target rows."""

    # How far ahead the features look. Part of the identity of the data rather
    # than of the loop reading it: at lead 7 the features are the forecast issued
    # a week before the target hour, at lead 0 they are the analysis for it, and
    # the two are different datasets over the same span. Anything applying a
    # causality rule -- "the baseline may only average what was observable when
    # the forecast went out" -- has to ask the source, not a config it was handed.
    forecast_lead_days: int

    def get_data(self, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        ...


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if an implementation breaks the column contract."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"data source is missing required columns: {missing}")
    if not df[COLUMNS[0]].is_monotonic_increasing:
        raise ValueError("data source must return rows sorted by timestamp")
    return df[COLUMNS].reset_index(drop=True)
