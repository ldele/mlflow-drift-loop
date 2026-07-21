"""The swappable data-layer interface.

One contract, two implementations (synthetic now, Open-Meteo in Phase 2)::

    get_data(window_start, window_end) -> DataFrame[timestamp, features..., target]

Windows are half-open: ``[start, end)``.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from driftloop.config import COLUMNS


class DataSource(Protocol):
    """Anything that can serve a time window of feature/target rows."""

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
