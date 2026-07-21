"""Real-data source: Open-Meteo weather + air quality (Phase 2).

Same ``get_data(start, end)`` contract as the synthetic source, so the loop,
drift detection, and dashboard don't change. Two things make this real-world:

1. **Two endpoints, joined on time.** Weather (temperature, wind, humidity) comes
   from the ERA5 archive API; PM2.5 comes from the air-quality API. They're
   fetched separately and inner-joined on the hourly timestamp.
2. **Fetch-once, slice-many.** The loop asks for many overlapping windows, so the
   whole configured span is fetched once and cached to disk (parquet). Every
   ``get_data`` then slices the cached frame -- no repeated API hits, and the
   data is stable across runs (the same determinism the synthetic source has).

Networking note: this machine sits behind a TLS-intercepting proxy, so Python's
default certifi bundle rejects the handshake. ``truststore`` routes verification
through the OS trust store (which has the proxy's root), matching what
``uv --system-certs`` needed at install time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from driftloop.config import COLUMNS, OpenMeteoConfig
from driftloop.data.base import validate_frame

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data_cache"

# Map Open-Meteo's variable names onto the project's column contract.
WEATHER_VARS = {
    "temperature_2m": "temperature",
    "wind_speed_10m": "wind_speed",
    "relative_humidity_2m": "humidity",
}

_TRUSTSTORE_INJECTED = False


def _ensure_truststore() -> None:
    """Route TLS verification through the OS trust store (once per process)."""
    global _TRUSTSTORE_INJECTED
    if not _TRUSTSTORE_INJECTED:
        import truststore

        truststore.inject_into_ssl()
        _TRUSTSTORE_INJECTED = True


def _get_json(url: str, params: dict) -> dict:
    import requests

    _ensure_truststore()
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _fetch_span(cfg: OpenMeteoConfig) -> pd.DataFrame:
    """Fetch and join the whole [origin, horizon] span from both endpoints."""
    date_params = {
        "latitude": cfg.latitude,
        "longitude": cfg.longitude,
        "start_date": cfg.origin.strftime("%Y-%m-%d"),
        "end_date": cfg.horizon.strftime("%Y-%m-%d"),
        "timezone": cfg.timezone,
    }

    weather = _get_json(
        WEATHER_URL,
        {**date_params, "hourly": ",".join(WEATHER_VARS), "wind_speed_unit": "ms"},
    )["hourly"]
    air = _get_json(AIR_QUALITY_URL, {**date_params, "hourly": "pm2_5"})["hourly"]

    weather_df = pd.DataFrame(weather).rename(columns={"time": "timestamp", **WEATHER_VARS})
    air_df = pd.DataFrame(air).rename(columns={"time": "timestamp", "pm2_5": "pm25"})
    for frame in (weather_df, air_df):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    merged = weather_df.merge(air_df, on="timestamp", how="inner")
    # Real feeds have gaps; the loop needs clean rows. Drop any hour missing a
    # feature or the target, then keep the contract's column order.
    merged = merged.dropna(subset=COLUMNS).sort_values("timestamp").reset_index(drop=True)
    return merged[COLUMNS]


class OpenMeteoSource:
    """Data source implementing the ``get_data`` contract from real observations."""

    def __init__(self, config: OpenMeteoConfig | None = None, cache_dir: Path | None = None) -> None:
        self.config = config or OpenMeteoConfig()
        self.cache_dir = cache_dir or CACHE_DIR
        self._timeline: pd.DataFrame | None = None

    def _cache_path(self) -> Path:
        cfg = self.config
        stem = (
            f"openmeteo_{cfg.latitude}_{cfg.longitude}_"
            f"{cfg.origin.date()}_{cfg.horizon.date()}"
        ).replace(".", "p")
        return self.cache_dir / f"{stem}.parquet"

    def timeline(self, refresh: bool = False) -> pd.DataFrame:
        """The full cached span, fetched from the API on first use."""
        if self._timeline is not None and not refresh:
            return self._timeline

        cache_path = self._cache_path()
        if cache_path.exists() and not refresh:
            self._timeline = pd.read_parquet(cache_path)
            return self._timeline

        df = _fetch_span(self.config)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        self._timeline = df
        return df

    def get_data(self, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        if window_end <= window_start:
            raise ValueError(f"empty window: [{window_start}, {window_end})")
        full = self.timeline()
        mask = (full["timestamp"] >= window_start) & (full["timestamp"] < window_end)
        window = full.loc[mask].copy()
        if window.empty:
            raise ValueError(
                f"no rows in [{window_start}, {window_end}) -- outside the fetched span "
                f"[{self.config.origin.date()}, {self.config.horizon.date()}]?"
            )
        return validate_frame(window)
