"""Configuration objects and the column contract shared across the project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The data-layer contract: every implementation returns exactly these columns.
TIMESTAMP = "timestamp"
FEATURES = ["temperature", "wind_speed", "humidity"]
TARGET = "pm25"
COLUMNS = [TIMESTAMP, *FEATURES, TARGET]


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for the synthetic world.

    Two *independent* knobs, matching the two independent drift signals:

    - ``drift_strength`` shifts the coefficients of ``f`` in ``y = f(x) + noise``.
      The world's features look the same, but the learned relationship is wrong.
      This is concept drift -> it moves the *performance* signal.
    - ``feature_shift`` shifts the feature distributions themselves (stagnant,
      cooler, damper air after the drift date). This is covariate drift -> it
      moves the *data drift* (PSI) signal.

    Setting one to 0 and sweeping the other is how we prove the two detectors
    are actually independent (see ``scripts/sweep_knobs.py``).
    """

    origin: pd.Timestamp = pd.Timestamp("2025-01-01")
    horizon: pd.Timestamp = pd.Timestamp("2025-12-31")
    drift_date: pd.Timestamp = pd.Timestamp("2025-09-15")
    transition_days: float = 10.0
    drift_strength: float = 1.0
    feature_shift: float = 1.0
    noise_sigma: float = 3.0
    seed: int = 7


@dataclass(frozen=True)
class LoopConfig:
    """Windowing and decision thresholds for one scheduled run."""

    # Rolling window used to monitor the champion and to measure data drift.
    monitor_days: int = 14
    # How much recent history a challenger is trained on.
    challenger_train_days: int = 45
    # Most-recent slice, held out from the challenger so both models are judged
    # on data neither of them trained on.
    holdout_days: int = 7

    # Retrain trigger: champion RMSE on the monitor window vs. its RMSE at
    # training time. 1.25 == "the model got 25% worse".
    perf_drift_threshold: float = 1.25
    # PSI above this counts as a significant feature-distribution shift.
    # (Industry convention: <0.10 stable, 0.10-0.25 moderate, >0.25 significant.)
    psi_threshold: float = 0.25
    # A challenger must beat the champion by this fraction to be promoted.
    promotion_margin: float = 0.05

    experiment_name: str = "drift-loop-synthetic"
    registered_model_name: str = "pm25-ridge"


@dataclass(frozen=True)
class OpenMeteoConfig:
    """Location + span for the real-data source (Phase 2).

    Kraków sits in a basin and burns coal for winter heating, so PM2.5 is low and
    calm in summer and spikes under cold, still, inversion conditions once the
    heating season starts -- a genuine, narratable regime shift for a
    summer-trained model to decay through.

    The two Open-Meteo endpoints (weather archive + air quality) are fetched
    separately and joined on time; both offer hourly history for free.
    """

    name: str = "Kraków"
    latitude: float = 50.0647
    longitude: float = 19.9450
    origin: pd.Timestamp = pd.Timestamp("2025-05-01")
    horizon: pd.Timestamp = pd.Timestamp("2026-02-01")
    timezone: str = "GMT"


@dataclass(frozen=True)
class Profile:
    """A self-contained run target: which loop config, which MLflow backend file,
    and which run-metadata file the dashboard reads. Keeping Phase 1 and Phase 2
    in separate backend files lets each be reset and browsed independently."""

    key: str
    label: str
    loop: LoopConfig
    db_filename: str
    meta_filename: str


PROFILES: dict[str, Profile] = {
    "synthetic": Profile(
        key="synthetic",
        label="Synthetic (Phase 1)",
        loop=LoopConfig(),
        db_filename="mlflow.db",
        meta_filename="run_meta.json",
    ),
    "openmeteo": Profile(
        key="openmeteo",
        label="Open-Meteo · Kraków (Phase 2)",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo",
            registered_model_name="pm25-ridge-krakow",
        ),
        db_filename="mlflow_openmeteo.db",
        meta_filename="run_meta_openmeteo.json",
    ),
    # Phase 3: the live loop. Filled one cycle at a time by the scheduled job
    # (scripts/run_scheduled.py), against a backend that persists between runs.
    "scheduled": Profile(
        key="scheduled",
        label="Scheduled · live (Phase 3)",
        loop=LoopConfig(
            experiment_name="drift-loop-scheduled",
            registered_model_name="pm25-ridge-scheduled",
        ),
        db_filename="mlflow_scheduled.db",
        meta_filename="run_meta_scheduled.json",
    ),
}
