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
    """Location + span for the real-data source.

    Kraków sits in a basin and burns coal for winter heating, so PM2.5 is low and
    calm in summer and spikes under cold, still, inversion conditions once the
    heating season starts -- a genuine, narratable regime shift for a
    summer-trained model to decay through.

    The two Open-Meteo endpoints (weather archive + air quality) are fetched
    separately and joined on time; both offer hourly history for free.
    """

    name: str = "Kraków"
    country: str = "PL"
    latitude: float = 50.0647
    longitude: float = 19.9450
    origin: pd.Timestamp = pd.Timestamp("2025-05-01")
    horizon: pd.Timestamp = pd.Timestamp("2026-02-01")
    timezone: str = "GMT"


# The monitored locations. Adding a city means adding a config here and a Profile
# below that carries it; everything downstream follows -- the site's map plots one
# marker per profile that has a `location`.
#
# Each span is chosen to contain a clean training season *and* the season that
# spoils it, because that is the regime shift the loop exists to catch.
KRAKOW = OpenMeteoConfig()

# Monsoon rain scrubs the air to a September minimum, then crop-residue burning
# and winter inversions take PM2.5 from ~42 to ~127 ug/m3. The most violent of
# the three by a wide margin.
DELHI = OpenMeteoConfig(
    name="Delhi",
    country="IN",
    latitude=28.6139,
    longitude=77.2090,
    origin=pd.Timestamp("2025-05-01"),
    horizon=pd.Timestamp("2026-02-01"),
)

# The counterexample, and it is one on the measurements rather than by design.
# LA is popularly a summer-smog city, but hourly PM2.5 over 2025-26 peaks in
# November-December (~29 ug/m3) and bottoms out in June (~15) -- mildly *winter*
# -bad, and only a ~1.9x swing against Delhi's 3x and Krakow's 5x. Its span
# covers a whole annual cycle so the loop is watched through the world drifting
# up and back down again, and mostly declining to act.
LOS_ANGELES = OpenMeteoConfig(
    name="Los Angeles",
    country="US",
    latitude=34.0522,
    longitude=-118.2437,
    origin=pd.Timestamp("2025-09-01"),
    horizon=pd.Timestamp("2026-07-20"),
)


@dataclass(frozen=True)
class ReplayWindows:
    """Which slice bootstraps the champion, and the weekly replay that follows it.

    Per city, because the seasons don't line up: Krakow and Delhi train on clean
    summer air and walk into winter, while Los Angeles has no season worth the
    name and is replayed across a full year instead.
    """

    champion_train_start: pd.Timestamp
    champion_train_end: pd.Timestamp
    first_run: pd.Timestamp
    last_run: pd.Timestamp
    step_days: int = 7


@dataclass(frozen=True)
class Profile:
    """A self-contained run target: which loop config, which MLflow backend file,
    and which run-metadata file the dashboard reads. Each data source gets its own
    backend file so they can be reset and browsed independently."""

    key: str
    label: str
    loop: LoopConfig
    db_filename: str
    meta_filename: str
    # Where this profile's data physically comes from, if anywhere. Drives the
    # published map: a profile with a location gets a marker, one without doesn't.
    # `scheduled` is deliberately None -- it reads the same Kraków source as
    # `openmeteo`, so giving it a location would stack two markers on one point.
    # It is a *mode* (the loop running live), not a separate place.
    location: OpenMeteoConfig | None = None
    # How to replay this city's history. Set for the historical city profiles;
    # None for the synthetic world and for the live loop, which advances one
    # real cycle at a time rather than replaying a fixed span.
    replay: ReplayWindows | None = None


# Ordered lead-with-the-real-data first; the dashboard sidebar and the published
# site both follow this order.
PROFILES: dict[str, Profile] = {
    "openmeteo": Profile(
        key="openmeteo",
        label="Kraków",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo",
            registered_model_name="pm25-ridge-krakow",
        ),
        db_filename="mlflow_openmeteo.db",
        meta_filename="run_meta_openmeteo.json",
        location=KRAKOW,
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-06-01"),
            champion_train_end=pd.Timestamp("2025-08-01"),
            first_run=pd.Timestamp("2025-08-15"),
            last_run=pd.Timestamp("2026-01-20"),
        ),
    ),
    "openmeteo_delhi": Profile(
        key="openmeteo_delhi",
        label="Delhi",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo-delhi",
            registered_model_name="pm25-ridge-delhi",
        ),
        db_filename="mlflow_openmeteo_delhi.db",
        meta_filename="run_meta_openmeteo_delhi.json",
        location=DELHI,
        # Train on the monsoon-scrubbed minimum, walk into the burning season.
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-07-15"),
            champion_train_end=pd.Timestamp("2025-09-30"),
            first_run=pd.Timestamp("2025-10-10"),
            last_run=pd.Timestamp("2026-01-25"),
        ),
    ),
    "openmeteo_la": Profile(
        key="openmeteo_la",
        label="Los Angeles",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo-la",
            registered_model_name="pm25-ridge-la",
        ),
        db_filename="mlflow_openmeteo_la.db",
        meta_filename="run_meta_openmeteo_la.json",
        location=LOS_ANGELES,
        # A full annual cycle rather than one season: the world drifts up into
        # the mild winter peak and back down again.
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-09-01"),
            champion_train_end=pd.Timestamp("2025-10-25"),
            first_run=pd.Timestamp("2025-11-05"),
            last_run=pd.Timestamp("2026-07-15"),
        ),
    ),
    "synthetic": Profile(
        key="synthetic",
        label="Synthetic",
        loop=LoopConfig(),
        db_filename="mlflow.db",
        meta_filename="run_meta.json",
    ),
    # The live loop. Filled one cycle at a time by the scheduled job
    # (scripts/run_scheduled.py), against a backend that persists between runs.
    # (see CITY_CLI_NAMES below for the short names the run script accepts)
    "scheduled": Profile(
        key="scheduled",
        label="Live schedule",
        loop=LoopConfig(
            experiment_name="drift-loop-scheduled",
            registered_model_name="pm25-ridge-scheduled",
        ),
        db_filename="mlflow_scheduled.db",
        meta_filename="run_meta_scheduled.json",
    ),
}

# Short CLI names for the city profiles -> profile key, in dashboard order. One
# source of truth so `run_openmeteo.py --city` and the Streamlit "generate it
# now" button can't drift apart.
CITY_CLI_NAMES: dict[str, str] = {
    "krakow": "openmeteo",
    "delhi": "openmeteo_delhi",
    "la": "openmeteo_la",
}
