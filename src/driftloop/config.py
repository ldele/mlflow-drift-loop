"""Configuration objects and the column contract shared across the project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The data-layer contract: every implementation returns exactly these columns.
TIMESTAMP = "timestamp"

# Observed weather, and the only things that can drift: covariate drift here
# means these distributions moved. Chosen for the physics of how pollution
# accumulates and clears rather than for what is easy to fetch --
# shortwave_radiation drives the daytime convective mixing that dilutes PM2.5,
# surface_pressure catches the subsidence inversions that trap it, and
# precipitation scavenges it out of the air.
#
# Boundary layer height would be the single best feature here and is not in the
# list, because Open-Meteo does not archive previous model runs for it. At a
# seven-day lead it comes back null, so it cannot be used without abandoning the
# forecast framing.
DRIFT_FEATURES = [
    "temperature",
    "wind_speed",
    "humidity",
    "precipitation",
    "surface_pressure",
    "shortwave_radiation",
]

# The clock, on a circle so hour 23 sits next to hour 0. Derived from the
# timestamp rather than observed, so it cannot drift: every monitoring window
# contains all 24 hours, and a PSI on it would be flat forever. That is why it
# is excluded from DRIFT_FEATURES and present in FEATURES -- the diurnal cycle
# is most of what the climatology baseline was beating the model with.
CYCLICAL_FEATURES = ["hour_sin", "hour_cos"]

FEATURES = [*DRIFT_FEATURES, *CYCLICAL_FEATURES]
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
    # How much recent history a challenger is trained on. 180, not 45: PM2.5 is
    # a seasonal process, and a challenger trained on six weeks only ever sees
    # one season. It then serves into the next one and is worse than the model
    # it replaced -- which is exactly what a full annual replay exposed, and
    # what a replay stopping at the winter peak had been hiding.
    challenger_train_days: int = 180
    # Most-recent slice, held out from the challenger so both models are judged
    # on data neither of them trained on.
    holdout_days: int = 7

    # Retrain trigger: champion RMSE on the monitor window vs. its RMSE at
    # training time. 1.25 == "the model got 25% worse".
    #
    # This one ratchets, and it is the documented failure the project is built
    # around. The denominator is the champion's *own* baseline, so every
    # promotion resets it, and promotions happen at the seasonal peak -- so each
    # new champion inherits a higher bar than the model it replaced and the bar
    # never comes back down. Kraków spends its last 30 of 48 runs unable to fire
    # at any error whatsoever. Kept, because it is still the cheap first check;
    # the floor below is what stops it going permanently deaf.
    perf_drift_threshold: float = 1.25
    # The second trigger, and the one that does not move when a model is
    # promoted: skill against a 30-day hour-of-day profile of recent pollution.
    # -0.5 reads as "the champion is now 50% worse than doing nothing clever",
    # which is a statement about the model rather than about its own history.
    #
    # Scale-free on purpose. The other candidate fix was an absolute RMSE floor,
    # and it cannot be built: to wake Los Angeles, whose deaf stretch tops out at
    # 18 µg/m³, the floor has to sit below 18 -- where Delhi fires on 100% of its
    # runs and Johannesburg on 80%. There is no value in between, so an
    # "absolute" floor is a per-city tuning knob wearing a disguise, and the
    # cities stop being comparable. A ratio against a yardstick that holds still
    # is the same idea without that defect.
    #
    # None disables it, which is the pre-2026-08 behaviour.
    skill_floor: float | None = None
    # PSI above this counts as a significant feature-distribution shift.
    # (Industry convention: <0.10 stable, 0.10-0.25 moderate, >0.25 significant.)
    psi_threshold: float = 0.25
    # A challenger must beat the champion by this fraction to be promoted.
    promotion_margin: float = 0.05

    experiment_name: str = "drift-loop-synthetic"
    registered_model_name: str = "pm25-ridge"


# How far ahead the model predicts. At 7 the weather features are the forecast
# for the target hour *as it was issued seven days earlier*, so the model is a
# real forecaster and inherits the weather forecast's own error. At 0 it reads
# analysed weather for the target hour instead, which makes it a same-hour
# estimate rather than a forecast.
#
# Open-Meteo publishes previous model runs only out to day 7, so 7 is the
# ceiling, not an arbitrary choice.
FORECAST_LEAD_DAYS = 7


@dataclass(frozen=True)
class OpenMeteoConfig:
    """Location, span, and forecast lead for the real-data source.

    Kraków sits in a basin and burns coal for winter heating, so PM2.5 is low and
    calm in summer and spikes under cold, still, inversion conditions once the
    heating season starts -- a genuine, narratable regime shift for a
    summer-trained model to decay through.

    Two Open-Meteo endpoints are fetched separately and joined on time. The
    target is always *observed* PM2.5 from the air-quality API. The features come
    from the archived forecast runs when ``forecast_lead_days`` is set, and from
    the ERA5 analysis when it is 0.
    """

    name: str = "Kraków"
    country: str = "PL"
    latitude: float = 50.0647
    longitude: float = 19.9450
    origin: pd.Timestamp = pd.Timestamp("2025-05-01")
    # Runs to within a fortnight of the present, like every other city. It used
    # to stop at 2026-02-01, just past the winter peak, which left the page's
    # European city half a year out of date next to the rest.
    horizon: pd.Timestamp = pd.Timestamp("2026-07-15")
    timezone: str = "GMT"
    forecast_lead_days: int = FORECAST_LEAD_DAYS


# The monitored locations. Adding a city means adding a config here and a Profile
# below that carries it; everything downstream follows -- the site's map plots one
# marker per profile that has a `location`.
#
# Each span is chosen to contain a clean training season *and* the season that
# spoils it, because that is the regime shift the loop exists to catch.
KRAKOW = OpenMeteoConfig()

# Monsoon rain scrubs the air to a September minimum, then crop-residue burning
# and winter inversions take PM2.5 from ~42 to ~127 ug/m3. The most violent of
# the six by a wide margin.
DELHI = OpenMeteoConfig(
    name="Delhi",
    country="IN",
    latitude=28.6139,
    longitude=77.2090,
    origin=pd.Timestamp("2025-05-01"),
    horizon=pd.Timestamp("2026-07-15"),
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

# --- Southern hemisphere -----------------------------------------------------
#
# The three above all peak in *northern* winter, which leaves open the question
# of whether the loop's thresholds quietly encode a calendar. These three peak in
# June-August, when Krakow and Delhi are at their cleanest, and are run on
# identical settings. Each was chosen on the measurements, not on reputation:
# eighteen candidate cities were fetched and ranked by their PM2.5 swing first.

# Krakow's twin, six months out of phase: a coastal basin that traps winter
# inversions. The widest swing of every city measured -- ~18 ug/m3 in December
# to ~94 in June, a 5.1x walk -- which makes it the strongest evidence that the
# detection is reading the world and not the date.
SANTIAGO = OpenMeteoConfig(
    name="Santiago",
    country="CL",
    latitude=-33.4489,
    longitude=-70.6693,
    origin=pd.Timestamp("2025-10-15"),
    horizon=pd.Timestamp("2026-07-15"),
)

# Highveld winter: cold, still nights trap domestic coal and wood smoke under an
# inversion. ~23 ug/m3 in January to ~84 in July, a 3.6x swing.
JOHANNESBURG = OpenMeteoConfig(
    name="Johannesburg",
    country="ZA",
    latitude=-26.2041,
    longitude=28.0473,
    origin=pd.Timestamp("2025-11-01"),
    horizon=pd.Timestamp("2026-07-15"),
)

# Winter wood-heater smoke, and the Los Angeles lesson repeating: Sydney is the
# reputationally obvious Australian city and its PM2.5 is flat (1.5x). Melbourne
# swings 3.1x. The catch is that it does so between ~5 and ~15 ug/m3, so the
# drift is real in ratio while the air stays close to the WHO guideline
# throughout -- a clean city whose model still goes stale.
MELBOURNE = OpenMeteoConfig(
    name="Melbourne",
    country="AU",
    latitude=-37.8136,
    longitude=144.9631,
    origin=pd.Timestamp("2025-09-01"),
    horizon=pd.Timestamp("2026-07-15"),
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
        # Train on clean summer air, walk into the heating season, then keep
        # going out the other side: a full annual cycle, so the loop is watched
        # through the world getting worse *and* recovering.
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-06-01"),
            champion_train_end=pd.Timestamp("2025-08-01"),
            first_run=pd.Timestamp("2025-08-15"),
            last_run=pd.Timestamp("2026-07-10"),
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
        # Train on the monsoon-scrubbed minimum, walk into the burning season,
        # then on through the following monsoon as the air clears again.
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-07-15"),
            champion_train_end=pd.Timestamp("2025-09-30"),
            first_run=pd.Timestamp("2025-10-10"),
            last_run=pd.Timestamp("2026-07-10"),
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
    "openmeteo_santiago": Profile(
        key="openmeteo_santiago",
        label="Santiago",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo-santiago",
            registered_model_name="pm25-ridge-santiago",
        ),
        db_filename="mlflow_openmeteo_santiago.db",
        meta_filename="run_meta_openmeteo_santiago.json",
        location=SANTIAGO,
        # Train on the southern summer minimum, replay into the winter inversion.
        # The same shape as Krakow's replay, on the opposite half of the year.
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-11-15"),
            champion_train_end=pd.Timestamp("2026-02-01"),
            first_run=pd.Timestamp("2026-02-10"),
            last_run=pd.Timestamp("2026-07-10"),
        ),
    ),
    "openmeteo_joburg": Profile(
        key="openmeteo_joburg",
        label="Johannesburg",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo-joburg",
            registered_model_name="pm25-ridge-joburg",
        ),
        db_filename="mlflow_openmeteo_joburg.db",
        meta_filename="run_meta_openmeteo_joburg.json",
        location=JOHANNESBURG,
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-12-01"),
            champion_train_end=pd.Timestamp("2026-02-15"),
            first_run=pd.Timestamp("2026-02-25"),
            last_run=pd.Timestamp("2026-07-10"),
        ),
    ),
    "openmeteo_melbourne": Profile(
        key="openmeteo_melbourne",
        label="Melbourne",
        loop=LoopConfig(
            experiment_name="drift-loop-openmeteo-melbourne",
            registered_model_name="pm25-ridge-melbourne",
        ),
        db_filename="mlflow_openmeteo_melbourne.db",
        meta_filename="run_meta_openmeteo_melbourne.json",
        location=MELBOURNE,
        # A longer replay than the other two: the swing is real but narrow, so the
        # loop is watched over ~30 weekly runs rather than ~20.
        replay=ReplayWindows(
            champion_train_start=pd.Timestamp("2025-09-15"),
            champion_train_end=pd.Timestamp("2025-12-01"),
            first_run=pd.Timestamp("2025-12-10"),
            last_run=pd.Timestamp("2026-07-10"),
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
    "santiago": "openmeteo_santiago",
    "joburg": "openmeteo_joburg",
    "melbourne": "openmeteo_melbourne",
}
