"""Serve the promoted champion behind an HTTP API.

The loop's whole output is an alias: ``champion`` points at whichever registered
version last cleared the promotion gate. This reads that alias and answers
requests with it, which is the step that turns a registry entry into something
a consumer can actually call.

Three properties are worth stating, because they are the ones a reviewer asks
about:

- **The alias is the contract, not a version number.** Nothing here pins a
  version. Promote in the registry and ``POST /reload`` picks the new one up
  without a redeploy -- that is the seam between the weekly loop and serving.
- **Serving never writes to the tracking store.** It sets the tracking URI and
  reads; it does not call ``setup()``, which would create an experiment as a
  side effect. A read-only consumer should leave no trace.
- **The hour encoding is not reimplemented here.** ``add_cyclical_features`` is
  the same function the training path uses, so the serving features cannot
  drift away from the trained ones.

Run it with ``uvicorn driftloop.serving:app``, or ``python scripts/serve.py``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from driftloop.config import (
    CITY_CLI_NAMES,
    FEATURES,
    FORECAST_LEAD_DAYS,
    PROFILES,
    TARGET,
    TIMESTAMP,
)
from driftloop.data.base import add_cyclical_features
from driftloop.tracking import CHAMPION_ALIAS, ChampionRef, load_champion, tracking_uri

# Which profile's registry to serve. Every city keeps its own backend file and
# its own registered model, so serving is per-city too -- one container per
# city, selected by environment variable.
PROFILE_ENV_VAR = "DRIFTLOOP_PROFILE"
DEFAULT_PROFILE = "openmeteo"

# Bound the batch so one request cannot pull the process over on memory.
MAX_BATCH = 10_000

# pydantic v2 reserves the `model_` prefix for its own config. These fields are
# about the ML model, which is the honest name for them, so the namespace guard
# is switched off rather than the fields renamed.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


def _utc_now() -> pd.Timestamp:
    """Naive UTC, matching the timestamps the data layer stores."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


@dataclass
class ServedChampion:
    """The loaded champion plus when this process picked it up."""

    ref: ChampionRef
    profile_key: str
    loaded_at: pd.Timestamp


class Observation(BaseModel):
    """One hour of forecast weather, for the hour being predicted.

    The units are Open-Meteo's, because that is what the champion was trained
    on: degrees Celsius, metres per second, percent relative humidity,
    millimetres, hectopascals, watts per square metre.

    ``timestamp`` is the hour being predicted, not the hour the forecast was
    issued. At the shipped lead of seven days these features are the forecast
    for that hour as it stood a week earlier -- see ``FORECAST_LEAD_DAYS``. A
    naive timestamp is read as UTC, which is what the training data is in.
    """

    timestamp: datetime
    temperature: float = Field(description="2 m air temperature, degC")
    wind_speed: float = Field(ge=0, description="10 m wind speed, m/s")
    humidity: float = Field(ge=0, le=100, description="2 m relative humidity, %")
    precipitation: float = Field(ge=0, description="total precipitation, mm")
    surface_pressure: float = Field(gt=0, description="surface pressure, hPa")
    shortwave_radiation: float = Field(ge=0, description="shortwave radiation, W/m2")


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    """One predicted hour.

    Two numbers rather than one, deliberately. The champion is an unconstrained
    Ridge, so it can and does emit negative concentrations on clean hours; a
    negative PM2.5 is not a thing that exists, so ``pm25`` is floored at zero
    for consumers. ``pm25_raw`` is what the model actually said, kept because
    silently clamping a model's output is how you lose track of its behaviour.
    """

    timestamp: datetime
    pm25: float
    pm25_raw: float


class PredictResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    model_name: str
    model_version: str
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    """The champion's identity card: which version is answering, and how stale."""

    model_config = _ALLOW_MODEL_PREFIX

    profile: str
    city: str | None
    model_name: str
    model_version: str
    alias: str
    run_id: str
    trained_from: datetime
    trained_to: datetime
    training_age_days: float
    baseline_rmse: float
    features: list[str]
    target: str
    forecast_lead_days: int
    loaded_at: datetime


class ReloadResponse(BaseModel):
    """What re-reading the alias found."""

    previous_version: str | None
    current_version: str
    changed: bool


class HealthResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    status: str
    profile: str
    model_loaded: bool
    model_version: str | None


def load_served_champion(profile_key: str) -> ServedChampion | None:
    """Read the ``champion`` alias for one profile, or None if nothing is promoted.

    Points MLflow at that profile's backend file and reads. Returning None
    rather than raising is what lets the service start against an empty
    registry: a container that crash-loops because no model is promoted yet is
    harder to operate than one that reports itself unready.
    """
    profile = PROFILES[profile_key]
    mlflow.set_tracking_uri(tracking_uri(profile.db_filename))
    ref = load_champion(profile.loop.registered_model_name)
    if ref is None:
        return None
    return ServedChampion(ref=ref, profile_key=profile_key, loaded_at=_utc_now())


def _require_champion(request: Request) -> ServedChampion:
    served: ServedChampion | None = request.app.state.champion
    if served is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"no version carries the '{CHAMPION_ALIAS}' alias in profile "
                f"'{request.app.state.profile_key}' -- bootstrap or promote one, then POST /reload"
            ),
        )
    return served


def _model_info(served: ServedChampion) -> ModelInfo:
    profile = PROFILES[served.profile_key]
    ref = served.ref
    age = (_utc_now() - ref.train_end) / pd.Timedelta(1, unit="D")
    return ModelInfo(
        profile=served.profile_key,
        city=profile.location.name if profile.location else None,
        model_name=profile.loop.registered_model_name,
        model_version=ref.version,
        alias=CHAMPION_ALIAS,
        run_id=ref.run_id,
        trained_from=ref.train_start.to_pydatetime(),
        trained_to=ref.train_end.to_pydatetime(),
        training_age_days=round(float(age), 2),
        baseline_rmse=ref.baseline_rmse,
        features=list(FEATURES),
        target=TARGET,
        forecast_lead_days=FORECAST_LEAD_DAYS,
        loaded_at=served.loaded_at.to_pydatetime(),
    )


def _feature_frame(observations: list[Observation]) -> pd.DataFrame:
    """Turn the request body into the exact feature matrix the champion expects.

    Timestamps are normalised to naive UTC first. The training data is GMT, so
    an aware timestamp from another zone has to be converted before the hour is
    read off it -- otherwise the diurnal encoding would be hours out of phase
    with what the model learned.
    """
    df = pd.DataFrame([o.model_dump() for o in observations])
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True).dt.tz_localize(None)
    add_cyclical_features(df)
    return df


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load the champion once at startup instead of once per request."""
    app.state.champion = load_served_champion(app.state.profile_key)
    yield


def resolve_profile_key(name: str) -> str:
    """Accept either a profile key (``openmeteo``) or a city name (``krakow``).

    The run scripts speak city names and the profile registry speaks keys. The
    container has to name the city twice -- once to build the registry, once to
    serve it -- so accepting both is what stops those two drifting apart into a
    service that starts up pointed at an empty backend.
    """
    if name in PROFILES:
        return name
    if name in CITY_CLI_NAMES:
        return CITY_CLI_NAMES[name]
    raise ValueError(
        f"unknown profile '{name}' -- profiles {sorted(PROFILES)}, "
        f"or cities {sorted(CITY_CLI_NAMES)}"
    )


def create_app(profile_key: str | None = None) -> FastAPI:
    """Build the service for one profile's registry."""
    profile_key = resolve_profile_key(
        profile_key or os.environ.get(PROFILE_ENV_VAR, DEFAULT_PROFILE)
    )

    app = FastAPI(
        title="Air quality drift watch -- champion serving",
        description=(
            "Serves whichever model version currently holds the `champion` alias "
            "in the MLflow registry. Predicts a city's hourly PM2.5 "
            f"{FORECAST_LEAD_DAYS} days ahead from the weather forecast for that hour."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )
    app.state.profile_key = profile_key
    app.state.champion = None

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness plus readiness. 200 always; `model_loaded` carries the truth.

        Kept at 200 even with nothing promoted so a probe can distinguish "the
        process is up but has no model" from "the process is down".
        """
        served: ServedChampion | None = request.app.state.champion
        return HealthResponse(
            status="ok" if served else "no_model",
            profile=request.app.state.profile_key,
            model_loaded=served is not None,
            model_version=served.ref.version if served else None,
        )

    @app.get("/model", response_model=ModelInfo)
    def model(request: Request) -> ModelInfo:
        """Which version is answering, what it was trained on, and how stale it is."""
        return _model_info(_require_champion(request))

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, body: PredictRequest) -> PredictResponse:
        """Predict PM2.5 for a batch of forecast hours."""
        served = _require_champion(request)
        df = _feature_frame(body.observations)
        raw = served.ref.pipeline.predict(df[FEATURES])
        return PredictResponse(
            model_name=PROFILES[served.profile_key].loop.registered_model_name,
            model_version=served.ref.version,
            predictions=[
                Prediction(
                    timestamp=ts.to_pydatetime(),
                    pm25=max(0.0, float(value)),
                    pm25_raw=float(value),
                )
                for ts, value in zip(df[TIMESTAMP], raw)
            ],
        )

    @app.post("/reload", response_model=ReloadResponse)
    def reload(request: Request) -> ReloadResponse:
        """Re-read the alias, picking up a promotion without a restart.

        This is the hook the weekly loop needs: it promotes in the registry,
        then something calls this. Serving does not poll, so a version stays put
        until asked -- deliberate, because an unannounced model swap under live
        traffic is worse than a stale one.
        """
        previous = request.app.state.champion
        served = load_served_champion(request.app.state.profile_key)
        if served is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"no version carries the '{CHAMPION_ALIAS}' alias -- nothing to load",
            )
        request.app.state.champion = served
        previous_version = previous.ref.version if previous else None
        return ReloadResponse(
            previous_version=previous_version,
            current_version=served.ref.version,
            changed=previous_version != served.ref.version,
        )

    return app


app = create_app()
