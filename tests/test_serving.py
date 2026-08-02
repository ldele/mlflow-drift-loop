"""Guards on the serving layer -- especially that it serves what the loop promoted.

The registry is relocated into a tmp_path for every test by patching
``tracking.REPO_ROOT``, so these run against a real MLflow backend and a real
registered version rather than a mocked one, and never touch the repo's own.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftloop import tracking
from driftloop.config import FEATURES, PROFILES
from driftloop.data import SyntheticSource
from driftloop.loop import bootstrap_champion
from driftloop.model import train
from driftloop.serving import create_app
from driftloop.tracking import CHAMPION_ALIAS, log_and_register, setup

PROFILE_KEY = "openmeteo"
TRAIN_START = pd.Timestamp("2025-04-01")
TRAIN_END = pd.Timestamp("2025-07-01")

OBSERVATION = {
    "timestamp": "2026-01-15T12:00:00",
    "temperature": -2.0,
    "wind_speed": 1.2,
    "humidity": 88.0,
    "precipitation": 0.0,
    "surface_pressure": 1028.0,
    "shortwave_radiation": 40.0,
}


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """Relocate the whole tracking layer into tmp_path and bootstrap a champion."""
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    profile = PROFILES[PROFILE_KEY]
    setup(profile.loop.experiment_name, profile.db_filename)
    bootstrap_champion(SyntheticSource(), TRAIN_START, TRAIN_END, profile.loop)
    return profile


@pytest.fixture()
def client(registry):
    # The context manager is what runs the lifespan, and the lifespan is what
    # loads the champion. Constructing TestClient without it serves nothing.
    with TestClient(create_app(PROFILE_KEY)) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_path, monkeypatch):
    """A service pointed at a registry where nothing has been promoted."""
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    with TestClient(create_app(PROFILE_KEY)) as c:
        yield c


def test_health_reports_the_loaded_champion(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "1"
    assert body["profile"] == PROFILE_KEY


def test_model_card_names_the_version_and_its_training_window(client):
    body = client.get("/model").json()
    assert body["model_version"] == "1"
    assert body["alias"] == CHAMPION_ALIAS
    assert body["features"] == FEATURES
    assert pd.Timestamp(body["trained_from"]) == TRAIN_START
    assert body["baseline_rmse"] > 0
    # The staleness number is the point of the endpoint: this champion was
    # trained in 2025 and is being served well after.
    assert body["training_age_days"] > 0


def test_predict_returns_one_prediction_per_observation_in_order(client):
    hours = ["2026-01-15T00:00:00", "2026-01-15T06:00:00", "2026-01-15T18:00:00"]
    body = {"observations": [{**OBSERVATION, "timestamp": h} for h in hours]}
    payload = client.post("/predict", json=body).json()

    assert payload["model_version"] == "1"
    assert [p["timestamp"] for p in payload["predictions"]] == hours
    # Same weather, different hours -> the diurnal terms must move the answer.
    assert len({p["pm25_raw"] for p in payload["predictions"]}) == 3


def test_served_pm25_is_never_negative_but_the_raw_output_is_kept(client):
    payload = client.post("/predict", json={"observations": [OBSERVATION]}).json()
    prediction = payload["predictions"][0]
    assert prediction["pm25"] >= 0.0
    assert prediction["pm25"] == max(0.0, prediction["pm25_raw"])


def test_the_hour_is_read_in_utc_not_as_written(client):
    """A tz-aware timestamp must be converted before the diurnal encoding.

    12:00Z and 14:00+02:00 are the same instant. If serving read the hour off
    the string it would encode 12 for one and 14 for the other, putting the
    features two hours out of phase with what the model was trained on.
    """
    utc = client.post(
        "/predict", json={"observations": [{**OBSERVATION, "timestamp": "2026-01-15T12:00:00Z"}]}
    ).json()
    offset = client.post(
        "/predict",
        json={"observations": [{**OBSERVATION, "timestamp": "2026-01-15T14:00:00+02:00"}]},
    ).json()

    assert utc["predictions"][0]["pm25_raw"] == offset["predictions"][0]["pm25_raw"]


def test_prediction_matches_the_pipeline_called_directly(client, registry):
    """Serving must not quietly transform anything on the way through.

    The cyclical encoding is rebuilt here from the formula rather than by
    calling the shared helper, so this would still catch serving swapping sin
    for cos or reading the wrong hour.
    """
    payload = client.post("/predict", json={"observations": [OBSERVATION]}).json()

    champion = tracking.load_champion(registry.loop.registered_model_name)
    frame = pd.DataFrame([{k: v for k, v in OBSERVATION.items() if k != "timestamp"}])
    radians = 2 * np.pi * pd.Timestamp(OBSERVATION["timestamp"]).hour / 24.0
    frame["hour_sin"] = np.sin(radians)
    frame["hour_cos"] = np.cos(radians)
    expected = float(champion.pipeline.predict(frame[FEATURES])[0])

    assert payload["predictions"][0]["pm25_raw"] == pytest.approx(expected)


def test_impossible_humidity_is_rejected_before_it_reaches_the_model(client):
    body = {"observations": [{**OBSERVATION, "humidity": 140.0}]}
    assert client.post("/predict", json=body).status_code == 422


def test_an_empty_batch_is_rejected(client):
    assert client.post("/predict", json={"observations": []}).status_code == 422


def test_a_service_with_nothing_promoted_starts_and_reports_itself_unready(empty_client):
    """It must not crash-loop: an empty registry is a state to report, not a fault."""
    health = empty_client.get("/health").json()
    assert health["model_loaded"] is False
    assert health["status"] == "no_model"

    assert empty_client.post("/predict", json={"observations": [OBSERVATION]}).status_code == 503
    assert empty_client.get("/model").status_code == 503


def test_reload_picks_up_a_promotion_without_a_restart(client, registry):
    """The seam between the weekly loop and serving, asserted directly."""
    assert client.get("/health").json()["model_version"] == "1"

    # Promote a second version behind the running service's back, exactly as a
    # cycle that cleared the gate would.
    setup(registry.loop.experiment_name, registry.db_filename)
    later = train(SyntheticSource().get_data(pd.Timestamp("2025-08-01"), pd.Timestamp("2025-11-01")))
    with mlflow.start_run(run_name="test-promotion"):
        version = log_and_register(later, registry.loop.registered_model_name, alias=CHAMPION_ALIAS)
    assert version == "2"

    # Still serving the old one until asked -- serving does not poll.
    assert client.get("/health").json()["model_version"] == "1"

    body = client.post("/reload").json()
    assert body == {"previous_version": "1", "current_version": "2", "changed": True}
    assert client.get("/health").json()["model_version"] == "2"

    # A second reload finds nothing new.
    assert client.post("/reload").json()["changed"] is False
