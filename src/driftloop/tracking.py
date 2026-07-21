"""MLflow wiring: experiment setup, model registry, champion lookup.

Note on the backend: the plan said "local file store", but the MLflow **Model
Registry** is not supported by the file store -- it needs a database backend.
So we default to a local SQLite file (`mlflow.db`), which is still zero-setup
and single-file, and gives us the registry and its promotion history.

Note on stages: MLflow 3 deprecated `Staging`/`Production` stage transitions in
favour of **aliases**. We use the aliases `champion` and `challenger`, which is
also the vocabulary this project already speaks.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from driftloop.model import TrainedModel, effective_coefficients

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = "mlflow.db"


def _db_path(db_filename: str) -> Path:
    return REPO_ROOT / db_filename


def _artifact_dir(db_filename: str) -> Path:
    """Give each backend its own artifact directory so resetting one profile
    never touches another's artifacts. ``mlflow.db`` -> ``mlartifacts`` (the
    original); ``mlflow_openmeteo.db`` -> ``mlartifacts_openmeteo``."""
    stem = Path(db_filename).stem
    suffix = stem[len("mlflow"):].lstrip("_") if stem.startswith("mlflow") else stem
    return REPO_ROOT / (f"mlartifacts_{suffix}" if suffix else "mlartifacts")


def tracking_uri(db_filename: str = DEFAULT_DB) -> str:
    return f"sqlite:///{_db_path(db_filename).as_posix()}"


def setup(experiment_name: str, db_filename: str = DEFAULT_DB) -> MlflowClient:
    """Point MLflow at this profile's local backend and ensure the experiment."""
    mlflow.set_tracking_uri(tracking_uri(db_filename))
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name, artifact_location=_artifact_dir(db_filename).as_uri()
        )
    mlflow.set_experiment(experiment_name)
    return client


@dataclass
class ChampionRef:
    """The champion, plus everything needed to judge it fairly."""

    pipeline: Pipeline
    version: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    baseline_rmse: float
    run_id: str


def _version_tags(trained: TrainedModel) -> dict[str, str]:
    tags = {
        "train_start": trained.train_start.isoformat(),
        "train_end": trained.train_end.isoformat(),
        "baseline_rmse": f"{trained.baseline_rmse:.6f}",
        "n_rows": str(trained.n_rows),
    }
    # The learned coefficients (in original feature units) are the model's
    # fingerprint. Storing them per version turns concept drift into something
    # you can plot: watch them move as the champion is retrained.
    for name, value in effective_coefficients(trained.pipeline).items():
        tags[f"coef_{name}"] = f"{value:.6f}"
    return tags


def log_and_register(
    trained: TrainedModel,
    model_name: str,
    alias: str | None = None,
) -> str:
    """Log the sklearn pipeline to the active run and register a new version.

    Returns the registered version number.
    """
    # Pass pip_requirements explicitly: it skips MLflow's requirement-inference
    # subprocess, which is slow and (on some Windows Python builds) fragile.
    info = mlflow.sklearn.log_model(
        sk_model=trained.pipeline,
        name="model",
        pip_requirements=["scikit-learn", "pandas", "numpy", "scipy"],
    )
    registered = mlflow.register_model(info.model_uri, model_name)
    version = str(registered.version)  # MLflow may hand back an int; tags need str

    client = MlflowClient()
    for key, value in _version_tags(trained).items():
        client.set_model_version_tag(model_name, version, key, value)
    if alias:
        client.set_registered_model_alias(model_name, alias, version)
    return version


def promote(model_name: str, version: str) -> None:
    """Move the `champion` alias onto a version. This is the promotion event."""
    MlflowClient().set_registered_model_alias(model_name, CHAMPION_ALIAS, version)


def load_champion(model_name: str) -> ChampionRef | None:
    """Load the current champion from the registry, or None if there isn't one."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception:  # no registered model, or no champion alias yet
        return None

    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@{CHAMPION_ALIAS}")
    return ChampionRef(
        pipeline=pipeline,
        version=mv.version,
        train_start=pd.Timestamp(mv.tags["train_start"]),
        train_end=pd.Timestamp(mv.tags["train_end"]),
        baseline_rmse=float(mv.tags["baseline_rmse"]),
        run_id=mv.run_id,
    )


def reset(db_filename: str = DEFAULT_DB) -> None:
    """Wipe one profile's local backend so a rerun starts clean.

    MLflow only *soft*-deletes experiments and models through its API, which
    then blocks reusing the same name. For a local single-file backend the
    honest "fresh" is to remove the backing files -- call this *before*
    ``setup()``, at process start, while nothing holds the sqlite file open.
    """
    db_path = _db_path(db_filename)
    artifact_dir = _artifact_dir(db_filename)
    if db_path.exists():
        db_path.unlink()
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
