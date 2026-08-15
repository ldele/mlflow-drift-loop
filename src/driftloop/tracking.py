"""MLflow wiring: experiment setup, model registry, champion lookup.

The backend is a local SQLite file rather than a file store, because the file
store does not support the Model Registry and the registry is what carries the
promotion history.

Promotion moves an alias, not a stage. MLflow 3 deprecated the
`Staging`/`Production` transitions in favour of aliases, and `champion` /
`challenger` are the terms the rest of the project already uses.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from driftloop.model import TrainedModel, effective_coefficients, is_linear

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
    # When this version last sat a holdout exam and won, or None for a version
    # that has never sat one. Written by `mark_certified`.
    last_certified: pd.Timestamp | None = None
    # Probation state, written by `mark_promoted` and `mark_probation_cleared`.
    # `promoted_at` dates the promotion and `replaced_version` names the model
    # displaced, which together are everything needed to re-run that decision
    # later against the alternative it beat. `probation_cleared` stops the check
    # repeating every run for the rest of the version's life.
    promoted_at: pd.Timestamp | None = None
    replaced_version: str | None = None
    probation_cleared: bool = False

    def certified_at(self) -> pd.Timestamp:
        """The date this version's certificate runs from.

        Falls back to the end of its training data, which is the honest answer
        for a version that has never been examined: a bootstrap champion's
        knowledge stops there, and nothing has tested it since.
        """
        return self.last_certified if self.last_certified is not None else self.train_end


def _version_tags(trained: TrainedModel) -> dict[str, str]:
    tags = {
        "train_start": trained.train_start.isoformat(),
        "train_end": trained.train_end.isoformat(),
        "baseline_rmse": f"{trained.baseline_rmse:.6f}",
        "n_rows": str(trained.n_rows),
    }
    # Coefficients in original feature units, stored per version: this is what
    # makes concept drift plottable and what lets `retrospect` score a version
    # without unpickling it.
    #
    # Only a linear model has them. A tree is recovered from its logged artifact
    # instead (`retrospect.registered_models`); writing a placeholder here would
    # make an unscoreable version look scoreable.
    if is_linear(trained.pipeline):
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


def mark_promoted(model_name: str, version: str, as_of: pd.Timestamp, replaced: str) -> None:
    """Record when a version was promoted and which version it displaced.

    Kept on the version because probation has to re-run that decision weeks
    later, against the same alternative, and the run log cannot be trusted to
    name it: the run that promotes overwrites its own `champion_version` tag
    with the winner, so the loser is not recoverable from the run alone.
    """
    client = MlflowClient()
    client.set_model_version_tag(model_name, version, "promoted_at", as_of.isoformat())
    client.set_model_version_tag(model_name, version, "replaced_version", str(replaced))


def mark_probation_cleared(model_name: str, version: str) -> None:
    """Record that a version's probation has been judged, whichever way.

    Set on a pass and on a failure. A rolled-back version keeps the tag so that
    being promoted again later does not put it back on probation for a decision
    already taken.
    """
    MlflowClient().set_model_version_tag(model_name, version, "probation_cleared", "true")


def load_version(model_name: str, version: str) -> Pipeline:
    """Load one registered version's pipeline by number rather than by alias."""
    return mlflow.sklearn.load_model(f"models:/{model_name}/{version}")


def mark_certified(model_name: str, version: str, as_of: pd.Timestamp) -> None:
    """Record that a version sat a holdout exam on ``as_of`` and kept its place.

    The counterpart to `promote` for the model that was not replaced. Kept on
    the version rather than derived from the run log so that reading it costs
    one registry lookup the loop already makes, and so a champion carries its
    own certificate wherever the registry goes.
    """
    MlflowClient().set_model_version_tag(
        model_name, version, "last_certified", as_of.isoformat()
    )


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
        # str, as in log_and_register: MLflow returns the version as an int
        # here and as a str elsewhere, and the two get compared downstream.
        version=str(mv.version),
        train_start=pd.Timestamp(mv.tags["train_start"]),
        train_end=pd.Timestamp(mv.tags["train_end"]),
        baseline_rmse=float(mv.tags["baseline_rmse"]),
        run_id=mv.run_id,
        last_certified=(
            pd.Timestamp(mv.tags["last_certified"]) if "last_certified" in mv.tags else None
        ),
        promoted_at=(
            pd.Timestamp(mv.tags["promoted_at"]) if "promoted_at" in mv.tags else None
        ),
        replaced_version=mv.tags.get("replaced_version"),
        probation_cleared=mv.tags.get("probation_cleared") == "true",
    )


def reset(db_filename: str = DEFAULT_DB) -> None:
    """Wipe one profile's local backend so a rerun starts clean.

    MLflow only soft-deletes experiments and models through its API, which then
    blocks reusing the same name, so this removes the backing files instead.
    Call it before ``setup()``, at process start, while nothing holds the
    sqlite file open.
    """
    db_path = _db_path(db_filename)
    artifact_dir = _artifact_dir(db_filename)
    if db_path.exists():
        db_path.unlink()
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
