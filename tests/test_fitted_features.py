"""A fitted model is scored on the columns it was fit on, whatever `FEATURES` says now.

`b1a15b8` widened `config.FEATURES` from three columns to eight on 2026-08-01. The scheduled
champion had been fit on the three, and every helper that scored a fitted model handed it all
eight, so scikit-learn refused: "The feature names should match those that were passed during
fit." The weekly Action failed on that every Monday from 2026-08-03 to 2026-09-07.
`model.fitted_features` reads the model's own `feature_names_in_`, so a change to the feature
list can no longer strand a model the registry still serves.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from driftloop.config import FEATURES, TARGET
from driftloop.model import (
    effective_coefficients,
    error_metrics,
    fitted_features,
    predictions_frame,
    rmse,
    squared_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BEFORE_WIDENING = ["temperature", "wind_speed", "humidity"]


def _frame(n: int = 48, seed: int = 0) -> pd.DataFrame:
    """A window carrying every column today's `FEATURES` names, the target and the clock."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df[TARGET] = 10 + 2 * df["temperature"] - df["wind_speed"] + rng.normal(scale=0.1, size=n)
    df["timestamp"] = pd.date_range("2026-08-01", periods=n, freq="h")
    return df


def _fit(columns: list[str]) -> Pipeline:
    """The shipped shape: a scaler named `scale` in front of a Ridge named `ridge`."""
    df = _frame()
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]).fit(df[columns], df[TARGET])


def test_a_model_fit_before_the_widening_scores_todays_frame() -> None:
    """The failure itself, through every scoring helper at once."""
    model = _fit(BEFORE_WIDENING)
    df = _frame(seed=1)
    expected = model.predict(df[BEFORE_WIDENING])
    assert np.isfinite(error_metrics(model, df)["rmse"])
    assert np.isfinite(rmse(model, df))
    assert len(squared_errors(model, df)) == len(df)
    assert np.allclose(predictions_frame(model, df)["predicted"], expected)


def test_a_model_names_its_own_columns_in_the_order_it_was_fit() -> None:
    assert fitted_features(_fit(BEFORE_WIDENING)) == BEFORE_WIDENING
    assert fitted_features(_fit(FEATURES)) == FEATURES


def test_an_estimator_fit_on_a_bare_array_falls_back_to_features() -> None:
    df = _frame()
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    model.fit(df[FEATURES].to_numpy(), df[TARGET])
    assert fitted_features(model) == FEATURES


def test_coefficients_are_labelled_by_the_models_own_columns() -> None:
    """`zip(FEATURES, coefs)` labelled the old champion's three slopes correctly only because
    the widening appended; a reordering would have mislabelled them without a sound."""
    coefs = effective_coefficients(_fit(BEFORE_WIDENING))
    assert list(coefs) == [*BEFORE_WIDENING, "intercept"]


def _committed_champion_dir() -> Path | None:
    """The scheduled champion's artifact directory, found through the committed registry.

    Read repository-relative rather than through `models:/...`, whose URIs the CI runner wrote
    as its own absolute paths.
    """
    db = REPO_ROOT / "mlflow_scheduled.db"
    if not db.is_file():
        return None
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute(
            "select mv.source from registered_model_aliases a join model_versions mv "
            "on mv.name = a.name and mv.version = a.version where a.alias = 'champion'"
        ).fetchone()
    finally:
        con.close()
    if row is None or not str(row[0]).startswith("models:/"):
        return None
    model_id = str(row[0]).removeprefix("models:/")
    path = REPO_ROOT / "mlartifacts_scheduled" / "models" / model_id / "artifacts"
    return path if path.is_dir() else None


def test_the_committed_scheduled_champion_scores_a_window_with_todays_features() -> None:
    """The test that would have failed on 2026-08-01, rather than six Mondays after it."""
    path = _committed_champion_dir()
    if path is None:
        pytest.skip("no committed scheduled champion in this checkout")
    import mlflow.sklearn

    model = mlflow.sklearn.load_model(path.as_posix())
    metrics = error_metrics(model, _frame(seed=2))
    assert np.isfinite(metrics["rmse"])
    assert metrics["n"] == 48
