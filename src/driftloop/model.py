"""The model itself: deliberately boring.

A Ridge on six weather variables and the hour of day. A simple model decays
*legibly* when the relationship shifts, which is what the demo is about: a big
model would absorb some of the drift and blur the story.

It was three features and grew to eight, which is why anything counting them
reads the list from `config.FEATURES` rather than hardcoding a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from driftloop.config import FEATURES, TARGET

# The two model kinds. Ridge ships; gbm exists only for the ablation that asks
# whether this project's findings belong to the world or to a linear model.
RIDGE = "ridge"
GBM = "gbm"


@dataclass
class TrainedModel:
    pipeline: Pipeline
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_rows: int
    # RMSE on a chronological tail the model did not fit. This is the number
    # performance drift is measured against later.
    baseline_rmse: float


def build_pipeline(alpha: float = 1.0, kind: str = RIDGE) -> Pipeline:
    """The champion, or the gradient-boosted stand-in used to interrogate it.

    ``gbm`` exists for one experiment and is not a candidate for shipping. Every
    result in this repository is compatible with a duller explanation than the
    one it offers: that a near-linear model misspecifies seasonal structure, and
    what looks like "the world moved so retraining pays" is refitting papering
    over that. Separating the two needs a challenger flexible enough to absorb
    the nonlinearity, which is what this is for. See ``scripts/ablate_model.py``.

    No scaler on the tree: it splits on thresholds, so monotone rescaling of a
    feature changes nothing it can learn, and leaving the scaler in would imply
    otherwise to anyone reading the pipeline.
    """
    if kind == GBM:
        return Pipeline([("gbm", HistGradientBoostingRegressor(random_state=0))])
    if kind != RIDGE:
        raise ValueError(f"unknown model kind: {kind!r}")
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def is_linear(pipeline: Pipeline) -> bool:
    """Whether this pipeline can be serialised as slopes and an intercept.

    The registry stores linear models as nine numbers, and ``retrospect`` scores
    any version from those tags without unpickling anything. A tree cannot be
    written down that way, so the caller has to know which it is holding rather
    than discovering it through an AttributeError three layers down.
    """
    return "ridge" in pipeline.named_steps


def rmse(model: Pipeline, df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    pred = model.predict(df[FEATURES])
    return float(np.sqrt(np.mean((df[TARGET].to_numpy() - pred) ** 2)))


def error_metrics(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    """RMSE plus the two other numbers every regression report shows: MAE and R^2."""
    if df.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    actual = df[TARGET].to_numpy()
    pred = model.predict(df[FEATURES])
    return {
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "mae": float(mean_absolute_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
        "n": int(len(df)),
    }


def predictions_frame(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """timestamp | actual | predicted | residual for a window -- the raw material
    of the predicted-vs-actual and residual panels."""
    pred = model.predict(df[FEATURES])
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "actual": df[TARGET].to_numpy(),
            "predicted": pred,
            "residual": df[TARGET].to_numpy() - pred,
        }
    )


def effective_coefficients(pipeline: Pipeline) -> dict[str, float]:
    """The linear model's coefficients expressed in the *original* feature units.

    The pipeline standardises features before the Ridge, so ``ridge.coef_`` is in
    z-score units and isn't directly readable. Folding the scaler back in gives
    the slope per real unit (per degree, per m/s, per %RH) -- which is what makes
    the concept-drift story legible: watch these move across model versions.
    """
    scaler: StandardScaler = pipeline.named_steps["scale"]
    ridge: Ridge = pipeline.named_steps["ridge"]
    coefs = ridge.coef_ / scaler.scale_
    intercept = float(ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_))
    out = {feature: float(c) for feature, c in zip(FEATURES, coefs)}
    out["intercept"] = intercept
    return out


def train(
    df: pd.DataFrame,
    alpha: float = 1.0,
    val_fraction: float = 0.2,
    kind: str = RIDGE,
) -> TrainedModel:
    """Fit on a training window and record an honest baseline RMSE.

    The baseline comes from a chronological tail split (not a random split):
    the model is scored on data that comes *after* what it fit, which is the
    same shape as how it will be scored in production.
    """
    if len(df) < 50:
        raise ValueError(f"not enough rows to train: {len(df)}")

    split = int(len(df) * (1 - val_fraction))
    fit_df, val_df = df.iloc[:split], df.iloc[split:]

    warmup = build_pipeline(alpha, kind).fit(fit_df[FEATURES], fit_df[TARGET])
    baseline = rmse(warmup, val_df)

    # Refit on the full window so the deployed model uses all available data.
    final = build_pipeline(alpha, kind).fit(df[FEATURES], df[TARGET])

    return TrainedModel(
        pipeline=final,
        train_start=df["timestamp"].iloc[0],
        train_end=df["timestamp"].iloc[-1],
        n_rows=len(df),
        baseline_rmse=baseline,
    )
