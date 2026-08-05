"""Is the model worth having, and is its one hyper-parameter set sensibly?

Two questions the loop never answers about itself:

1. **Does it beat anything?** A champion RMSE means nothing on its own. Here it
   is scored against four predictors that need no training at all.
2. **Why alpha=1.0?** It was a default. This tunes it with a forward-chaining
   split and reports the whole curve, so the choice is visible.

Everything is scored **per monitoring window** -- the same 14-day slices the
loop reports on -- so three things sit in one column of numbers:

* the champion the loop actually served at that point, having retrained,
* the same champion frozen at its first version, never retrained,
* and the baselines.

That is what makes the comparison mean something. Scoring a frozen champion
across the whole replay span would measure the counterfactual, not the product.

One caveat is stated rather than buried: persistence and seasonal-naive use
*past PM2.5*, which the Ridge never sees -- it predicts from weather alone. So
they are not like-for-like on features. They are the right question anyway: is a
weather model better than repeating the last reading? Where it is not, publish
that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from driftloop.config import FEATURES, TARGET
from driftloop.model import build_pipeline

# Swept on a log grid: alpha's effect is multiplicative, so linear steps would
# spend most of their samples at the insensitive end.
ALPHA_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)

# Predictors that get to look at past values of the target. The Ridge does not.
USES_PAST_TARGET = {"persistence", "seasonal_naive"}

DETAIL = {
    "champion_served": "the model the loop served, retrained as needed",
    "champion_frozen": "the first champion, never retrained",
    "pooled_cities": "one model trained on all six cities at once, never retrained",
    "persistence": "repeat the previous observation",
    "seasonal_naive": "repeat the same hour yesterday",
    "climatology": "the training mean for that hour of day",
    "train_mean": "the training-window mean, forever",
}


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


@dataclass
class Scored:
    name: str
    detail: str
    median_rmse: float
    windows: int
    uses_past_target: bool


def autoregressive_lags(lead_days: int) -> tuple[int, int]:
    """Positional lags for the two baselines that are allowed to see past PM2.5.

    A forecaster issuing at T for T+lead may use observations up to T and no
    later. So persistence repeats the reading from *issue time* rather than the
    one an hour before the target -- the difference between a fair baseline and
    one handed the answer. Seasonal naive steps back a further day, because at a
    whole-day lead it would otherwise land on exactly the persistence row and
    stop being a separate predictor.
    """
    if lead_days <= 0:
        return 1, 24
    lead_hours = lead_days * 24
    return lead_hours, lead_hours + 24


def detail_map(lead_days: int) -> dict[str, str]:
    """Predictor descriptions, which depend on whether this is a forecast."""
    if lead_days <= 0:
        return DETAIL
    return {
        **DETAIL,
        "champion_frozen": "the first champion, never retrained",
        "persistence": "repeat the last reading available when the forecast was issued",
        "seasonal_naive": "the same hour, the day before the forecast was issued",
    }


def fit_pooled(training: dict[str, pd.DataFrame], alpha: float = 1.0) -> tuple[Pipeline, list[str]]:
    """One model over every city at once, with a per-city intercept.

    Worth having as a baseline because it tests the premise of the whole layout.
    Six cities each get their own model here; if one model over all of them did
    just as well, that choice would be wrong.

    The per-city intercept is not a detail. Mean PM2.5 runs from 7 µg/m³ in
    Melbourne to 84 in Delhi, a spread of 11×, and a single shared intercept
    cannot straddle it: the same model without the city columns scores 43% worse
    on average. With them, the weather slopes are shared and only the level is
    learned per place, which is the interesting question -- does weather push
    pollution around the same way everywhere?
    """
    cities = sorted(training)
    frame = pd.concat(
        [df.assign(_city=name) for name, df in training.items()], ignore_index=True
    )
    x = _pooled_matrix(frame, cities)
    pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipeline.fit(x, frame[TARGET].to_numpy(dtype=float))
    return pipeline, cities


def _pooled_matrix(frame: pd.DataFrame, cities: list[str]) -> np.ndarray:
    """Weather features, then one indicator column per city."""
    indicators = np.zeros((len(frame), len(cities)), dtype=float)
    if "_city" in frame:
        for i, city in enumerate(cities):
            indicators[:, i] = (frame["_city"] == city).to_numpy(dtype=float)
    return np.column_stack([frame[FEATURES].to_numpy(dtype=float), indicators])


def predict_pooled(pipeline: Pipeline, cities: list[str], frame: pd.DataFrame, city: str) -> np.ndarray:
    return pipeline.predict(_pooled_matrix(frame.assign(_city=city), cities))


def predictor_columns(
    timeline: pd.DataFrame,
    train: pd.DataFrame,
    alpha: float = 1.0,
    lead_days: int = 0,
    pooled: tuple[Pipeline, list[str]] | None = None,
    city: str | None = None,
) -> pd.DataFrame:
    """Every predictor's output for every row of the timeline, computed once.

    Lags run over the whole timeline rather than per window, so a window's first
    rows get a real previous value instead of a gap. They are positional: rows
    with a missing feature or target were dropped upstream, so "the previous
    row" is the previous *observation* and can be more than an hour back.
    """
    out = pd.DataFrame({"timestamp": timeline["timestamp"], "actual": timeline[TARGET]})

    frozen = build_pipeline(alpha).fit(train[FEATURES], train[TARGET])
    out["champion_frozen"] = frozen.predict(timeline[FEATURES])

    if pooled is not None and city is not None:
        pipeline, cities = pooled
        out["pooled_cities"] = predict_pooled(pipeline, cities, timeline, city)

    persistence_lag, seasonal_lag = autoregressive_lags(lead_days)
    out["persistence"] = timeline[TARGET].shift(persistence_lag).to_numpy(dtype=float)
    out["seasonal_naive"] = timeline[TARGET].shift(seasonal_lag).to_numpy(dtype=float)
    out["train_mean"] = float(train[TARGET].mean())

    by_hour = train.groupby(pd.to_datetime(train["timestamp"]).dt.hour)[TARGET].mean()
    out["climatology"] = pd.to_datetime(timeline["timestamp"]).dt.hour.map(by_hour).to_numpy(float)
    return out


def score_windows(
    columns: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    served_rmse: list[float] | None = None,
    lead_days: int = 0,
) -> list[Scored]:
    """Median RMSE per predictor across the monitoring windows.

    ``served_rmse`` is the loop's own per-window champion RMSE, read from the run
    it already logged rather than recomputed -- reimplementing the promotion
    policy here would risk benchmarking a loop that isn't the one that shipped.
    """
    # Derived from what was computed rather than hardcoded, so a
    # predictor that could not be built (the pooled model needs every city's
    # cache) is absent instead of scoring as NaN.
    names = [
        name
        for name in ("champion_frozen", "pooled_cities", "persistence",
                     "seasonal_naive", "climatology", "train_mean")
        if name in columns
    ]
    per_window: dict[str, list[float]] = {name: [] for name in names}

    stamps = pd.to_datetime(columns["timestamp"])
    for start, end in windows:
        slice_ = columns.loc[(stamps >= start) & (stamps < end)]
        if slice_.empty:
            continue
        actual = slice_["actual"].to_numpy(dtype=float)
        for name in names:
            per_window[name].append(_rmse(actual, slice_[name].to_numpy(dtype=float)))

    detail = detail_map(lead_days)
    scored = []
    if served_rmse:
        clean = [v for v in served_rmse if v is not None and not np.isnan(v)]
        if clean:
            scored.append(Scored("champion_served", detail["champion_served"],
                                 float(np.median(clean)), len(clean), False))
    for name in names:
        values = [v for v in per_window[name] if not np.isnan(v)]
        if values:
            scored.append(Scored(name, detail[name], float(np.median(values)),
                                 len(values), name in USES_PAST_TARGET))
    return sorted(scored, key=lambda s: s.median_rmse)


@dataclass
class AlphaSweep:
    best: float
    shipped: float
    curve: list[tuple[float, float]] = field(default_factory=list)
    n_splits: int = 0

    @property
    def penalty_pct(self) -> float:
        """How much worse the shipped alpha is than the best, in percent."""
        scores = dict(self.curve)
        best, shipped = scores.get(self.best), scores.get(self.shipped)
        if not best or best != best or shipped is None or shipped != shipped:
            return float("nan")
        return float((shipped / best - 1.0) * 100.0)


def tune_alpha(
    train: pd.DataFrame,
    grid: tuple[float, ...] = ALPHA_GRID,
    n_splits: int = 5,
    shipped: float = 1.0,
) -> AlphaSweep:
    """Forward-chaining CV over the training window.

    ``TimeSeriesSplit`` never lets a fold train on rows that come after the ones
    it scores, which a plain K-fold would. On autocorrelated hourly data a random
    fold leaks badly enough that every alpha looks fine.
    """
    if len(train) < n_splits + 1:
        return AlphaSweep(best=shipped, shipped=shipped)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    x, y = train[FEATURES], train[TARGET]

    curve: list[tuple[float, float]] = []
    for alpha in grid:
        folds = [
            _rmse(y.iloc[score].to_numpy(dtype=float),
                  build_pipeline(alpha).fit(x.iloc[fit], y.iloc[fit]).predict(x.iloc[score]))
            for fit, score in splitter.split(x)
        ]
        curve.append((float(alpha), float(np.mean(folds))))

    return AlphaSweep(min(curve, key=lambda p: p[1])[0], shipped, curve, n_splits)
