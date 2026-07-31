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
from sklearn.model_selection import TimeSeriesSplit

from driftloop.config import FEATURES, TARGET
from driftloop.model import build_pipeline

# Swept on a log grid: alpha's effect is multiplicative, so linear steps would
# spend most of their samples at the insensitive end.
ALPHA_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)

# Predictors that get to look at past values of the target. The Ridge does not.
USES_PAST_TARGET = {"persistence", "seasonal_naive"}

DETAIL = {
    "champion_served": "the model the loop actually served, retrained as needed",
    "champion_frozen": "the first champion, never retrained",
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


def predictor_columns(timeline: pd.DataFrame, train: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    """Every predictor's output for every row of the timeline, computed once.

    Lags run over the whole timeline rather than per window, so a window's first
    rows get a real previous value instead of a gap. They are positional: rows
    with a missing feature or target were dropped upstream, so "the previous
    row" is the previous *observation* and can be more than an hour back.
    """
    out = pd.DataFrame({"timestamp": timeline["timestamp"], "actual": timeline[TARGET]})

    frozen = build_pipeline(alpha).fit(train[FEATURES], train[TARGET])
    out["champion_frozen"] = frozen.predict(timeline[FEATURES])

    out["persistence"] = timeline[TARGET].shift(1).to_numpy(dtype=float)
    out["seasonal_naive"] = timeline[TARGET].shift(24).to_numpy(dtype=float)
    out["train_mean"] = float(train[TARGET].mean())

    by_hour = train.groupby(pd.to_datetime(train["timestamp"]).dt.hour)[TARGET].mean()
    out["climatology"] = pd.to_datetime(timeline["timestamp"]).dt.hour.map(by_hour).to_numpy(float)
    return out


def score_windows(
    columns: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    served_rmse: list[float] | None = None,
) -> list[Scored]:
    """Median RMSE per predictor across the monitoring windows.

    ``served_rmse`` is the loop's own per-window champion RMSE, read from the run
    it already logged rather than recomputed -- reimplementing the promotion
    policy here would risk benchmarking a loop that isn't the one that shipped.
    """
    names = ["champion_frozen", "persistence", "seasonal_naive", "climatology", "train_mean"]
    per_window: dict[str, list[float]] = {name: [] for name in names}

    stamps = pd.to_datetime(columns["timestamp"])
    for start, end in windows:
        slice_ = columns.loc[(stamps >= start) & (stamps < end)]
        if slice_.empty:
            continue
        actual = slice_["actual"].to_numpy(dtype=float)
        for name in names:
            per_window[name].append(_rmse(actual, slice_[name].to_numpy(dtype=float)))

    scored = []
    if served_rmse:
        clean = [v for v in served_rmse if v is not None and not np.isnan(v)]
        if clean:
            scored.append(Scored("champion_served", DETAIL["champion_served"],
                                 float(np.median(clean)), len(clean), False))
    for name in names:
        values = [v for v in per_window[name] if not np.isnan(v)]
        if values:
            scored.append(Scored(name, DETAIL[name], float(np.median(values)),
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
