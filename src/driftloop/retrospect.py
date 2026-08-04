"""Scoring every model version on every window, after the fact.

The loop logs what *it* decided at the time: the champion in service, its RMSE,
and the ratio that drove the retrain. That is enough to run the loop and not
enough to judge it, because three questions need models scored on windows they
never served:

1. **Does an individual model decay?** The logged ``champion_rmse`` is one line
   across eight different champions, so no single model's decay is visible in
   it. Scoring one version across every window separates the models.
2. **Is the model any good?** RMSE has no scale. A number is only a verdict
   against an alternative you could have deployed instead.
3. **Did the promotion gate work, or did it overfit a seven-day exam?** The gate
   saw one holdout margin. Whether that margin was *delivered* over the
   challenger's service life is a different number, and it needs the replaced
   champion scored on windows it never served.

None of this needs refitting. ``log_and_register`` writes every version's
coefficients as tags in original feature units, so a version is reconstructable
from the registry alone as ``intercept + sum(coef_i * x_i)``. That is exact to
the six decimals the tags are written with, which ``tests/test_retrospect.py``
asserts against the loop's own logged RMSE.

## The skill baseline

``skill = 1 - rmse_model / rmse_climatology``: positive means the model beat the
baseline, zero means it matched it, negative means it lost.

The baseline is an hour-of-day mean over the ``CLIMATOLOGY_DAYS`` before the
window, and it is deliberately *not* derived from the champion's training
window. A baseline built from the champion's own training data inherits the
champion's staleness -- a winter-trained climatology also predicts high in July
-- so it cannot measure the thing we are trying to measure. The baseline has to
be independent of the model it is judging.

It is held to the same causality rule as everything else here: the reference
period ends ``forecast_lead_days`` before the window starts, so every hour it
averages was observable when the forecast for the window's first hour was
issued. It does get to see recent PM2.5, which the model never does, and that
is stated wherever the number is shown rather than buried -- it is the same
information-set distinction the benchmark card already draws. The justification
is that it is the alternative you could actually deploy: if a month-old daily
profile beats the model, the model is not paying for itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driftloop.config import DRIFT_FEATURES, FEATURES, TARGET, TIMESTAMP
from driftloop.data.base import DataSource

# How much history the climatology baseline averages over. A month is long
# enough to average out weather and short enough to still be the current season,
# which is the property that keeps this baseline honest across a full year.
CLIMATOLOGY_DAYS = 30


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


@dataclass(frozen=True)
class RegisteredModel:
    """A registered version, rebuilt from its coefficient tags.

    The tags hold slopes in *original* units (see ``model.effective_coefficients``),
    so prediction is a plain dot product and needs no scaler.
    """

    version: int
    coefficients: dict[str, float]
    intercept: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    baseline_rmse: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), self.intercept, dtype=float)
        for feature, coef in self.coefficients.items():
            out += coef * df[feature].to_numpy(dtype=float)
        return out

    def importance(self, reference: pd.DataFrame) -> dict[str, float]:
        """|slope| x the feature's spread: µg/m³ of prediction per 1-sd move.

        Raw slopes are per original unit and so are not comparable to each other
        -- a slope per hPa and a slope per W/m² are different questions. Scaling
        each by the feature's own standard deviation over a reference window puts
        them in one unit (µg/m³) and answers the comparable question: how much
        does this feature actually move the prediction?
        """
        return {
            feature: abs(coef) * float(np.std(reference[feature].to_numpy(dtype=float)))
            for feature, coef in self.coefficients.items()
        }


def registered_models(client, model_name: str) -> dict[int, RegisteredModel]:
    """Rebuild every registered version from the registry's coefficient tags.

    Versions whose tags predate coefficient logging are skipped rather than
    guessed at: a version we cannot reconstruct is one we must not score.
    """
    out: dict[int, RegisteredModel] = {}
    for mv in client.search_model_versions(f"name='{model_name}'"):
        tags = mv.tags or {}
        if "coef_intercept" not in tags or any(f"coef_{f}" not in tags for f in FEATURES):
            continue
        out[int(mv.version)] = RegisteredModel(
            version=int(mv.version),
            coefficients={f: float(tags[f"coef_{f}"]) for f in FEATURES},
            intercept=float(tags["coef_intercept"]),
            train_start=pd.to_datetime(tags.get("train_start")),
            train_end=pd.to_datetime(tags.get("train_end")),
            baseline_rmse=float(tags.get("baseline_rmse", "nan")),
        )
    return dict(sorted(out.items()))


def climatology_prediction(
    source: DataSource,
    window: pd.DataFrame,
    window_start: pd.Timestamp,
    lead_days: int,
    days: int = CLIMATOLOGY_DAYS,
) -> np.ndarray:
    """Hour-of-day means from the period ending ``lead_days`` before the window.

    Ending the reference early is what makes this usable as a baseline at all: a
    forecast for the window's first hour was issued ``lead_days`` earlier, so
    anything later than that was not available to whoever issued it.
    """
    reference_end = window_start - pd.Timedelta(lead_days, unit="D")
    reference = source.get_data(reference_end - pd.Timedelta(days, unit="D"), reference_end)
    if reference.empty:
        return np.full(len(window), np.nan)
    by_hour = reference.groupby(pd.to_datetime(reference[TIMESTAMP]).dt.hour)[TARGET].mean()
    hours = pd.to_datetime(window[TIMESTAMP]).dt.hour
    # A reference month missing an hour entirely leaves NaN, which _rmse drops
    # rather than silently scoring that hour against zero.
    return hours.map(by_hour).to_numpy(dtype=float)


def training_window_stats(source: DataSource, model: RegisteredModel) -> dict[str, dict[str, float]]:
    """Each feature's central range over the window a model was trained on.

    Drawn behind the feature series as a band, this is the covariate-drift claim
    stated physically: the band is what the model was shown, the line is what the
    world did afterwards, and a line leaving its band is the reason a retrain is
    justified. The 10th-90th percentile rather than the full range, because one
    freak hour should not widen the band to cover everything.
    """
    window = source.get_data(model.train_start, model.train_end + pd.Timedelta(1, unit="h"))
    stats: dict[str, dict[str, float]] = {}
    for feature in DRIFT_FEATURES:
        if feature not in window or window.empty:
            continue
        values = window[feature].to_numpy(dtype=float)
        stats[feature] = {
            "mean": float(np.mean(values)),
            "lo": float(np.percentile(values, 10)),
            "hi": float(np.percentile(values, 90)),
        }
    return stats


@dataclass
class Retrospective:
    """Every version scored on every window, plus what that says about the loop."""

    as_of: list[pd.Timestamp] = field(default_factory=list)
    climatology_rmse: list[float] = field(default_factory=list)
    champion_version: list[int] = field(default_factory=list)
    champion_rmse: list[float] = field(default_factory=list)
    champion_skill: list[float] = field(default_factory=list)
    # Per-window mean of each weather feature, and of the target. This is the
    # covariate story told in the units the features are actually measured in --
    # °C, m/s, hPa -- rather than as a PSI, which compresses "it got 20 °C
    # colder" into a number whose scale nobody can read.
    feature_means: dict[str, list[float]] = field(default_factory=dict)
    target_mean: list[float] = field(default_factory=list)
    # version -> its RMSE on *every* window, including ones it never served.
    # That is what turns "the champion line" into one decay curve per model.
    version_rmse: dict[int, list[float]] = field(default_factory=dict)
    version_skill: dict[int, list[float]] = field(default_factory=dict)
    promoted_at: dict[int, pd.Timestamp] = field(default_factory=dict)
    gate: list[dict] = field(default_factory=list)

    def skill_of(self, version: int) -> list[float]:
        return self.version_skill.get(version, [])


def retraining_value(result: Retrospective) -> dict[str, float | int]:
    """What retraining was worth, measured two ways because they answer differently.

    ``across_replay`` compares the median error of the champion the loop served
    against the median of the first champion held frozen, over every window. It
    is the product measure: what the whole arrangement delivered, including the
    weeks before the loop had retrained anything.

    ``when_it_acted`` compares the two models *window by window* and takes the
    median of the per-window ratio, over the windows where a retrained model was
    actually serving. It answers the different question of whether retraining
    helped when it happened.

    The first can be badly misleading on its own, which is why both are
    published. Comparing one median against another is unpaired, so where both
    distributions are dominated by the same seasonal swing the comparison mostly
    measures the season. Johannesburg does not promote anything until run 14 of
    20, so 70% of its windows have the two models identical, both medians land on
    the same value, and the headline reads 0.0% while every window where a
    retrained model was serving improved on the original.
    """
    frozen_version = min(result.version_rmse) if result.version_rmse else None
    if frozen_version is None or not result.champion_rmse:
        return {}

    served = np.asarray(result.champion_rmse, dtype=float)
    frozen = np.asarray(result.version_rmse[frozen_version], dtype=float)
    usable = ~(np.isnan(served) | np.isnan(frozen))
    if not usable.any():
        return {}

    out: dict[str, float | int] = {
        "across_replay": float((1 - np.median(served[usable]) / np.median(frozen[usable])) * 100),
        "windows": int(usable.sum()),
    }

    # Windows where the loop had replaced the original, so the two models differ
    # and the comparison is about retraining rather than about nothing.
    acted = usable & (served != frozen)
    out["acted_windows"] = int(acted.sum())
    if acted.any():
        ratio = served[acted] / frozen[acted]
        out["when_it_acted"] = float((1 - np.median(ratio)) * 100)
        out["win_rate"] = float((served[acted] < frozen[acted]).mean() * 100)
    return out


def _service_spans(as_of: list[pd.Timestamp], versions: list[int]) -> dict[int, tuple[int, int]]:
    """First and last window index each version was the serving champion for."""
    spans: dict[int, tuple[int, int]] = {}
    for i, v in enumerate(versions):
        first, _ = spans.get(v, (i, i))
        spans[v] = (first, i)
    return spans


def build(
    source: DataSource,
    runs: pd.DataFrame,
    models: dict[int, RegisteredModel],
    monitor_days: int,
    lead_days: int,
    promotion_margin: float,
) -> Retrospective:
    """Score every reconstructable version on every monitoring window.

    ``runs`` is the loop's own monitor-cycle log, one row per scheduled run, with
    ``as_of`` and ``champion_version`` columns, plus the holdout metrics the gate
    calibration reads. Everything else is derived, so this cannot disagree with
    what the loop did -- it only adds what the loop had no reason to compute at
    the time.
    """
    result = Retrospective()
    if runs.empty or not models:
        return result

    windows: list[pd.DataFrame] = []
    result.feature_means = {f: [] for f in DRIFT_FEATURES}
    for as_of in runs["as_of"]:
        start = as_of - pd.Timedelta(monitor_days, unit="D")
        window = source.get_data(start, as_of)
        windows.append(window)
        result.as_of.append(as_of)
        actual = window[TARGET].to_numpy(dtype=float)
        climatology = climatology_prediction(source, window, start, lead_days)
        result.climatology_rmse.append(_rmse(actual, climatology))
        result.target_mean.append(float(np.mean(actual)) if len(actual) else float("nan"))
        for feature in DRIFT_FEATURES:
            values = window[feature].to_numpy(dtype=float)
            result.feature_means[feature].append(float(np.mean(values)) if len(values) else float("nan"))

    for version, model in models.items():
        rmses, skills = [], []
        for window, climatology_rmse in zip(windows, result.climatology_rmse):
            actual = window[TARGET].to_numpy(dtype=float)
            value = _rmse(actual, model.predict(window))
            rmses.append(value)
            skills.append(
                float("nan")
                if not climatology_rmse or np.isnan(climatology_rmse) or np.isnan(value)
                else 1.0 - value / climatology_rmse
            )
        result.version_rmse[version] = rmses
        result.version_skill[version] = skills

    champion_versions = [int(v) for v in runs["champion_version"]]
    result.champion_version = champion_versions
    for i, version in enumerate(champion_versions):
        result.champion_rmse.append(result.version_rmse.get(version, [float("nan")] * len(windows))[i])
        result.champion_skill.append(result.version_skill.get(version, [float("nan")] * len(windows))[i])

    spans = _service_spans(result.as_of, champion_versions)
    for version, (first, _) in spans.items():
        result.promoted_at[version] = result.as_of[first]

    result.gate = _gate_calibration(result, runs, spans, champion_versions, promotion_margin)
    return result


def _gate_calibration(
    result: Retrospective,
    runs: pd.DataFrame,
    spans: dict[int, tuple[int, int]],
    champion_versions: list[int],
    promotion_margin: float,
) -> list[dict]:
    """What the seven-day exam promised against what the winner went on to do.

    The gate promotes on a single holdout window. That margin is in-sample *for
    the decision*: it is the number the decision was made on. The out-of-sample
    check is what the new champion actually delivered over its service life,
    measured against the champion it replaced scored on those same windows --
    which is exactly the counterfactual the loop never had a reason to compute.

    A gate that works shows the two rising together. A gate overfitting a short
    exam shows exam margins scattered against delivered ones.
    """
    out: list[dict] = []
    for version, (first, last) in sorted(spans.items()):
        if first == 0:  # the bootstrap champion replaced nobody
            continue
        replaced = champion_versions[first - 1]
        if replaced == version or version not in result.version_rmse or replaced not in result.version_rmse:
            continue

        row = runs.iloc[first]
        champion_holdout = row.get("metrics.champion_rmse_holdout")
        challenger_holdout = row.get("metrics.challenger_rmse")
        if champion_holdout is None or pd.isna(champion_holdout) or pd.isna(challenger_holdout):
            continue

        # From the run *after* the promotion. The monitor window at the promotion
        # itself is [as_of-14d, as_of), whose first half is inside the window the
        # challenger trained on -- scoring it there would be marking its own
        # homework. One step later the window is [as_of-7d, as_of+7d), which the
        # challenger never trained on, so the comparison is clean.
        # At least one window even for a champion replaced the following week:
        # scoring it one step past its retirement is a counterfactual, not a
        # leak, and dropping those promotions would quietly bias the calibration
        # toward the long-serving models that are the interesting failures.
        served = slice(first + 1, max(last + 1, first + 2))
        new_rmse = [v for v in result.version_rmse[version][served] if not np.isnan(v)]
        old_rmse = [v for v in result.version_rmse[replaced][served] if not np.isnan(v)]
        if not new_rmse or not old_rmse:
            continue

        new_median, old_median = float(np.median(new_rmse)), float(np.median(old_rmse))
        out.append(
            {
                "version": version,
                "replaced": replaced,
                "as_of": result.as_of[first],
                # What the gate saw: the challenger's edge on the holdout exam.
                "exam_margin": float(1.0 - challenger_holdout / champion_holdout)
                if champion_holdout
                else float("nan"),
                # What it delivered: the same edge over the windows it served,
                # against the model it displaced, scored on those windows too.
                "delivered_margin": float(1.0 - new_median / old_median) if old_median else float("nan"),
                "weeks_served": last - first + 1,
                "required_margin": promotion_margin,
            }
        )
    return out
